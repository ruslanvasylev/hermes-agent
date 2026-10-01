"""Opt-in, request-local evidence for a single Responses API HTTP attempt.

Capture is deliberately provisional.  Only a guarded transport may create a
physical attempt, and neither object signs or admits a downstream result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import threading
from typing import Any, Callable, Literal
from uuid import uuid4

import httpx


_SUPPORTED_OPENAI_SDK_VERSION = "2.24.0"


class NativeAdmissionError(RuntimeError):
    """A protected request failed closed; callers must not retry or fall back."""


@dataclass(frozen=True)
class NativeTurnContext:
    """Opt-in policy for the once-per-turn prologue, never stored on an agent.

    A caller passes a fresh context directly to ``build_turn_context`` for
    each provisional preparation or guarded dispatch.  It carries no provider
    authority and cannot turn a captured request into an admitted attempt.
    """

    phase: Literal["prepare", "dispatch"]

    def __post_init__(self) -> None:
        if self.phase not in ("prepare", "dispatch"):
            raise NativeAdmissionError("native_turn_phase_invalid")


@dataclass(frozen=True)
class PreparedRequest:
    method: str
    url: str
    body: bytes = field(repr=False)
    body_sha256: str
    body_bytes: int


@dataclass(frozen=True)
class AttemptObservation:
    physical_attempt_id: str
    prepared: PreparedRequest
    provider_response_id: str
    provider_model: str
    response_status: str
    terminal_event_count: int
    stream_exhausted: bool
    stream_closed: bool
    tool_call_count: int


def _prepared(request: httpx.Request) -> PreparedRequest:
    # httpx.Request.read() buffers a streaming body.  The SDK's openapi_dumps
    # body has already been serialized by the time this transport sees it.
    body = request.read()
    return PreparedRequest(
        method=request.method,
        url=str(request.url),
        body=body,
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_bytes=len(body),
    )


class CaptureTransport(httpx.BaseTransport):
    """Capture one SDK-built HTTP request; never open a network connection."""

    def __init__(self) -> None:
        self.prepared: PreparedRequest | None = None
        self.send_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.send_count += 1
        if self.send_count != 1:
            raise NativeAdmissionError("native_capture_multiple_sends")
        self.prepared = _prepared(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
            request=request,
        )


def capture_responses_request(
    create_client: Callable[[httpx.Client], Any], api_kwargs: dict[str, Any]
) -> PreparedRequest:
    """Run the real SDK serializer against a no-network HTTPX transport.

    ``create_client`` must apply the same SDK version, base URL, default headers
    and request options used by the later request-local client, with retries 0.
    A live transport must compare again at dispatch; this capture never grants
    permission to send.
    """
    import openai

    if openai.__version__ != _SUPPORTED_OPENAI_SDK_VERSION:
        raise NativeAdmissionError("native_sdk_version_unsupported")
    transport = CaptureTransport()
    try:
        with httpx.Client(transport=transport, trust_env=False) as http_client:
            client = create_client(http_client)
            try:
                if (getattr(client, "_client", None) is not http_client
                        or getattr(client, "max_retries", None) != 0
                        or any(value is not None for value in getattr(http_client, "_mounts", {}).values())):
                    raise NativeAdmissionError("native_capture_client_unsafe")
                stream = client.responses.create(**{**api_kwargs, "stream": True})
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            finally:
                close_client = getattr(client, "close", None)
                if callable(close_client):
                    close_client()
    except NativeAdmissionError:
        raise
    except Exception as exc:
        raise NativeAdmissionError("native_capture_failed") from exc
    if transport.send_count != 1 or transport.prepared is None:
        raise NativeAdmissionError("native_capture_missing_request")
    return transport.prepared


class GuardedTransport(httpx.BaseTransport):
    """Veto drift and a second attempt immediately before the delegate send.

    The caller supplies the already configured proxy/TLS/socket delegate.  No
    request may route through another HTTPX mount or transport in strict mode.
    """

    def __init__(self, delegate: httpx.BaseTransport, prepared: PreparedRequest) -> None:
        self.delegate = delegate
        self.prepared = prepared
        self.physical_attempt_id: str | None = None
        self.send_count = 0
        self._lock = threading.Lock()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.send_count += 1
            if self.send_count != 1:
                raise NativeAdmissionError("native_second_send")
            if _prepared(request) != self.prepared:
                raise NativeAdmissionError("native_request_drift")
            self.physical_attempt_id = uuid4().hex
        return self.delegate.handle_request(request)

    def close(self) -> None:
        self.delegate.close()


@dataclass(frozen=True)
class DispatchPolicy:
    guard: GuardedTransport
    _observation: AttemptObservation | None = field(default=None, init=False, repr=False, compare=False)
    _observation_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False,
    )

    @property
    def observation(self) -> AttemptObservation | None:
        """The original stream-owned record, kept off the mutable response."""
        return self._observation

    def require_client(self, client: Any) -> None:
        http_client = getattr(client, "_client", None)
        if not isinstance(http_client, httpx.Client) or getattr(http_client, "_transport", None) is not self.guard:
            raise NativeAdmissionError("native_guard_not_installed")
        # HTTPX proxy mounts can bypass Client._transport.  Protected clients
        # must use a single explicitly configured delegate beneath the guard.
        mounts = getattr(http_client, "_mounts", {})
        if any(transport is not None for transport in mounts.values()):
            raise NativeAdmissionError("native_guard_bypass_mount")
        if getattr(client, "max_retries", None) != 0:
            raise NativeAdmissionError("native_sdk_retry_enabled")

    def observe(self, final: Any) -> AttemptObservation:
        with self._observation_lock:
            if self._observation is not None:
                raise NativeAdmissionError("native_observation_reissued")
            guard = self.guard
            if guard.send_count != 1 or not guard.physical_attempt_id:
                raise NativeAdmissionError("native_physical_attempt_missing")
            observation = AttemptObservation(
                physical_attempt_id=guard.physical_attempt_id,
                prepared=guard.prepared,
                provider_response_id=final.id,
                provider_model=final.model,
                response_status=final.status,
                terminal_event_count=final.native_terminal_event_count,
                stream_exhausted=final.native_stream_exhausted,
                stream_closed=final.native_stream_closed,
                tool_call_count=final.native_tool_call_count,
            )
            object.__setattr__(self, "_observation", observation)
            return observation


@dataclass
class NativeRequestTurn:
    """One-use request policy passed explicitly through a protected turn."""

    phase: Literal["prepare", "dispatch"]
    prepared: PreparedRequest | None = None
    delegate_factory: Callable[[], httpx.BaseTransport] | None = None
    dispatch_policy: DispatchPolicy | None = field(default=None, init=False)
    _claimed: bool = field(default=False, init=False, repr=False)

    def claim(self) -> None:
        if self._claimed:
            raise NativeAdmissionError("native_turn_reused")
        if self.phase == "prepare":
            if self.prepared is not None or self.delegate_factory is not None:
                raise NativeAdmissionError("native_prepare_input_invalid")
        elif self.phase == "dispatch":
            if not isinstance(self.prepared, PreparedRequest):
                raise NativeAdmissionError("native_dispatch_preparation_missing")
        else:
            raise NativeAdmissionError("native_turn_phase_invalid")
        self._claimed = True


@dataclass(frozen=True)
class NativePreparationResult:
    prepared: PreparedRequest


@dataclass(frozen=True)
class NativeExecutionResult:
    response: Any
    attempt: AttemptObservation
    outer_callback_count: int
    outer_retry_count: int
    fallback_activated: bool
    compression_count: int
