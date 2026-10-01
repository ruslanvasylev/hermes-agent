"""Protected first request through a real AIAgent and SDK, with no network."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.native_request import (
    NativeAdmissionError, NativeExecutionResult, NativePreparationResult,
    NativeRequestTurn,
)


def _agent(tmp_path, suffix):
    from run_agent import AIAgent

    home = tmp_path / suffix
    home.mkdir()
    with patch.dict("os.environ", {"HERMES_HOME": str(home)}):
        agent = AIAgent(
            model="test-model",
            provider="openai-codex",
            api_key="fixture-only",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
            session_id=f"native-{suffix}",
        )
    agent.api_mode = "codex_responses"
    agent._cached_system_prompt = "Fixed native identity"
    agent._api_max_retries = 3
    agent._fallback_chain = [object()]
    return agent


def _sse(model="test-model", *, terminal="response.completed"):
    events = [
        {"type": "response.output_text.delta", "delta": "hello"},
        {"type": terminal, "response": {"id": "resp_native", "model": model,
                                    "status": "completed" if terminal == "response.completed" else "failed",
                                    "output": []}},
    ]
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def _run(agent, turn):
    return agent.run_conversation(
        "Review the fixed source", conversation_history=[], task_id="same-task",
        native_turn=turn,
    )


def test_real_first_request_preparation_and_guarded_dispatch_are_identical(tmp_path):
    prepared = _run(_agent(tmp_path, "prep"), NativeRequestTurn("prepare"))
    assert isinstance(prepared, NativePreparationResult)
    assert prepared.prepared.method == "POST"
    delegated = []

    def delegate(request):
        delegated.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse(), request=request,
        )

    turn = NativeRequestTurn(
        "dispatch", prepared=prepared.prepared,
        delegate_factory=lambda: httpx.MockTransport(delegate),
    )
    result = _run(_agent(tmp_path, "dispatch"), turn)
    assert isinstance(result, NativeExecutionResult)
    assert len(delegated) == 1
    assert result.attempt.prepared == prepared.prepared
    assert result.attempt.provider_response_id == "resp_native"
    assert result.attempt.stream_exhausted and result.attempt.stream_closed
    assert result.outer_callback_count == 1
    assert result.outer_retry_count == result.compression_count == 0
    assert not result.fallback_activated


def test_mutated_dispatch_body_is_vetoed_before_network(tmp_path):
    prepared = _run(_agent(tmp_path, "prep-mutate"), NativeRequestTurn("prepare"))
    delegated = []
    turn = NativeRequestTurn(
        "dispatch", prepared=prepared.prepared,
        delegate_factory=lambda: httpx.MockTransport(lambda req: delegated.append(req)),
    )
    agent = _agent(tmp_path, "dispatch-mutate")
    agent._cached_system_prompt = "Changed identity"
    with pytest.raises(Exception):
        _run(agent, turn)
    assert delegated == []
    assert turn.dispatch_policy is not None
    assert turn.dispatch_policy.guard.send_count == 1
    assert turn.dispatch_policy.guard.physical_attempt_id is None


def test_connect_error_does_not_retry_or_activate_fallback(tmp_path):
    prepared = _run(_agent(tmp_path, "prep-fail"), NativeRequestTurn("prepare"))
    sends = []

    def fail(request):
        sends.append(request)
        raise httpx.ConnectError("fixture connect failure", request=request)

    turn = NativeRequestTurn(
        "dispatch", prepared=prepared.prepared,
        delegate_factory=lambda: httpx.MockTransport(fail),
    )
    agent = _agent(tmp_path, "dispatch-fail")
    agent._try_activate_fallback = MagicMock(return_value=True)
    agent._compress_context = MagicMock()
    with pytest.raises(Exception):
        _run(agent, turn)
    assert len(sends) == 1
    agent._try_activate_fallback.assert_not_called()
    agent._compress_context.assert_not_called()


def test_failed_terminal_cannot_return_native_result(tmp_path):
    prepared = _run(_agent(tmp_path, "prep-terminal"), NativeRequestTurn("prepare"))
    sends = []

    def delegate(request):
        sends.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse(terminal="response.failed"), request=request,
        )

    turn = NativeRequestTurn("dispatch", prepared=prepared.prepared,
                             delegate_factory=lambda: httpx.MockTransport(delegate))
    with pytest.raises(NativeAdmissionError):
        _run(_agent(tmp_path, "dispatch-terminal"), turn)
    assert len(sends) == 1


def test_turn_policy_is_one_use_and_never_stored_on_agent(tmp_path):
    agent = _agent(tmp_path, "reuse")
    turn = NativeRequestTurn("prepare")
    _run(agent, turn)
    with pytest.raises(NativeAdmissionError, match="native_turn_reused"):
        _run(agent, turn)
    assert not hasattr(agent, "_native_dispatch_policy")


def test_request_middleware_mutation_is_vetoed_before_delegate(tmp_path, monkeypatch):
    from hermes_cli import middleware

    prepared = _run(_agent(tmp_path, "prep-request-mw"), NativeRequestTurn("prepare"))
    sends = []
    turn = NativeRequestTurn(
        "dispatch", prepared=prepared.prepared,
        delegate_factory=lambda: httpx.MockTransport(lambda req: sends.append(req)),
    )
    monkeypatch.setattr(middleware, "_has_middleware", lambda kind: kind == "llm_request")
    monkeypatch.setattr(
        middleware, "_invoke_middleware",
        lambda kind, **kwargs: [{"request": {**kwargs["request"], "instructions": "changed"}}],
    )
    with pytest.raises(Exception):
        _run(_agent(tmp_path, "dispatch-request-mw"), turn)
    assert sends == []
    assert turn.dispatch_policy.guard.physical_attempt_id is None


def test_execution_middleware_mutation_and_duplicate_are_fatal(tmp_path, monkeypatch):
    from hermes_cli import middleware

    prepared = _run(_agent(tmp_path, "prep-exec-mw"), NativeRequestTurn("prepare"))
    sends = []

    def delegate(request):
        sends.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_sse(), request=request)

    def mutate(request, next_call, **_):
        return next_call({**request, "instructions": "changed after request middleware"})

    monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda kind: [mutate])
    turn = NativeRequestTurn("dispatch", prepared=prepared.prepared,
                             delegate_factory=lambda: httpx.MockTransport(delegate))
    with pytest.raises(Exception):
        _run(_agent(tmp_path, "dispatch-exec-mw"), turn)
    assert sends == []

    def duplicate(request, next_call, **_):
        first = next_call(request)
        try:
            next_call(request)
        except RuntimeError:
            pass
        return first

    monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda kind: [duplicate])
    turn = NativeRequestTurn("dispatch", prepared=prepared.prepared,
                             delegate_factory=lambda: httpx.MockTransport(delegate))
    with pytest.raises(RuntimeError, match="more than once"):
        _run(_agent(tmp_path, "dispatch-duplicate"), turn)
    assert len(sends) == 1


def test_compression_trigger_stops_before_capture_or_compress(tmp_path):
    agent = _agent(tmp_path, "compression")
    compressor = agent.context_compressor
    compressor.protect_first_n = 0
    compressor.protect_last_n = 0
    compressor.threshold_tokens = 1
    compressor.last_prompt_tokens = 0
    compressor.should_defer_preflight_to_real_usage = lambda _tokens: False
    compressor.get_active_compression_failure_cooldown = lambda: None
    compressor.should_compress = lambda _tokens: True
    agent._prepare_native_request = MagicMock()
    agent._compress_context = MagicMock()
    with pytest.raises(NativeAdmissionError, match="native_turn_preflight_compression_required"):
        _run(agent, NativeRequestTurn("prepare"))
    agent._prepare_native_request.assert_not_called()
    agent._compress_context.assert_not_called()
    assert compressor.last_prompt_tokens == 0


def test_execution_middleware_cannot_replace_stream_owned_attempt(tmp_path, monkeypatch):
    from hermes_cli import middleware

    prepared = _run(_agent(tmp_path, "prep-forged-attempt"), NativeRequestTurn("prepare"))
    sends = []

    def delegate(request):
        sends.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse(), request=request,
        )

    def replace_attempt(request, next_call, **_):
        response = next_call(request)
        actual = response.native_attempt
        response.native_attempt = replace(
            actual,
            provider_response_id="forged-provider-id",
            prepared=replace(actual.prepared, body_sha256="f" * 64),
            terminal_event_count=9,
            tool_call_count=4,
        )
        return response

    monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda kind: [replace_attempt])
    turn = NativeRequestTurn(
        "dispatch", prepared=prepared.prepared,
        delegate_factory=lambda: httpx.MockTransport(delegate),
    )
    with pytest.raises(NativeAdmissionError, match="native_execution_observation_invalid"):
        _run(_agent(tmp_path, "dispatch-forged-attempt"), turn)
    assert len(sends) == 1
