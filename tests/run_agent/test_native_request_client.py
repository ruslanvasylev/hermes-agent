"""Request-local native SDK client construction rejects bypass routes."""

import httpx
import pytest

from agent.native_request import NativeAdmissionError, NativeRequestTurn, PreparedRequest


def _prepared():
    return PreparedRequest("POST", "https://example.invalid/v1/responses", "0" * 64, 1)


def test_native_client_rejects_custom_http_client_without_send():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    sends = []
    agent._client_kwargs = {"api_key": "fixture-only", "base_url": "https://example.invalid/v1",
                            "http_client": httpx.Client(
                                transport=httpx.MockTransport(lambda r: sends.append(r)),
                                mounts={"https://": httpx.MockTransport(lambda r: sends.append(r))},
                            )}
    agent._openai_client_lock = lambda: __import__("contextlib").nullcontext()
    with pytest.raises(NativeAdmissionError, match="native_custom_client_unsupported"):
        agent._native_openai_client_kwargs({})
    assert sends == []
    agent._client_kwargs["http_client"].close()


def test_native_client_rejects_unsupported_delegate_before_send():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent._client_kwargs = {"api_key": "fixture-only", "base_url": "https://example.invalid/v1"}
    agent._openai_client_lock = lambda: __import__("contextlib").nullcontext()
    turn = NativeRequestTurn("dispatch", prepared=_prepared(), delegate_factory=lambda: object())
    with pytest.raises(NativeAdmissionError, match="native_transport_unsupported"):
        agent._create_native_request_client(turn, {})
