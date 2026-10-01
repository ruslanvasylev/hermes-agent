"""Real SDK serialization and request-local no-network admission tests."""

from dataclasses import FrozenInstanceError
import hashlib
import json

import httpx
import openai
import pytest

from agent.native_request import (
    DispatchPolicy,
    GuardedTransport,
    NativeAdmissionError,
    capture_responses_request,
)


KWARGS = {"model": "test-model", "instructions": "identity", "input": "hello", "store": False}


def _client(http_client):
    return openai.OpenAI(
        api_key="fixture-only",
        base_url="https://example.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )


def test_capture_uses_real_sdk_serialization_without_network():
    prepared = capture_responses_request(_client, KWARGS)
    assert prepared.method == "POST"
    assert prepared.url == "https://example.invalid/v1/responses"
    assert prepared.body_bytes == len(prepared.body) > len(KWARGS["input"])
    assert prepared.body_sha256 == hashlib.sha256(prepared.body).hexdigest()
    assert json.loads(prepared.body)["model"] == KWARGS["model"]
    assert "body=" not in repr(prepared)
    with pytest.raises(FrozenInstanceError):
        prepared.body = b"changed"


def test_capture_exposes_effective_sdk_extra_body_fields():
    prepared = capture_responses_request(
        _client,
        {**KWARGS, "max_output_tokens": 16,
         "extra_body": {"model": "overridden-model", "max_output_tokens": 32}},
    )
    payload = json.loads(prepared.body)
    assert payload["model"] == "overridden-model"
    assert payload["max_output_tokens"] == 32
    assert prepared.body_bytes == len(prepared.body)
    assert prepared.body_sha256 == hashlib.sha256(prepared.body).hexdigest()


def test_capture_rejects_factory_that_bypasses_no_network_client():
    def unsafe_factory(_http_client):
        return openai.OpenAI(api_key="fixture-only", base_url="https://example.invalid/v1", max_retries=0)

    with pytest.raises(NativeAdmissionError, match="native_capture_client_unsafe"):
        capture_responses_request(unsafe_factory, KWARGS)


def test_guard_exact_send_then_replay_is_vetoed():
    prepared = capture_responses_request(_client, KWARGS)
    delegated = []

    def handler(request):
        delegated.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b"data: [DONE]\n\n", request=request)

    guard = GuardedTransport(httpx.MockTransport(handler), prepared)
    with httpx.Client(transport=guard, trust_env=False) as http_client:
        client = _client(http_client)
        DispatchPolicy(guard).require_client(client)
        stream = client.responses.create(**{**KWARGS, "stream": True})
        stream.close()
        assert guard.physical_attempt_id and len(delegated) == 1
        with pytest.raises(openai.APIConnectionError) as exc:
            client.responses.create(**{**KWARGS, "stream": True})
        assert isinstance(exc.value.__cause__, NativeAdmissionError)
    assert guard.send_count == 2 and len(delegated) == 1


def test_body_mutation_rejected_before_delegate():
    prepared = capture_responses_request(_client, KWARGS)
    delegated = []
    guard = GuardedTransport(httpx.MockTransport(lambda request: delegated.append(request)), prepared)
    with httpx.Client(transport=guard, trust_env=False) as http_client:
        client = _client(http_client)
        with pytest.raises(openai.APIConnectionError) as exc:
            client.responses.create(**{**KWARGS, "input": "changed", "stream": True})
        assert isinstance(exc.value.__cause__, NativeAdmissionError)
    assert guard.physical_attempt_id is None
    assert not delegated


def test_extra_body_drift_rejected_before_delegate():
    prepared = capture_responses_request(_client, KWARGS)
    delegated = []
    guard = GuardedTransport(httpx.MockTransport(lambda request: delegated.append(request)), prepared)
    with httpx.Client(transport=guard, trust_env=False) as http_client:
        client = _client(http_client)
        with pytest.raises(openai.APIConnectionError) as exc:
            client.responses.create(**{**KWARGS, "extra_body": {"model": "other-model"}, "stream": True})
        assert isinstance(exc.value.__cause__, NativeAdmissionError)
    assert guard.physical_attempt_id is None
    assert not delegated


def test_proxy_mount_bypass_is_rejected():
    prepared = capture_responses_request(_client, KWARGS)
    guard = GuardedTransport(httpx.MockTransport(lambda request: httpx.Response(200)), prepared)
    with httpx.Client(transport=guard, mounts={"https://": httpx.MockTransport(lambda request: httpx.Response(200))}, trust_env=False) as http_client:
        with pytest.raises(NativeAdmissionError, match="bypass_mount"):
            DispatchPolicy(guard).require_client(_client(http_client))
