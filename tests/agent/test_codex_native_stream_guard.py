"""Strict Codex stream evidence is opt-in and cannot be inferred from text."""

from types import SimpleNamespace

import httpx
import pytest
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from agent.codex_runtime import _consume_codex_event_stream, run_codex_stream
from agent.native_request import DispatchPolicy, GuardedTransport, NativeAdmissionError, PreparedRequest


def _terminal(kind="response.completed", *, model="test-model", rid="resp_1", status="completed"):
    return {"type": kind, "response": {"id": rid, "model": model, "status": status}}


class _Stream:
    def __init__(self, events, *, close_error=False):
        self.events = events
        self.close_error = close_error
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError("close failed")


class _Agent:
    _interrupt_requested = False

    def _fire_stream_delta(self, text):
        pass

    def _fire_reasoning_delta(self, text):
        pass

    def _touch_activity(self, text):
        pass

    def _client_log_context(self):
        return "fixture"


def _policy_client(stream):
    prepared = PreparedRequest("POST", "https://example.invalid/v1/responses", "0" * 64, 1)
    guard = GuardedTransport(httpx.MockTransport(lambda request: httpx.Response(200)), prepared)
    guard.send_count = 1
    guard.physical_attempt_id = "physical_1"
    http_client = httpx.Client(transport=guard, trust_env=False)
    client = SimpleNamespace(_client=http_client, max_retries=0,
                             responses=SimpleNamespace(create=lambda **kwargs: stream))
    return DispatchPolicy(guard), client, http_client


def test_guarded_completed_stream_records_real_terminal_and_close():
    stream = _Stream([{"type": "response.output_text.delta", "delta": "hello"}, _terminal()])
    policy, client, http_client = _policy_client(stream)
    try:
        final = run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert final.output_text == "hello"
    assert final.native_attempt.provider_response_id == "resp_1"
    assert final.native_attempt.provider_model == "test-model"
    assert final.native_attempt.stream_exhausted and final.native_attempt.stream_closed
    assert final.native_attempt.terminal_event_count == 1
    assert stream.closed


@pytest.mark.parametrize("item", [
    {"type": "function_call", "name": "shell", "call_id": "c1"},
    ResponseFunctionToolCall(type="function_call", name="shell", call_id="c1", arguments="{}"),
    {"type": "custom_tool_call"},
    {"type": "web_search_call"},
    {"type": "file_search_call"},
    {"type": "code_interpreter_call"},
    {"type": "image_generation_call"},
    {"type": "computer_call"},
    {"type": "local_shell_call"},
    {"type": "mcp_call"},
    {"type": "mcp_list_tools"},
    {"type": "function_shell_call"},
    {"type": "apply_patch_call"},
    {"type": "compaction"},
    {"type": "unrecognized_executable"},
])
def test_terminal_only_tool_output_never_yields_native_attempt(item):
    terminal = _terminal()
    terminal["response"]["output"] = [item]
    stream = _Stream([terminal])
    policy, client, http_client = _policy_client(stream)
    try:
        with pytest.raises(NativeAdmissionError, match="native_tool_call"):
            run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert stream.closed


@pytest.mark.parametrize("item", [
    {"type": "message", "role": "assistant", "content": []},
    SimpleNamespace(type="reasoning", summary=[]),
])
def test_terminal_only_safe_output_is_admitted(item):
    terminal = _terminal()
    terminal["response"]["output"] = [item]
    stream = _Stream([terminal])
    policy, client, http_client = _policy_client(stream)
    try:
        final = run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert final.native_attempt.tool_call_count == 0


def test_streamed_non_function_tool_item_is_rejected():
    stream = _Stream([{"type": "response.output_item.done", "item": {"type": "mcp_call"}}, _terminal()])
    policy, client, http_client = _policy_client(stream)
    try:
        with pytest.raises(NativeAdmissionError, match="native_tool_call"):
            run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert stream.closed


@pytest.mark.parametrize("events,reason", [
    ([{"type": "response.output_text.delta", "delta": "hello"}], "native_terminal_missing"),
    ([_terminal(), _terminal()], "native_duplicate_terminal"),
    ([_terminal(model="")], "native_provider_model_mismatch"),
    ([_terminal(model="other")], "native_provider_model_mismatch"),
    ([_terminal(rid="")], "native_provider_response_id_missing"),
    ([_terminal("response.incomplete", status="incomplete")], "native_terminal_not_completed"),
    ([_terminal("response.failed", status="failed")], "native_terminal_not_completed"),
    ([{"type": "response.output_item.done", "item": {"type": "function_call"}}, _terminal()], "native_tool_call"),
])
def test_bad_terminal_never_yields_native_attempt(events, reason):
    stream = _Stream(events)
    policy, client, http_client = _policy_client(stream)
    try:
        with pytest.raises(NativeAdmissionError, match=reason):
            run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert stream.closed


def test_guarded_connect_error_has_no_inner_retry():
    policy, client, http_client = _policy_client(_Stream([]))
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise httpx.ConnectError("offline")

    client.responses.create = fail
    try:
        with pytest.raises(NativeAdmissionError, match="native_stream_connect_failed"):
            run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert len(calls) == 1


def test_guarded_midstream_error_and_close_failure():
    def broken():
        yield {"type": "response.output_text.delta", "delta": "partial"}
        raise httpx.ReadTimeout("stalled")

    stream = _Stream(broken())
    policy, client, http_client = _policy_client(stream)
    try:
        with pytest.raises(NativeAdmissionError, match="native_stream_transport_failed"):
            run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    assert stream.closed

    stream = _Stream([_terminal()], close_error=True)
    policy, client, http_client = _policy_client(stream)
    try:
        with pytest.raises(NativeAdmissionError, match="native_stream_close_failed"):
            run_codex_stream(_Agent(), {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()


def test_interrupt_and_ordinary_stream_regression():
    agent = _Agent()
    agent._interrupt_requested = True
    stream = _Stream([_terminal()])
    policy, client, http_client = _policy_client(stream)
    try:
        with pytest.raises(NativeAdmissionError, match="native_stream_interrupted"):
            run_codex_stream(agent, {"model": "test-model"}, client=client, native_policy=policy)
    finally:
        http_client.close()
    ordinary = _consume_codex_event_stream(
        [{"type": "response.output_text.delta", "delta": "partial"}], model="test-model")
    assert ordinary.status == "completed" and ordinary.output_text == "partial"
