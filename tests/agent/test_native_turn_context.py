"""Opt-in native prologue behavior on the actual first-turn owner."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.native_request import NativeAdmissionError, NativeTurnContext
from tests.agent.test_turn_context import _FakeAgent, _build


@pytest.fixture(autouse=True)
def _no_external_context():
    with (
        patch("agent.auxiliary_client.set_runtime_main"),
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False),
    ):
        yield


def test_preparation_keeps_actual_message_assembly_without_persisting():
    ordinary = _FakeAgent()
    native = _FakeAgent()
    history = [{"role": "assistant", "content": "prior"}]
    ordinary_result = _build(ordinary, conversation_history=history)
    native._restore_primary_runtime = MagicMock()
    native._ensure_db_session = MagicMock()
    native._persist_session = MagicMock()

    prepared = _build(
        native,
        conversation_history=history,
        native_context=NativeTurnContext("prepare"),
    )

    assert prepared.messages == ordinary_result.messages
    assert prepared.active_system_prompt == ordinary_result.active_system_prompt
    native._restore_primary_runtime.assert_not_called()
    native._ensure_db_session.assert_not_called()
    native._persist_session.assert_not_called()


def test_dispatch_keeps_legitimate_inbound_persistence():
    agent = _FakeAgent()
    agent._restore_primary_runtime = MagicMock()
    _build(agent, native_context=NativeTurnContext("dispatch"))
    agent._restore_primary_runtime.assert_not_called()
    assert agent._ensure_db_prompt_at_call == "SYSTEM"
    assert agent._persist_calls == 1


def test_active_fallback_and_request_context_reject_before_prologue_effects():
    agent = _FakeAgent()
    agent._fallback_activated = True
    agent._restore_primary_runtime = MagicMock()
    with pytest.raises(NativeAdmissionError, match="native_turn_active_fallback"):
        _build(agent, native_context=NativeTurnContext("prepare"))
    agent._restore_primary_runtime.assert_not_called()
    assert agent._persist_calls == 0

    agent._fallback_activated = False
    with patch("hermes_cli.plugins.has_hook", return_value=True):
        with pytest.raises(NativeAdmissionError, match="pre_llm_context_unsupported"):
            _build(agent, native_context=NativeTurnContext("prepare"))
    assert agent._persist_calls == 0

    agent._memory_manager = object()
    with pytest.raises(NativeAdmissionError, match="memory_context_unsupported"):
        _build(agent, native_context=NativeTurnContext("prepare"))
    assert agent._persist_calls == 0


def test_preflight_compression_rejects_before_metrics_status_or_compression():
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent.context_compressor = SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1,
        last_prompt_tokens=0,
        should_defer_preflight_to_real_usage=lambda _tokens: False,
        get_active_compression_failure_cooldown=lambda: None,
        should_compress=lambda _tokens: True,
    )
    agent._compress_context = MagicMock()
    agent._emit_status = MagicMock()
    with pytest.raises(NativeAdmissionError, match="preflight_compression_required"):
        _build(agent, native_context=NativeTurnContext("prepare"))
    assert agent.context_compressor.last_prompt_tokens == 0
    agent._compress_context.assert_not_called()
    agent._emit_status.assert_not_called()
    assert agent._persist_calls == 0


def test_native_preflight_respects_existing_deferral():
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent.context_compressor = SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1,
        last_prompt_tokens=0,
        last_real_prompt_tokens=1,
        should_defer_preflight_to_real_usage=lambda _tokens: True,
        get_active_compression_failure_cooldown=lambda: None,
        should_compress=lambda _tokens: True,
    )
    agent._compress_context = MagicMock()
    _build(agent, native_context=NativeTurnContext("prepare"))
    agent._compress_context.assert_not_called()
    assert agent.context_compressor.last_prompt_tokens == 0


def test_disposable_real_agent_preparation_uses_same_history(tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="cli",
            quiet_mode=True,
            session_db=SessionDB(db_path=tmp_path / "session.db"),
            session_id="native-fixture",
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "FIXED SYSTEM"
    agent._restore_primary_runtime = MagicMock()
    agent._ensure_db_session = MagicMock()
    agent._persist_session = MagicMock()
    result = _build(
        agent,
        conversation_history=[{"role": "assistant", "content": "earlier"}],
        native_context=NativeTurnContext("prepare"),
    )
    assert result.messages[-2:] == [
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "hello"},
    ]
    assert result.active_system_prompt == "FIXED SYSTEM"
    agent._restore_primary_runtime.assert_not_called()
    agent._ensure_db_session.assert_not_called()
    agent._persist_session.assert_not_called()
