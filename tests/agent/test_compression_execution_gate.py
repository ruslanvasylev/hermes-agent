"""Compression middleware veto precedes auxiliary resolution for every engine."""
from types import SimpleNamespace
from unittest import mock

import pytest

from agent.conversation_compression import compress_context


class ReachedFeasibility(Exception):
    pass


def test_refusing_execution_owner_preserves_input_without_auxiliary_resolution():
    agent = SimpleNamespace(session_id="ordinary-session", _cached_system_prompt="native cached instructions")
    messages = [{"role": "user", "content": "full original input"}]
    seen = []

    def refuse(*, request, next_call, **context):
        seen.append((request, context))
        return None

    with mock.patch("hermes_cli.middleware._get_middleware_callbacks", return_value=[refuse]), mock.patch(
        "agent.conversation_compression.check_compression_model_feasibility") as feasibility:
        result = compress_context(agent, messages, "original override")
    assert result[0] is messages and result[1] == agent._cached_system_prompt
    feasibility.assert_not_called()
    assert seen[0][0] == {"messages": messages, "system_message": "original override"}
    assert seen[0][1]["execution_kind"] == "context_compression"


@pytest.mark.parametrize("with_observer", [False, True])
def test_ordinary_compression_reaches_original_feasibility_once(with_observer):
    agent = SimpleNamespace(session_id="ordinary-session")
    seen = []

    def passthrough(*, request, next_call, **context):
        seen.append(context["execution_kind"])
        return next_call(request)

    with mock.patch("hermes_cli.middleware._get_middleware_callbacks", return_value=[passthrough] if with_observer else []), mock.patch(
        "agent.conversation_compression.check_compression_model_feasibility", side_effect=ReachedFeasibility) as feasibility:
        with pytest.raises(ReachedFeasibility):
            compress_context(agent, [{"role": "user", "content": "ordinary input"}], "native instructions")
    feasibility.assert_called_once_with(agent)
    assert seen == (["context_compression"] if with_observer else [])
