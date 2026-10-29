"""Constructor-owned prompt state without clients, credentials or runtime state.

These helpers share the real constructor's config/tool semantics. Constructing
state is pure; rendering still resolves native SOUL/context/skills/environment
and subscription dependencies through the public system-prompt builder. A
state object or rendered string alone never attests those dependencies.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any


def apply_prompt_configuration(agent: Any, config: dict) -> None:
    """Apply the exact prompt flags used by the normal AIAgent constructor."""
    section = config.get("agent", {})
    if not isinstance(section, dict):
        section = {}
    agent._tool_use_enforcement = section.get("tool_use_enforcement", "auto")
    agent._task_completion_guidance = bool(section.get("task_completion_guidance", True))
    agent._parallel_tool_call_guidance = bool(section.get("parallel_tool_call_guidance", True))
    agent._environment_probe = bool(section.get("environment_probe", True))
    hints = config.get("platform_hints", {})
    agent._platform_hint_overrides = hints if isinstance(hints, dict) else {}


def apply_prompt_tools(agent: Any, tools: list[dict]) -> None:
    """Apply the constructor's exact resolved schema snapshot and guidance."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    agent.tools = tools
    agent.valid_tool_names = {tool["function"]["name"] for tool in tools} if tools else set()
    agent._kanban_worker_guidance = KANBAN_GUIDANCE if "kanban_show" in agent.valid_tool_names else ""



def cron_prompt_options(*, workdir_present: bool) -> dict:
    """Prompt-affecting cron constructor flags, shared with the scheduler."""
    if type(workdir_present) is not bool:
        raise ValueError("native_prompt_workdir_unresolved")
    return {"platform": "cron", "skip_context_files": not workdir_present,
            "load_soul_identity": True, "skip_memory": True}


def construct_cron_prompt_state(*, config: dict, model: str, provider: str,
                                tools: list[dict], context_length: int,
                                prefill_messages: list[dict],
                                session_id: str, workdir_present: bool) -> Any:
    """Build cron's prompt-only state from already resolved native inputs.

    ``tools`` is the actual native tool-resolution result, never a fabricated
    empty list. Model/provider and context length must be resolved by their
    native owners before use. This function does not resolve or attest them.
    Cron's SOUL, memory and workdir flags match scheduler construction; this
    function deliberately cannot construct a client, compressor or SessionDB.
    """
    if (not isinstance(config, dict) or not isinstance(model, str) or not model
            or not isinstance(provider, str) or not provider
            or not isinstance(session_id, str) or not session_id
            or type(context_length) is not int or context_length <= 0
            or type(workdir_present) is not bool or not isinstance(tools, list)
            or not isinstance(prefill_messages, list)):
        raise ValueError("native_prompt_state_unresolved")
    state = SimpleNamespace(model=model, provider=provider, session_id=session_id,
        **cron_prompt_options(workdir_present=workdir_present),
        prefill_messages=deepcopy(prefill_messages), pass_session_id=False,
        _memory_store=None, _memory_manager=None, _memory_enabled=False,
        _user_profile_enabled=False,
        context_compressor=SimpleNamespace(context_length=context_length))
    apply_prompt_configuration(state, deepcopy(config))
    apply_prompt_tools(state, deepcopy(tools))
    return state
