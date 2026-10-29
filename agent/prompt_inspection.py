"""Native prompt-only inspection using the public builder and shared init state.

This is a construction primitive, not an authority or admission witness. Native
owners must supply resolved model/provider, context length and actual schemas;
callers must separately qualify every source and dynamic dependency. Denied
inspection dependencies must not become empty/default prompt fragments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.prompt_state import construct_cron_prompt_state


@dataclass(frozen=True)
class NativePromptHelpers:
    load_soul_md: Callable[..., str]
    build_environment_hints: Callable[..., str]
    build_context_files_prompt: Callable[..., str]
    build_nous_subscription_prompt: Callable[..., str]
    build_skills_system_prompt: Callable[..., str]
    get_toolset_for_tool: Callable[..., Any]


def native_prompt_helpers() -> NativePromptHelpers:
    """Use native implementations without importing the runtime runner."""
    from agent import prompt_builder
    from tools.registry import registry

    return NativePromptHelpers(
        prompt_builder.load_soul_md, prompt_builder.build_environment_hints,
        prompt_builder.build_context_files_prompt, prompt_builder.build_nous_subscription_prompt,
        prompt_builder.build_skills_system_prompt, registry.get_toolset_for_tool)


def inspect_cron_prompt(**native_inputs) -> dict:
    """Render resolved cron state without constructing AIAgent/runtime owners.

    This does not disable tool guidance, SOUL, environment probes, subscription
    guidance or context files to make inspection pass. A mandatory native
    audit scope refuses observed synchronous network, credential and write
    dependencies with a denial that optional native fallbacks cannot hide.
    Isolated execution/source qualification remains the caller's responsibility.
    """
    from agent.prompt_inspection_scope import (
        PromptInspectionUnavailable, PromptSourceUnavailable, prompt_source_inspection,
    )

    try:
        with prompt_source_inspection():
            from agent.system_prompt import build_system_prompt_parts

            state = construct_cron_prompt_state(**native_inputs)
            parts = build_system_prompt_parts(state, helpers=native_prompt_helpers())
    except PromptSourceUnavailable:
        raise PromptInspectionUnavailable("native_instruction_inspection_unavailable") from None
    return {"parts": parts, "instructions": "\n\n".join(
        parts[key] for key in ("stable", "context", "volatile") if parts[key]),
        "tools": state.tools, "model": state.model, "provider": state.provider,
        "prefill_messages": state.prefill_messages,
        "scope": "native_resolved_prompt_construction_not_source_or_request_attestation"}
