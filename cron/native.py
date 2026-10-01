"""Generic, opt-in cron preparation and run-local archive custody.

The selected plugin owns policy, commit and any signature.  Hermes owns the
actual prompt assembly, SDK request measurement, guarded send and archive path.
Nothing here grants external admission or starts a separate scheduler.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from agent.native_request import (
    AttemptObservation, NativeAdmissionError, NativeExecutionResult,
    NativePreparationResult, NativeRequestTurn, PreparedRequest,
)


@dataclass(frozen=True)
class CronNativeVariant:
    form_id: str
    script_stdout: str
    owner_binding: str


@dataclass(frozen=True)
class CronNativeProfile:
    provider: str
    model: str
    api_mode: str
    base_url: str
    context_length: int
    threshold_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class CronNativePrepared:
    form_id: str
    preparation_token: str
    owner_binding: str
    request: PreparedRequest
    profile: CronNativeProfile


@dataclass(frozen=True)
class CronNativeArchiveInput:
    job_id: str
    owner_binding: str
    form_id: str
    preparation_token: str
    archive_path: Path
    archive_sha256: str
    attempt: AttemptObservation
    outer_callback_count: int
    outer_retry_count: int
    fallback_activated: bool
    compression_count: int


@dataclass(frozen=True)
class CronNativeSelection:
    form_id: str
    preparation_token: str
    owner_binding: str
    archive_callback: Callable[[CronNativeArchiveInput], Any]
    delegate_factory: Callable[[], Any] | None = None


@dataclass(frozen=True)
class CronNativePlan:
    variants: tuple[CronNativeVariant, ...]
    commit: Callable[[tuple[CronNativePrepared, ...]], CronNativeSelection]


class CronNativePreparationProvider(Protocol):
    def prepare(self, *, job: dict, script_output: str) -> CronNativePlan: ...


@dataclass(frozen=True)
class SelectedCronNativeTurn:
    prompt: str
    prepared: CronNativePrepared
    selection: CronNativeSelection


@dataclass(frozen=True)
class CronNativeCustody:
    job_id: str
    selected: SelectedCronNativeTurn
    execution: NativeExecutionResult


_custody: ContextVar[CronNativeCustody | None] = ContextVar("cron_native_custody", default=None)


def clear_run_custody() -> None:
    _custody.set(None)


def publish_run_custody(custody: CronNativeCustody) -> None:
    if _custody.get() is not None:
        raise NativeAdmissionError("cron_native_custody_reissued")
    _custody.set(custody)


def take_run_custody() -> CronNativeCustody | None:
    value = _custody.get()
    _custody.set(None)
    return value


def profile_of(agent: Any) -> CronNativeProfile:
    compressor = getattr(agent, "context_compressor", None)
    return CronNativeProfile(
        provider=str(agent.provider or ""), model=str(agent.model or ""),
        api_mode=str(agent.api_mode or ""), base_url=str(agent.base_url or ""),
        context_length=int(getattr(compressor, "context_length", 0) or 0),
        threshold_tokens=int(getattr(compressor, "threshold_tokens", 0) or 0),
        max_tokens=int(getattr(agent, "max_tokens", 0) or 0),
    )


def prepare_and_commit(
    *, job: dict, plan: CronNativePlan, build_prompt: Callable[[dict, tuple], str],
    make_agent: Callable[[], Any], persist_user_message: str,
) -> SelectedCronNativeTurn:
    """Capture every final SDK request, then call the trusted commit once."""
    if not isinstance(plan, CronNativePlan) or not callable(plan.commit):
        raise NativeAdmissionError("cron_native_plan_invalid")
    variants = plan.variants
    if not isinstance(variants, tuple) or not variants:
        raise NativeAdmissionError("cron_native_forms_missing")
    form_ids: set[str] = set()
    prepared: list[CronNativePrepared] = []
    prompts: dict[str, str] = {}
    profile: CronNativeProfile | None = None
    for variant in variants:
        if (not isinstance(variant, CronNativeVariant)
                or not isinstance(variant.form_id, str) or not variant.form_id.strip()
                or variant.form_id in form_ids
                or not isinstance(variant.script_stdout, str) or not variant.script_stdout
                or not isinstance(variant.owner_binding, str) or not variant.owner_binding):
            raise NativeAdmissionError("cron_native_form_invalid")
        form_ids.add(variant.form_id)
        prompt = build_prompt(job, (True, variant.script_stdout))
        if not isinstance(prompt, str) or not prompt:
            raise NativeAdmissionError("cron_native_prompt_missing")
        agent = make_agent()
        try:
            current_profile = profile_of(agent)
            if profile is not None and current_profile != profile:
                raise NativeAdmissionError("cron_native_profile_drift")
            profile = current_profile
            result = agent.run_conversation(
                prompt, persist_user_message=persist_user_message,
                native_turn=NativeRequestTurn("prepare"),
            )
            if not isinstance(result, NativePreparationResult):
                raise NativeAdmissionError("cron_native_capture_missing")
            prepared.append(CronNativePrepared(
                form_id=variant.form_id, preparation_token=uuid4().hex,
                owner_binding=variant.owner_binding, request=result.prepared,
                profile=current_profile,
            ))
            prompts[variant.form_id] = prompt
        finally:
            agent.close()
    selection = plan.commit(tuple(prepared))
    if not isinstance(selection, CronNativeSelection) or not callable(selection.archive_callback):
        raise NativeAdmissionError("cron_native_commit_invalid")
    matches = [item for item in prepared if item.form_id == selection.form_id]
    if (len(matches) != 1 or selection.preparation_token != matches[0].preparation_token
            or selection.owner_binding != matches[0].owner_binding):
        raise NativeAdmissionError("cron_native_selection_unprepared")
    return SelectedCronNativeTurn(
        prompt=prompts[selection.form_id], prepared=matches[0], selection=selection,
    )


def custody_from_execution(
    job: dict, selected: SelectedCronNativeTurn, execution: NativeExecutionResult,
) -> CronNativeCustody:
    if not isinstance(execution, NativeExecutionResult):
        raise NativeAdmissionError("cron_native_execution_missing")
    if (execution.attempt.prepared != selected.prepared.request
            or execution.outer_callback_count != 1 or execution.outer_retry_count != 0
            or execution.fallback_activated or execution.compression_count != 0):
        raise NativeAdmissionError("cron_native_execution_custody_invalid")
    return CronNativeCustody(job_id=str(job["id"]), selected=selected, execution=execution)
