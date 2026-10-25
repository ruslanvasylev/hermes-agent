"""Optional per-conversation application dispatch and cancellation budgets.

This counts native execution dispatches, not SDK/transport wire attempts.  The
elapsed limit requests the agent's normal cooperative cancellation; it cannot
kill an arbitrary Python tool thread or undo an already submitted remote call.
"""
from __future__ import annotations

import math
import threading
import time


class ApiRunBudgetExceeded(InterruptedError):
    """A local run control outcome, never a provider failure to retry."""


def validate_run_limits(api_max_attempts=None, max_run_seconds=None):
    if api_max_attempts is not None:
        if (
            isinstance(api_max_attempts, bool)
            or not isinstance(api_max_attempts, int)
            or api_max_attempts < 1
        ):
            raise ValueError("api_max_attempts must be a positive integer")
    if max_run_seconds is not None:
        if (
            isinstance(max_run_seconds, bool)
            or not isinstance(max_run_seconds, (int, float))
            or not math.isfinite(max_run_seconds)
            or max_run_seconds <= 0
        ):
            raise ValueError("max_run_seconds must be positive and finite")
    return api_max_attempts, max_run_seconds


class ApiRunBudget:
    def __init__(self, api_max_attempts=None, max_run_seconds=None):
        self.max_attempts, self.max_seconds = validate_run_limits(api_max_attempts, max_run_seconds)
        self.started = time.monotonic()
        self.attempts = 0
        self.reason = None
        self._lock = threading.Lock()

    def remaining_seconds(self):
        if self.max_seconds is None:
            return None
        return max(0.0, self.max_seconds - (time.monotonic() - self.started))

    def expire(self):
        with self._lock:
            self.reason = self.reason or "max_run_seconds"

    def claim_attempt(self):
        with self._lock:
            if self.remaining_seconds() == 0:
                self.reason = self.reason or "max_run_seconds"
            if self.max_attempts is not None and self.attempts >= self.max_attempts:
                self.reason = self.reason or "api_max_attempts"
            if self.reason:
                raise ApiRunBudgetExceeded(self.reason)
            self.attempts += 1

    def exhaust_attempts(self):
        with self._lock:
            self.reason = self.reason or "api_max_attempts"

    def cap_timeout(self, configured):
        remaining = self.remaining_seconds()
        if remaining is None:
            return configured
        if remaining <= 0:
            self.expire()
            raise ApiRunBudgetExceeded(self.reason)
        if isinstance(configured, (int, float)):
            return min(configured, remaining)
        return remaining

    def metadata(self):
        return {
            "application_attempts": self.attempts,
            "api_max_attempts": self.max_attempts,
            "max_run_seconds": self.max_seconds,
            "run_elapsed_seconds": max(0.0, time.monotonic() - self.started),
            "budget_exhausted_reason": self.reason,
        }


def active_run_budget(agent):
    budget = getattr(agent, "_api_run_budget", None)
    return budget if isinstance(budget, ApiRunBudget) else None


def run_with_budget(agent, call, *args, **kwargs):
    limits = getattr(agent, "_api_run_budget_limits", (None, None))
    if limits == (None, None):
        return call(*args, **kwargs)
    if limits[0] is not None and getattr(agent, "api_mode", None) == "codex_app_server":
        raise ValueError(
            "api_max_attempts is not supported by codex_app_server; "
            "use a transport with native application dispatch accounting"
        )
    budget = ApiRunBudget(*limits)
    agent._api_run_budget = budget
    timer = None
    if budget.max_seconds is not None:
        def expire():
            budget.expire()
            agent.interrupt("Run elapsed budget exhausted")
        timer = threading.Timer(budget.max_seconds, expire)
        timer.daemon = True
        timer.start()
    try:
        result = call(*args, **kwargs)
    except ApiRunBudgetExceeded:
        result = {"final_response": "", "completed": False}
    finally:
        if timer:
            timer.cancel()
    if budget.remaining_seconds() == 0:
        budget.expire()
    if budget.reason:
        result = dict(result)
        result.update(
            completed=False,
            failed=True,
            error_type="ApiRunBudgetExceeded",
            error_code=budget.reason,
            error=f"Run budget exhausted: {budget.reason}",
            turn_exit_reason="api_run_budget_exhausted",
            api_run_budget=budget.metadata(),
        )
        callback = getattr(agent, "event_callback", None)
        if callback:
            try:
                callback("api_run_budget_exhausted", {
                    "session_id": agent.session_id or "", **budget.metadata(),
                })
            except Exception:
                pass
    return result
