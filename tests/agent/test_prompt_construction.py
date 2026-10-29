"""Opaque native constructor/turn epochs, source replay and final request controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", ["pre_request", "forged", "changed_soul", "changed_auth", "changed_config",
    "changed_tools", "changed_environment", "changed_cached_prompt", "prefill", "ephemeral", "custom_context",
    "custom_system", "history", "foreign_callback", "registration_veto", "native_request",
    "mutated_initial_instructions", "extra_assistant_input", "fork_snapshot", "changed_profile_callback", "changed_profile_data",
    "changed_profile_registry", "changed_message_converter", "changed_tools_converter", "changed_request_config", "changed_converter_data", "changed_transport_callback", "spoofed_stdlib_converter", "builtin_tools_converter", "foreign_profile_receiver", "foreign_agent_receiver"])
def test_real_native_issued_epoch_and_closed_construction(case):
    run = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "tests/fixtures/prompt_construction/native_driver.py"), case],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["case"] == case
    assert result["provider_dispatches"] == 0
    assert result["network_attempts"] == []
    assert result["inspection_events"] == []
