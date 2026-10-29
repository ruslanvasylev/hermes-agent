"""A reaped owned child remains safe when psutil loses its parent walk."""
from __future__ import annotations

import os
import signal
import subprocess
import sys

import psutil
import pytest


def test_reaped_owned_child_refuses_without_signalling_reused_pid(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    original = psutil.Process

    class VanishedParentWalk:
        def parents(self):
            raise psutil.NoSuchProcess(child.pid)

    monkeypatch.setattr(psutil, "Process", lambda pid: VanishedParentWalk()
                        if pid == child.pid else original(pid))
    # The signal is never sent: a recorded Popen proves this owned child was
    # reaped, so its old numeric PID cannot grant authority to signal anything.
    with pytest.raises(ProcessLookupError, match="owned test child"):
        os.kill(child.pid, signal.SIGTERM)


def test_foreign_pid_parent_walk_failure_still_refuses(monkeypatch):
    foreign_pid = 2 ** 30
    original = psutil.Process

    class VanishedParentWalk:
        def parents(self):
            raise psutil.NoSuchProcess(foreign_pid)

    monkeypatch.setattr(psutil, "Process", lambda pid: VanishedParentWalk()
                        if pid == foreign_pid else original(pid))
    with pytest.raises(RuntimeError, match="outside the test process subtree"):
        os.kill(foreign_pid, signal.SIGTERM)
