"""The hermetic wrapper retains the caller's explicit temporary-directory seam."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_wrapper_preserves_explicit_tmpdir_without_credentials(tmp_path):
    root = tmp_path / "source"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "run_tests.sh"
    wrapper.write_bytes((Path(__file__).resolve().parents[1] / "scripts/run_tests.sh").read_bytes())
    wrapper.chmod(0o755)
    runner = scripts / "run_tests_parallel.py"
    runner.write_text(
        "import json, os; print(json.dumps({key: os.environ.get(key) for key in "
        "['TMPDIR', 'TZ', 'LANG', 'WRAPPER_PROBE_API_KEY']}))\n", encoding="utf-8",
    )
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("", encoding="utf-8")
    (venv / "python").symlink_to(sys.executable)
    temp_root = tmp_path / "explicit temporary directory"
    temp_root.mkdir()
    environment = dict(os.environ, TMPDIR=str(temp_root), WRAPPER_PROBE_API_KEY="must-not-pass")
    result = subprocess.run([str(wrapper)], cwd=root, env=environment,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed == {"TMPDIR": str(temp_root), "TZ": "UTC", "LANG": "C.UTF-8",
                        "WRAPPER_PROBE_API_KEY": None}
