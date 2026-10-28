"""Read-only execution scope for trusted native synchronous prompt inspection.

The audit hook acts only in this ContextVar scope. Ordinary native execution is
unchanged. This is a fail-closed diagnostic boundary over qualified native
Python owners, not a sandbox for hostile extensions or arbitrary native code.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import sys
import threading


class PromptSourceUnavailable(BaseException):
    """Mandatory refusal; optional native Exception fallbacks cannot hide it."""


class PromptInspectionUnavailable(RuntimeError):
    """The isolated inspection did not produce a complete native result."""


_ACTIVE = ContextVar("native_prompt_source_inspection", default=False)
_REGISTERED = False
_REGISTRATION_LOCK = threading.Lock()
_MUTATIONS = {"os.rename", "os.remove", "os.rmdir", "os.link", "os.symlink", "os.chmod",
              "os.truncate", "os.utime", "os.chown", "os.setxattr", "os.removexattr", "sqlite3.connect"}
_EXTERNAL = {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "ctypes.dlopen"}


def prompt_source_inspection_active() -> bool:
    return _ACTIVE.get()


def _audit(event, args):
    if not _ACTIVE.get():
        return
    reason = None
    if event.startswith("socket.") or event in _EXTERNAL or event in _MUTATIONS:
        reason = "native_prompt_dependency_not_read_only"
    if event == "os.mkdir" and (len(args) > 2 and args[2] != -1 or not Path(args[0]).is_dir()):
        reason = "native_prompt_dependency_would_create_directory"
    if event == "open":
        path, mode, flags = args
        name = str(path).replace("\\", "/").lower()
        if isinstance(path, (str, bytes, os.PathLike)):
            name += " " + str(Path(os.fsdecode(path)).resolve()).replace("\\", "/").lower()
        if any(token in name for token in ["/.env", "/auth.json", "/credentials", "/secrets/",
                                           "/.codex/", "/.netrc", "/.ssh/", "/.gnupg/"]):
            reason = "native_prompt_dependency_would_read_credentials"
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            reason = "native_prompt_dependency_would_write"
    if reason is not None:
        raise PromptSourceUnavailable(reason)


def _register_audit_hook():
    global _REGISTERED
    with _REGISTRATION_LOCK:
        if not _REGISTERED:
            sys.addaudithook(_audit)
            _REGISTERED = True


@contextmanager
def prompt_source_inspection():
    _register_audit_hook()
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
