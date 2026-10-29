"""Native-issued, process-local evidence of fresh cron prompt construction.

Resolved native helper outputs are captured during the real native build, not
learned from a model request. Pure reconstruction replays those inputs through
the same native formula. Opaque handles are never serialization/approval dicts.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import threading
from types import CodeType, SimpleNamespace
import weakref

from agent.prompt_inspection_scope import PromptInspectionUnavailable, PromptSourceUnavailable, prompt_source_inspection

_ROOT = Path(__file__).resolve().parents[1]
_ISSUER = object()
_LOCK = threading.RLock()
_CONSTRUCTORS = weakref.WeakKeyDictionary()
_SNAPSHOTS = weakref.WeakKeyDictionary()
_TURNS = {}
_CAPTURE = ContextVar("native_prompt_construction_capture", default=None)
_REPLAY = ContextVar("native_prompt_construction_replay", default=None)
_AUDIT_REENTRY = ContextVar("native_prompt_construction_audit_reentry", default=False)
_AUDIT_REGISTERED = False
_CODE_CACHE = {}


def _after_fork():
    # A lock owned by another parent thread may remain locked forever in the
    # child. Drop process-local issuance before any child handle lookup/lock.
    global _LOCK, _CONSTRUCTORS, _SNAPSHOTS, _TURNS, _CODE_CACHE
    _LOCK = threading.RLock()
    _CONSTRUCTORS = weakref.WeakKeyDictionary()
    _SNAPSHOTS = weakref.WeakKeyDictionary()
    _TURNS = {}
    _CODE_CACHE = {}
    _CAPTURE.set(None)
    _REPLAY.set(None)
    _AUDIT_REENTRY.set(False)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
_FIELDS = ("model", "provider", "api_mode", "base_url", "session_id", "platform", "skip_context_files",
           "load_soul_identity", "pass_session_id", "tools", "valid_tool_names", "_kanban_worker_guidance",
           "_tool_use_enforcement", "_task_completion_guidance", "_parallel_tool_call_guidance",
           "_environment_probe", "_platform_hint_overrides", "prefill_messages", "ephemeral_system_prompt",
           "reasoning_config", "max_tokens", "service_tier", "_force_ascii_payload", "_codex_reasoning_replay_enabled")
_OWNERS = {
    "soul": {"agent/prompt_builder.py:load_soul_md"},
    "environment": {"agent/prompt_builder.py:build_environment_hints"},
    "context": {"agent/prompt_builder.py:build_context_files_prompt"},
    "subscription": {"agent/prompt_builder.py:build_nous_subscription_prompt"},
    "skills": {"agent/prompt_builder.py:build_skills_system_prompt"},
    "toolset": {"model_tools.py:get_toolset_for_tool", "tools/registry.py:ToolRegistry.get_toolset_for_tool"},
    "cwd": {"agent/runtime_cwd.py:resolve_context_cwd"},
    "compact_skills": {"agent/coding_context.py:coding_compact_skill_categories"},
    "coding": {"agent/coding_context.py:coding_system_blocks"},
    "environment_probe": {"tools/env_probe.py:get_environment_probe_line"},
    "profile": {"agent/file_safety.py:_resolve_active_profile_name"},
    "time": {"hermes_time.py:now"},
}


def _fail():
    raise PromptInspectionUnavailable("native_instruction_inspection_unavailable")


def _normal(value):
    if value is None or type(value) in (str, int, bool, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normal(item) for item in value), key=repr)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _normal(item) for key, item in sorted(value.items())}
    if isinstance(value, Path):
        return {"native_path": str(value)}
    if isinstance(value, datetime):
        return {"native_datetime": value.isoformat()}
    _fail()


def _hash(value):
    return hashlib.sha256(json.dumps(_normal(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()



def _text_hash(value):
    if not isinstance(value, str):
        _fail()
    return hashlib.sha256(value.encode()).hexdigest()


def _fingerprint(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        return None


def _code_objects(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _code_objects(value)



def _code_payload(code):
    def constant(value):
        if isinstance(value, CodeType):
            return _code_payload(value)
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, (tuple, list)):
            return [constant(item) for item in value]
        if isinstance(value, frozenset):
            return sorted((constant(item) for item in value), key=repr)
        if value is Ellipsis:
            return {"ellipsis": True}
        if isinstance(value, complex):
            return {"complex": repr(value)}
        return value
    return {"name": code.co_name, "qualname": code.co_qualname,
            "args": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
            "flags": code.co_flags, "bytecode": code.co_code.hex(),
            "constants": [constant(item) for item in code.co_consts],
            "names": code.co_names, "varnames": code.co_varnames,
            "freevars": code.co_freevars, "cellvars": code.co_cellvars,
            "firstline": code.co_firstlineno, "linetable": code.co_linetable.hex(),
            "exceptions": code.co_exceptiontable.hex()}


def _qualified(callback, expected=None):
    function = getattr(callback, "__func__", callback)
    code = getattr(function, "__code__", None)
    if code is None:
        _fail()
    path = Path(code.co_filename).resolve()
    if not path.is_relative_to(_ROOT):
        _fail()
    relative = str(path.relative_to(_ROOT))
    identity = relative + ":" + code.co_qualname
    if expected is not None and identity not in expected:
        _fail()
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    key = (str(path), source_hash, sys.flags.optimize)
    candidates = _CODE_CACHE.get(key)
    if candidates is None:
        compiled = compile(raw, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
        candidates = {item.co_qualname: item for item in _code_objects(compiled)}
        _CODE_CACHE[key] = candidates
    matching = candidates.get(code.co_qualname)
    if matching is None or matching != code:
        _fail()
    return {"owner": identity, "source_sha256": source_hash,
            "code_sha256": _hash(_code_payload(code))}


def _caller(expected, depth=2):
    frame = sys._getframe(depth)
    path = Path(frame.f_code.co_filename).resolve()
    if not path.is_relative_to(_ROOT) or str(path.relative_to(_ROOT)) + ":" + frame.f_code.co_qualname != expected:
        _fail()
    _qualified(SimpleNamespace(__code__=frame.f_code), {expected})



def _environment_identity():
    values = {}
    for name in list(os.environ):
        upper = name.upper()
        credential = any(part in upper for part in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "AUTH")) or upper.endswith(("_KEY", "_API_KEY"))
        # Inspection never reads credential material from the environment.
        values[name] = {"credential_key_present": True} if credential else os.environ[name]
    return _hash({"cwd": os.getcwd(), "environment": values})


def _state(agent):
    value = {name: deepcopy(getattr(agent, name, None)) for name in _FIELDS}
    value["context_length"] = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    value["memory_present"] = bool(getattr(agent, "_memory_store", None) or getattr(agent, "_memory_manager", None))
    return value


def _eligible(agent, state):
    from run_agent import AIAgent
    return (type(agent) is AIAgent and state["platform"] == "cron" and state["skip_context_files"] is True
            and state["load_soul_identity"] is True and not state["prefill_messages"]
            and not state["ephemeral_system_prompt"] and not state["memory_present"]
            and not getattr(agent, "request_overrides", None) and not state["_force_ascii_payload"]
            and type(state["context_length"]) is int and state["context_length"] > 0)


def _sources(agent):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import get_config_path
    home = get_hermes_home()
    paths = [get_config_path(), home / "SOUL.md", home / ".env", home / "auth.json"]
    return {str(path.resolve()): _fingerprint(path) for path in paths}


def begin_constructor(agent):
    """Internal real-constructor call site; no caller declaration grants issue."""
    try:
        _caller("agent/agent_init.py:init_agent")
        _caller("run_agent.py:AIAgent.__init__", depth=3)
        from run_agent import AIAgent
        if type(agent) is not AIAgent:
            return
        with _LOCK:
            _CONSTRUCTORS[agent] = {"pid": os.getpid(), "epoch": secrets.token_hex(16),
                "sources": _sources(agent), "complete": False, "turn": None}
    except Exception:
        return


def finish_constructor(agent):
    try:
        _caller("agent/agent_init.py:init_agent")
        with _LOCK:
            row = _CONSTRUCTORS.get(agent)
            state = _state(agent)
            if row is None or not _eligible(agent, state) or any(_fingerprint(path) != fp for path, fp in row["sources"].items()):
                return
            row.update(complete=True, state=state, state_sha256=_hash(state), environment=_environment_identity())
    except Exception:
        return


def note_turn(agent, turn_id, *, user_message, system_message, conversation_history):
    try:
        _caller("agent/turn_context.py:build_turn_context")
        with _LOCK:
            row = _CONSTRUCTORS.get(agent)
            if (row is None or not row["complete"] or row["turn"] is not None or system_message is not None
                    or conversation_history or not isinstance(user_message, str)
                    or getattr(agent, "_cached_system_prompt", None) is not None):
                return
            row["turn"] = {"id": turn_id, "user_message_sha256": hashlib.sha256(user_message.encode()).hexdigest(),
                           "user_message": user_message}
    except Exception:
        return


@dataclass(frozen=True)
class ConstructedPromptView:
    instructions: str
    tools: tuple
    user_message_sha256: str
    proof: dict


class PromptConstructionSnapshot:
    __slots__ = ("__weakref__",)

    def __new__(cls, issuer=None):
        if issuer is not _ISSUER:
            raise TypeError("native prompt snapshots are issued, not constructed")
        return super().__new__(cls)

    def __reduce__(self):
        raise TypeError("native prompt snapshots cannot be serialized")

    def reconstruct(self):
        return _reconstruct(self)

    def expected_request(self):
        row = _current(self)
        if "request" not in row:
            _fail()
        from agent.request_construction import replay_request
        replay_request(row["agent"](), row["request_epoch"])
        return deepcopy(row["request"])

    def request_proof(self):
        row = _current(self)
        if "request" not in row:
            _fail()
        return {"schema_version": 2, "epoch": row["epoch"], "turn_id": row["turn"]["id"],
                "request_epoch_sha256": row["request_epoch"]["identity_sha256"],
                "native_context_length": row["state"]["context_length"],
                "request_sha256": row["request_sha256"], "code_owners": deepcopy(row["request_code_owners"]),
                "source_fingerprints_sha256": _hash(row["request_sources"]),
                "scope": "native_epoch_frozen_request_replayed_before_external_hooks_not_first_preflight"}


class _Capture:
    def __init__(self, agent, constructor):
        self.agent = weakref.ref(agent)
        self.constructor = constructor
        self.rows = []
        self.sources = dict(constructor["sources"])
        self.codes = {}
        self.valid = True
        self.audit_probe = None
        self.audit_probe_seen = None
        self.processes = []

    def complete(self, prompt):
        agent = self.agent()
        if not self.valid or agent is None:
            return
        try:
            from agent import system_prompt
            builders = [
                _qualified(agent._build_system_prompt, {"run_agent.py:AIAgent._build_system_prompt"}),
                _qualified(system_prompt.build_system_prompt, {"agent/system_prompt.py:build_system_prompt"}),
                _qualified(system_prompt.build_system_prompt_parts, {"agent/system_prompt.py:build_system_prompt_parts"}),
            ]
            for owner in builders:
                self.codes[owner["owner"]] = owner
            state = _state(agent)
            if (_hash(state) != self.constructor["state_sha256"] or _environment_identity() != self.constructor["environment"]
                    or any(_fingerprint(path) != fp for path, fp in self.sources.items())):
                return
            handle = PromptConstructionSnapshot(_ISSUER)
            row = {"pid": os.getpid(), "agent": self.agent, "epoch": self.constructor["epoch"],
                   "turn": deepcopy(self.constructor["turn"]), "state": state, "rows": deepcopy(self.rows),
                   "sources": dict(self.sources), "codes": dict(self.codes), "instructions_sha256": _text_hash(prompt),
                   "environment": self.constructor["environment"], "processes": deepcopy(self.processes)}
            from agent.request_construction import freeze_request
            request_epoch = freeze_request(agent, prompt, self.constructor["turn"]["user_message"])
            row["sources"] = dict(self.sources)
            row.update(request_epoch=request_epoch, request=deepcopy(request_epoch["request"]),
                       request_sha256=_hash(request_epoch["request"]),
                       request_model_sha256=request_epoch["model_sha256"],
                       request_code_owners=request_epoch["code_owners"], request_sources=request_epoch["sources"])
            with _LOCK:
                _SNAPSHOTS[handle] = row
                key = (state["session_id"], row["turn"]["id"])
                _TURNS[key] = (weakref.ref(agent, lambda _ref, k=key: _TURNS.pop(k, None)), handle)
        except Exception:
            return


class _NoCapture:
    def complete(self, _prompt):
        return


def _source_audit(event, args):
    capture = _CAPTURE.get()
    if capture is None or _AUDIT_REENTRY.get():
        return
    token = _AUDIT_REENTRY.set(True)
    try:
        if event == "native.prompt_construction_source_probe":
            if args == (capture.audit_probe,) and capture.audit_probe is not None:
                capture.audit_probe_seen = capture.audit_probe
            return
        if event.startswith("socket.") or event in {"os.system", "os.posix_spawn", "os.fork", "os.forkpty", "os.exec"}:
            capture.valid = False
        if event == "subprocess.Popen":
            executable, arguments, cwd, _environment = args
            if not isinstance(executable, (str, bytes, os.PathLike)) or not Path(os.fsdecode(executable)).is_absolute():
                capture.valid = False
            else:
                resolved = str(Path(os.fsdecode(executable)).resolve())
                capture.sources.setdefault(resolved, _fingerprint(resolved))
                capture.processes.append({"executable_identity": _hash(resolved), "arguments_sha256": _hash(arguments),
                                          "cwd_sha256": _hash(cwd)})
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes, os.PathLike)):
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                if not writing:
                    resolved = str(Path(os.fsdecode(path)).resolve())
                    capture.sources.setdefault(resolved, _fingerprint(resolved))
    except Exception:
        capture.valid = False
    finally:
        _AUDIT_REENTRY.reset(token)


@contextmanager
def capture_prompt(agent, system_message, conversation_history):
    global _AUDIT_REGISTERED
    capture = _NoCapture()
    try:
        _caller("agent/conversation_loop.py:_restore_or_build_system_prompt", depth=3)
        with _LOCK:
            constructor = _CONSTRUCTORS.get(agent)
            if (constructor and constructor["complete"] and constructor["turn"] is not None
                    and constructor["pid"] == os.getpid() and system_message is None and not conversation_history
                    and getattr(agent, "_current_turn_id", None) == constructor["turn"]["id"]
                    and not getattr(agent, "_current_api_request_id", "")):
                capture = _Capture(agent, constructor)
    except Exception:
        pass
    if isinstance(capture, _Capture):
        from agent.request_construction import prepare_request_dependencies
        prepare_request_dependencies()
    token = _CAPTURE.set(capture if isinstance(capture, _Capture) else None)
    try:
        if isinstance(capture, _Capture):
            try:
                if not _AUDIT_REGISTERED:
                    sys.addaudithook(_source_audit)
                capture.audit_probe = object()
                sys.audit("native.prompt_construction_source_probe", capture.audit_probe)
                if capture.audit_probe_seen is not capture.audit_probe:
                    capture.valid = False
                    _AUDIT_REGISTERED = False
                else:
                    _AUDIT_REGISTERED = True
            except Exception:
                capture.valid = False
        yield capture
    finally:
        _CAPTURE.reset(token)


def dependency(name, callback, *args, **kwargs):
    replay = _REPLAY.get()
    if replay is not None:
        index, rows = replay["index"], replay["rows"]
        if index >= len(rows) or rows[index]["name"] != name or rows[index]["arguments"] != _hash([args, kwargs]):
            _fail()
        actual = _qualified(callback, _OWNERS.get(name, set()))
        if actual != rows[index]["owner"]:
            _fail()
        replay["index"] += 1
        return deepcopy(rows[index]["result"])
    capture = _CAPTURE.get()
    if capture is None:
        return callback(*args, **kwargs)
    try:
        owner = _qualified(callback, _OWNERS.get(name, set()))
    except Exception:
        capture.valid = False
        return callback(*args, **kwargs)
    try:
        result = callback(*args, **kwargs)
        capture.rows.append({"name": name, "arguments": _hash([args, kwargs]),
                             "result": deepcopy(result), "owner": owner})
        capture.codes[owner["owner"]] = owner
        return result
    except Exception:
        capture.valid = False
        raise


def read_prompt_construction_snapshot(*, session_id: str, api_request_id: str | None = None):
    """Read a unique native-issued epoch, before dispatch or at first request.

    Before-request observers can address a session whose native turn already
    exists. Final middleware additionally supplies the exact native request id.
    No caller-provided epoch payload, prompt hash or approval is accepted.
    """
    with _LOCK:
        if api_request_id is None:
            matches = [value[1] for key, value in _TURNS.items() if key[0] == session_id and value[0]() is not None]
            if len(matches) != 1:
                _fail()
            handle = matches[0]
            row = _current(handle)
            if getattr(row["agent"](), "_current_api_request_id", ""):
                _fail()
        else:
            if not isinstance(api_request_id, str) or not api_request_id.endswith(":api:1"):
                _fail()
            turn = api_request_id[:-6]
            found = _TURNS.get((session_id, turn))
            if found is None:
                _fail()
            handle = found[1]
            _current(handle, api_request_id=api_request_id)
    return handle


def _current(handle, api_request_id=None):
    if type(handle) is not PromptConstructionSnapshot:
        _fail()
    with _LOCK:
        row = _SNAPSHOTS.get(handle)
    if row is None or row["pid"] != os.getpid():
        _fail()
    agent = row["agent"]()
    if (agent is None or not _eligible(agent, _state(agent)) or _hash(_state(agent)) != _hash(row["state"])
            or _environment_identity() != row["environment"]):
        _fail()
    expected = row["turn"]["id"] + ":api:1"
    if (getattr(agent, "_current_turn_id", None) != row["turn"]["id"]
            or getattr(agent, "_current_api_request_id", "") not in ("", expected)
            or api_request_id is not None and (api_request_id != expected or getattr(agent, "_current_api_request_id", None) != expected)
            or _text_hash(getattr(agent, "_cached_system_prompt", None)) != row["instructions_sha256"]
            or any(_fingerprint(path) != fp for path, fp in row["sources"].items())
            or any(_fingerprint(path) != fp for path, fp in row.get("request_sources", {}).items())):
        _fail()
    return row


def _reconstruct(handle):
    row = _current(handle)
    state = row["state"]
    agent = SimpleNamespace(**{name: deepcopy(value) for name, value in state.items() if name not in {"context_length", "memory_present"}})
    agent.context_compressor = SimpleNamespace(context_length=state["context_length"])
    agent._memory_store = agent._memory_manager = None
    replay = {"index": 0, "rows": row["rows"]}
    token = _REPLAY.set(replay)
    try:
        with prompt_source_inspection():
            from agent.system_prompt import build_system_prompt_parts
            parts = build_system_prompt_parts(agent)
        instructions = "\n\n".join(parts[key] for key in ("stable", "context", "volatile") if parts[key])
        if replay["index"] != len(row["rows"]) or _text_hash(instructions) != row["instructions_sha256"]:
            _fail()
        _current(handle)
        proof = {"schema_version": 1, "epoch": row["epoch"], "session_id": state["session_id"],
                 "turn_id": row["turn"]["id"], "instructions_sha256": row["instructions_sha256"],
                 "state_sha256": _hash(state), "dependencies_sha256": _hash(row["rows"]),
                 "source_fingerprints_sha256": _hash(row["sources"]), "code_owners": list(row["codes"].values()),
                 "environment_sha256": row["environment"], "resolved_processes_sha256": _hash(row["processes"]),
                 "environment_scope": "nonsecret_values_and_credential_key_presence_not_credential_material",
                 "scope": "native_issued_pre_request_construction_not_provider_wire_attestation"}
        return ConstructedPromptView(instructions, tuple(deepcopy(state["tools"])), row["turn"]["user_message_sha256"], proof)
    except PromptSourceUnavailable:
        _fail()
    finally:
        _REPLAY.reset(token)


def note_api_construction(agent, api_messages, api_kwargs):
    """Internal qualified native preflight boundary, before request middleware.

    This is not an incoming-request approval API. It requires the native-issued
    constructor/turn, independent prompt reconstruction and original-user hash,
    real compiled native builders and the unchanged native state/sources.
    """
    try:
        _caller("agent/conversation_loop.py:run_conversation")
        with _LOCK:
            found = _TURNS.get((agent.session_id, getattr(agent, "_current_turn_id", None)))
            if found is None or found[0]() is not agent:
                return
            handle = found[1]
        row = _current(handle, api_request_id=getattr(agent, "_current_api_request_id", ""))
        view = _reconstruct(handle)
        if (not isinstance(api_messages, list) or len(api_messages) != 2
                or any(not isinstance(item, dict) or set(item) != {"role", "content"} for item in api_messages)
                or api_messages[0] != {"role": "system", "content": view.instructions}
                or api_messages[1].get("role") != "user" or not isinstance(api_messages[1].get("content"), str)
                or hashlib.sha256(api_messages[1]["content"].encode()).hexdigest() != view.user_message_sha256
                or not isinstance(api_kwargs, dict) or api_kwargs.get("instructions") != view.instructions
                or agent.api_mode != "codex_responses" or agent.provider != "openai-codex"):
            return
        from agent.request_construction import replay_request
        replay_request(agent, row["request_epoch"])
        model_hash = _hash({key: value for key, value in api_kwargs.items()
                            if key not in {"timeout", "stream", "extra_headers"}})
        if model_hash != row["request_model_sha256"]:
            row["request_changed"] = True
        if row.get("request_changed"):
            row.pop("request", None)
    except Exception:
        # An invalid preflight cannot retain an earlier qualified request.
        if "row" in locals():
            row.pop("request", None)
            row["request_changed"] = True
        return
