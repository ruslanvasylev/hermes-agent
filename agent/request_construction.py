"""Replay native model inputs from a pre-hook epoch, never from first preflight.

Only the existing fresh Codex cron construction is supported. The native
builder and its called native Python dependencies execute on a frozen replica;
provider callbacks, resolved profile data and referenced mutable globals are
qualified on every replay. No client, auth resolver or model call is involved.
Transport-only timeout is captured separately and is not model authority.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import sysconfig
import importlib._bootstrap
import importlib._bootstrap_external
from types import SimpleNamespace
from urllib.parse import ParseResult, SplitResult

_FIELDS = ("model", "provider", "api_mode", "base_url", "session_id", "tools", "max_tokens",
           "reasoning_config", "request_overrides", "_base_url_hostname", "_base_url_lower",
           "_codex_reasoning_replay_enabled")
_METHODS = ("_build_api_kwargs", "_get_transport", "_prepare_messages_for_non_vision_model",
            "_content_has_image_parts", "_resolved_api_call_timeout")
_TRANSPORT = {"timeout", "stream", "extra_headers"}
_STDLIB = Path(sysconfig.get_path("stdlib")).resolve()
_DYNAMIC_STDLIB_CODES = frozenset((ParseResult.__new__.__code__, SplitResult.__new__.__code__))
_FROZEN_IMPORT_CODES = frozenset(
    value.__code__ for module in (importlib._bootstrap, importlib._bootstrap_external)
    for value in vars(module).values() if hasattr(value, "__code__"))


class _UnqualifiedRequest(BaseException):
    """Cannot be swallowed by provider builders' ordinary fallback handlers."""


def _native():
    from agent import prompt_construction
    return prompt_construction


def _model(request):
    return {key: value for key, value in request.items() if key not in _TRANSPORT}


def _timeout(self):
    return self._native_transport_timeout


def _profile():
    import providers
    from providers.base import ProviderProfile
    n = _native()
    owner = n._qualified(providers.get_provider_profile, {"providers/__init__.py:get_provider_profile"})
    if providers._discovered is not True:
        n._fail()  # Discovery belongs to native initialization, never replay.
    # Resolve only the selected provider; no credential/auth function is called.
    profile = providers.get_provider_profile("openai-codex")
    if type(profile) is not ProviderProfile:
        n._fail()
    if any(getattr(getattr(profile, name), "__self__", None) is not profile
           for name in ("build_api_kwargs_extras", "get_max_tokens")):
        n._fail()
    callbacks = [n._qualified(profile.build_api_kwargs_extras, {"providers/base.py:ProviderProfile.build_api_kwargs_extras"}),
                 n._qualified(profile.get_max_tokens, {"providers/base.py:ProviderProfile.get_max_tokens"})]
    # These are the base profile's request-bearing data. Auth/header fields are
    # deliberately not read. Subclasses are unsupported, never relabeled base.
    data = {key: deepcopy(getattr(profile, key)) for key in ("name", "api_mode", "aliases", "default_max_tokens")}
    return {"lookup": owner, "callbacks": callbacks, "data": data,
            "instance_id": id(profile), "registry_canonical": providers._ALIASES.get("openai-codex", "openai-codex"),
            "registry_selected_id": id(providers._REGISTRY.get(providers._ALIASES.get("openai-codex", "openai-codex")))}


def _state(agent):
    return {key: deepcopy(getattr(agent, key)) for key in _FIELDS if hasattr(agent, key)}


def _methods(agent):
    n = _native()
    if any(getattr(getattr(agent, name), "__self__", None) is not agent
           for name in _METHODS if name != "_content_has_image_parts"):
        n._fail()
    owners = [n._qualified(getattr(agent, name), {"run_agent.py:AIAgent." + name}) for name in _METHODS]
    from agent.transports.codex import ResponsesApiTransport
    transport = agent._get_transport()
    if type(transport) is not ResponsesApiTransport or set(vars(transport)) - {"_last_issuer_kind"}:
        n._fail()
    for name in ("build_kwargs", "preflight_kwargs", "_resolve_issuer_kind"):
        if getattr(getattr(transport, name), "__self__", None) is not transport:
            n._fail()
        owners.append(n._qualified(getattr(transport, name), {"agent/transports/codex.py:ResponsesApiTransport." + name}))
    return owners


def _execute(agent, state, instructions, user_message, timeout):
    """Run the genuine builders with a closed replica and qualify actual calls."""
    from agent.transports.codex import ResponsesApiTransport
    from agent import chat_completion_helpers
    from run_agent import AIAgent
    n = _native()
    n._qualified(chat_completion_helpers.build_api_kwargs, {"agent/chat_completion_helpers.py:build_api_kwargs"})
    replica = SimpleNamespace(**deepcopy(state))
    replica._transport_cache = {"codex_responses": ResponsesApiTransport()}
    replica._native_transport_timeout = timeout
    for name in _METHODS:
        if name == "_resolved_api_call_timeout":
            replica._resolved_api_call_timeout = _timeout.__get__(replica)
        elif name == "_content_has_image_parts":
            replica._content_has_image_parts = AIAgent._content_has_image_parts
        else:
            setattr(replica, name, getattr(AIAgent, name).__get__(replica))
    owners, globals_used = {}, {}
    previous = sys.getprofile()
    if previous is not None:
        n._fail()  # Do not hide or replace an existing profiler's authority.

    def observe(frame, event, _arg):
        if event != "call":
            return
        code = frame.f_code
        path = Path(code.co_filename).resolve()
        if path.is_relative_to(n._ROOT):
            try:
                owner = n._qualified(SimpleNamespace(__code__=code))
                owners[owner["owner"]] = owner
                for name in code.co_names:
                    if name not in frame.f_globals:
                        continue
                    value = frame.f_globals[name]
                    if value is None or type(value) in (str, int, bool, float, dict, list, tuple, set, frozenset):
                        try:
                            globals_used[owner["owner"] + ":" + name] = n._hash(value)
                        except Exception:
                            # Provider objects in the registry are bound by the
                            # selected identity+data above, not repr/serialization.
                            if name not in {"_REGISTRY"}:
                                raise
            except Exception as error:
                raise _UnqualifiedRequest() from error
        elif code in _FROZEN_IMPORT_CODES or code in _DYNAMIC_STDLIB_CODES:
            return  # Exact interpreter-issued code, never a trusted filename.
        elif path.is_relative_to(_STDLIB) and "site-packages" not in path.parts:
            try:
                raw = path.read_bytes()
                import hashlib
                source_sha = hashlib.sha256(raw).hexdigest()
                key = (str(path), source_sha, sys.flags.optimize)
                candidates = n._CODE_CACHE.get(key)
                if candidates is None:
                    compiled = compile(raw, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
                    candidates = {item.co_qualname: item for item in n._code_objects(compiled)}
                    n._CODE_CACHE[key] = candidates
                if candidates.get(code.co_qualname) != code:
                    raise _UnqualifiedRequest()
                # Interpreter caches may skip stdlib helper calls on replay.
                # Validate each actual code object; do not mistake a cache hit
                # for a changed native request-producing dependency graph.
            except Exception as error:
                raise _UnqualifiedRequest() from error
        else:
            # An outer qualified builder and a forged native/stdlib filename
            # cannot qualify a delegated callback. Native fallbacks cannot
            # swallow this refusal and continue with an alternate request.
            raise _UnqualifiedRequest()

    try:
        sys.setprofile(observe)
        request = replica._build_api_kwargs([
            {"role": "system", "content": instructions}, {"role": "user", "content": user_message}])
        request = replica._get_transport().preflight_kwargs(request, allow_stream=False)
    except _UnqualifiedRequest:
        n._fail()
    finally:
        sys.setprofile(previous)
    return request, [owners[key] for key in sorted(owners)], globals_used


def prepare_request_dependencies():
    """Load builder modules before capture; bytecode cache creation is not input."""
    from agent import chat_completion_helpers
    from agent.transports.codex import ResponsesApiTransport
    # No profile discovery, client/auth resolution or request is executed here.
    return chat_completion_helpers.build_api_kwargs, ResponsesApiTransport


def freeze_request(agent, instructions, user_message):
    """Called inside native prompt construction, before any session hook."""
    n = _native()
    if agent.provider != "openai-codex" or agent.api_mode != "codex_responses":
        n._fail()
    state, methods, profile = _state(agent), _methods(agent), _profile()
    # Native-owned resolution, no AP auth access. Timeout is not a model input;
    # its run-budget-dependent value is intentionally outside model comparison.
    timeout = agent._resolved_api_call_timeout()
    request, owners, globals_used = _execute(agent, state, instructions, user_message, timeout)
    if _state(agent) != state or _profile() != profile or _methods(agent) != methods:
        n._fail()
    sources = {str((n._ROOT / owner["owner"].split(":", 1)[0]).resolve()):
               n._fingerprint(n._ROOT / owner["owner"].split(":", 1)[0]) for owner in [*methods, *owners, profile["lookup"], *profile["callbacks"]]}
    identity = {"state": state, "methods": methods, "profile": profile, "globals": globals_used,
                "model_sha256": n._hash(_model(request)), "sources": sources, "executed_code_owners": owners}
    return {**identity, "identity_sha256": n._hash(identity), "request": request, "code_owners":
            list({owner["owner"]: owner for owner in [*methods, *owners, profile["lookup"], *profile["callbacks"]]}.values()),
            "instructions": instructions, "user_message": user_message, "timeout": timeout}


def replay_request(agent, frozen):
    """Validate live dependencies/data and rebuild from pre-hook frozen state."""
    n = _native()
    if (_state(agent) != frozen["state"] or _methods(agent) != frozen["methods"] or _profile() != frozen["profile"]
            or any(n._fingerprint(path) != fp for path, fp in frozen["sources"].items())):
        n._fail()
    request, owners, globals_used = _execute(agent, frozen["state"], frozen["instructions"], frozen["user_message"], frozen["timeout"])
    if (_state(agent) != frozen["state"] or _profile() != frozen["profile"] or _methods(agent) != frozen["methods"]
            or n._hash(_model(request)) != frozen["model_sha256"] or globals_used != frozen["globals"]):
        n._fail()
    if owners != frozen["executed_code_owners"]:
        n._fail()
    return deepcopy(request)
