"""Disposable real-native construction controls; no model/provider dispatch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CASE = sys.argv[1]


class StoppedBeforeProvider(BaseException):
    pass


network_attempts = []
inspection_events = []
inspecting = False


def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        network_attempts.append(event)
        raise StoppedBeforeProvider("external network denied")
    if inspecting:
        if event.startswith("socket.") or event in {"subprocess.Popen", "os.fork", "os.forkpty"}:
            inspection_events.append(event)
            raise StoppedBeforeProvider("inspection external operation")
        if event == "open":
            path, mode, flags = args
            name = str(path).replace("\\", "/").lower()
            write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if write or name.endswith("/auth.json") or "/.env" in name:
                inspection_events.append(event)
                raise StoppedBeforeProvider("inspection credential/write operation")


sys.addaudithook(audit)
with tempfile.TemporaryDirectory() as directory:
    home = Path(directory)
    hermes = home / "hermes"
    hermes.mkdir()
    os.environ.update(HOME=str(home), HERMES_HOME=str(hermes))
    os.chdir(home)
    config = hermes / "config.yaml"
    config.write_text("model:\n  default: gpt-6.1-sol\n  provider: openai-codex\n  context_length: 272000\nagent:\n  environment_probe: false\ntools:\n  tool_search:\n    enabled: off\n")
    soul = hermes / "SOUL.md"
    soul.write_text("# Native identity\nPreserve native identity, configured guidance and operator decisions.\n")
    auth = hermes / "auth.json"
    auth.write_text(json.dumps({"version": 1, "providers": {"openai-codex": {"access_token": "disposable-fixture-not-a-credential"}}}))
    import run_agent
    from run_agent import AIAgent
    from hermes_cli import plugins
    import agent.prompt_construction as construction
    from agent.prompt_inspection_scope import PromptInspectionUnavailable

    options = {"model": "gpt-6.1-sol", "api_key": "inspect-only", "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex", "api_mode": "codex_responses", "enabled_toolsets": ["file"],
        "disabled_toolsets": [], "quiet_mode": True, "skip_memory": True, "skip_context_files": CASE != "custom_context",
        "load_soul_identity": True, "platform": "cron", "session_id": "cron_aaaaaaaaaaaa_20261028_210000",
        "api_max_attempts": 3, "max_run_seconds": 600}
    if CASE == "prefill":
        options["prefill_messages"] = [{"role": "assistant", "content": "foreign prefill"}]
    if CASE == "ephemeral":
        options["ephemeral_system_prompt"] = "foreign ephemeral context"
    agent = AIAgent(**options)
    manager = plugins.PluginManager()
    plugins._plugin_manager = manager
    context = plugins.PluginContext(plugins.PluginManifest(name="native-construction-test"), manager)
    results = {"schema_version": 1, "case": CASE, "phase": None, "provider_dispatches": 0}

    def rejected(call):
        try:
            call()
        except (PromptInspectionUnavailable, TypeError):
            return True
        return False

    invalid_cases = {"prefill", "ephemeral", "custom_context", "custom_system", "history", "foreign_callback", "registration_veto"}
    request_mutations = {"changed_profile_callback", "changed_profile_data", "changed_profile_registry", "changed_message_converter", "changed_tools_converter", "changed_request_config", "changed_converter_data", "changed_transport_callback", "spoofed_stdlib_converter", "builtin_tools_converter"}
    mutations = {"changed_soul", "changed_auth", "changed_config", "changed_tools", "changed_environment", "changed_cached_prompt"}
    actual_cases = {"native_request", "mutated_initial_instructions", "extra_assistant_input"}
    observed = {}

    def started(**_event):
        global inspecting
        results["phase"] = "native_session_start_before_request"
        if CASE in invalid_cases:
            assert rejected(lambda: construction.read_prompt_construction_snapshot(session_id=agent.session_id))
            results["refused"] = True
            raise StoppedBeforeProvider()
        handle = construction.read_prompt_construction_snapshot(session_id=agent.session_id)
        before = {str(p.relative_to(hermes)): hashlib.sha256(p.read_bytes()).hexdigest() for p in hermes.rglob("*") if p.is_file()}
        inspecting = True
        try:
            view = handle.reconstruct()
        finally:
            inspecting = False
        after = {str(p.relative_to(hermes)): hashlib.sha256(p.read_bytes()).hexdigest() for p in hermes.rglob("*") if p.is_file()}
        assert before == after
        assert view.instructions == agent._cached_system_prompt and "Native identity" in view.instructions
        assert list(view.tools) == agent.tools and len(view.tools) == 4
        assert view.user_message_sha256 == hashlib.sha256(b"Native cron job prompt and selected data.").hexdigest()
        results.update(replayed=True, instructions_sha256=hashlib.sha256(view.instructions.encode()).hexdigest(), tools=len(view.tools))
        observed["handle"] = handle
        if CASE == "fork_snapshot":
            child = os.fork()
            if child == 0:
                os._exit(0 if rejected(handle.reconstruct) and rejected(handle.expected_request) else 1)
            _, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            results["forked_snapshot_refused"] = True
        if CASE == "forged":
            fake = object.__new__(construction.PromptConstructionSnapshot)
            assert rejected(lambda: fake.reconstruct())
            assert rejected(lambda: construction.PromptConstructionSnapshot())
            assert rejected(lambda: pickle.dumps(handle))
            assert rejected(lambda: construction._reconstruct({"accepted": True}))
            assert rejected(lambda: construction._reconstruct(SimpleNamespace(accepted=True)))
            prior = handle.expected_request()
            construction.note_api_construction(agent, [], {"accepted": True})
            assert handle.expected_request() == prior
            results["forgeries_refused"] = True
        if CASE in request_mutations:
            import providers
            import agent.codex_responses_adapter as adapter
            if CASE == "changed_profile_callback":
                providers.get_provider_profile = lambda _name: SimpleNamespace(build_api_kwargs_extras=lambda **_kwargs: ({}, {"reasoning": {"effort": "low"}}))
            elif CASE == "changed_profile_data":
                providers.get_provider_profile("openai-codex").default_max_tokens = 7
            elif CASE == "changed_profile_registry":
                from providers.base import ProviderProfile
                providers._REGISTRY["openai-codex"] = ProviderProfile(name="openai-codex", api_mode="codex_responses")
            elif CASE == "changed_message_converter":
                adapter._chat_messages_to_responses_input = lambda *_args, **_kwargs: [{"role": "user", "content": "foreign"}]
            elif CASE == "spoofed_stdlib_converter":
                import urllib.parse
                namespace = {"original": adapter._chat_messages_to_responses_input}
                exec(compile("def convert(*args, **kwargs): return original(*args, **kwargs)", urllib.parse.__file__, "exec"), namespace)
                adapter._chat_messages_to_responses_input = namespace["convert"]
            elif CASE == "builtin_tools_converter":
                adapter._responses_tools = list
            elif CASE == "changed_tools_converter":
                adapter._responses_tools = lambda *_args, **_kwargs: [{"type": "function", "name": "foreign"}]
            elif CASE == "changed_converter_data":
                adapter._RESPONSES_BUILTIN_TOOL_TYPES = adapter._RESPONSES_BUILTIN_TOOL_TYPES | {"foreign"}
            elif CASE == "changed_transport_callback":
                agent._get_transport().build_kwargs = lambda **_kwargs: handle.expected_request()
            else:
                agent.reasoning_config = {"effort": "low"}
            assert rejected(handle.expected_request)
            results["changed_request_dependency_refused"] = True
            raise StoppedBeforeProvider()
        if CASE in mutations:
            if CASE == "changed_soul":
                old = soul.stat()
                content = soul.read_text()
                soul.write_text(content.replace("Preserve", "REPLACE!"))
                os.utime(soul, ns=(old.st_atime_ns, old.st_mtime_ns))
            elif CASE == "changed_auth":
                auth.write_text(auth.read_text() + " ")
            elif CASE == "changed_config":
                config.write_text(config.read_text() + "# changed\n")
            elif CASE == "changed_tools":
                agent.tools[0]["function"]["description"] += " foreign modification"
            elif CASE == "changed_environment":
                os.environ["TERMINAL_CWD"] = str(home)
            else:
                agent._cached_system_prompt += " foreign modification"
            assert rejected(handle.reconstruct)
            results["changed_source_refused"] = True
        if CASE not in actual_cases:
            raise StoppedBeforeProvider()

    def execution(*, request, next_call, **event):
        global inspecting
        results["phase"] = "native_final_execution_boundary"
        inspecting = True
        try:
            handle = construction.read_prompt_construction_snapshot(session_id=event["session_id"], api_request_id=event["api_request_id"])
            view = handle.reconstruct()
            expected = handle.expected_request()
            proof = handle.request_proof()
        finally:
            inspecting = False
        assert proof["schema_version"] == 2
        assert proof["scope"].startswith("native_epoch_frozen_request")
        if CASE == "native_request":
            model_input = lambda value: {key: item for key, item in value.items() if key not in {"timeout", "stream", "extra_headers"}}
            if model_input(expected) != model_input(request):
                results["request_difference"] = {key: {"expected": repr(expected.get(key))[:140], "actual": repr(request.get(key))[:140]}
                                                 for key in expected.keys() | request.keys() if expected.get(key) != request.get(key)}
                raise StoppedBeforeProvider()
            assert request["instructions"] == view.instructions
            modified = handle.expected_request()
            modified["instructions"] = "caller-edited copy"
            assert handle.expected_request()["instructions"] == view.instructions
            results["complete_model_input_equal"] = True
        else:
            model_input = lambda value: {key: item for key, item in value.items() if key not in {"timeout", "stream", "extra_headers"}}
            assert model_input(expected) != model_input(request)
            assert expected["instructions"] == view.instructions
            results["foreign_request_refused_by_comparison"] = True
        raise StoppedBeforeProvider()

    def guarded_started(**event):
        try:
            return started(**event)
        except Exception as error:
            results["unexpected_inspection_failure"] = type(error).__name__
            raise StoppedBeforeProvider("inspection control failed") from error
    context.register_hook("on_session_start", guarded_started)
    if CASE in actual_cases:
        context.register_middleware("llm_execution", execution)
    if CASE in {"mutated_initial_instructions", "extra_assistant_input"}:
        def mutation(*, request, **_event):
            request = json.loads(json.dumps(request))
            if CASE == "mutated_initial_instructions":
                request["instructions"] = "arbitrary initial instructions"
            else:
                request["input"].append({"role": "assistant", "content": [{"type": "output_text", "text": "foreign history"}]})
            return {"request": request}
        context.register_middleware("llm_request", mutation)
    if CASE == "foreign_callback":
        run_agent.build_nous_subscription_prompt = lambda *_args, **_kwargs: "unissued callback approval"
    if CASE == "registration_veto":
        def veto(event, _args):
            if event == "sys.addaudithook":
                raise RuntimeError("reject registration")
        sys.addaudithook(veto)
    kwargs = {}
    if CASE == "custom_system":
        kwargs["system_message"] = "custom unqualified context"
    if CASE == "history":
        kwargs["conversation_history"] = [{"role": "assistant", "content": "foreign history"}]
    def unexpected_client(*_args, **_kwargs):
        results["unexpected_client_entry"] = True
        raise StoppedBeforeProvider("unexpected provider client entry")
    agent._create_request_openai_client = unexpected_client
    try:
        agent.run_conversation("Native cron job prompt and selected data.", **kwargs)
    except StoppedBeforeProvider:
        pass
    assert results["phase"] is not None
    assert not results.get("unexpected_client_entry"), results
    assert network_attempts == [], network_attempts
    assert inspection_events == [], inspection_events
    if CASE in actual_cases:
        assert results["phase"] == "native_final_execution_boundary", results
        assert "request_difference" not in results, results
    elif CASE not in invalid_cases:
        assert results.get("replayed"), results
    results.update(network_attempts=network_attempts, inspection_events=inspection_events,
                   scope="disposable native source construction; tool_search off/environment_probe false; no provider dispatch")
    print(json.dumps(results, sort_keys=True))
