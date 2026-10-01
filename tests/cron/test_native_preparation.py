"""Full cron staged native path with real AIAgent/SDK and no provider network."""

from __future__ import annotations

import hashlib
import json
import base64
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cron.native import CronNativePlan, CronNativeSelection, CronNativeVariant


def _terminal(request, *, model="test-model"):
    events = [
        {"type": "response.output_text.delta", "delta": "review draft"},
        {"type": "response.completed", "response": {
            "id": "resp_cron_native", "model": model, "status": "completed", "output": [],
        }},
    ]
    content = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content=content + b"data: [DONE]\n\n", request=request,
    )


def _job():
    return {
        "id": "abcdef123456", "name": "native fixture", "prompt": "Review this source",
        "script": "producer.py", "native_preparation_provider": "fixture",
        "skills": ["fixture-skill"], "context_from": "123456abcdef",
        "model": "test-model", "provider": "openai-codex",
        "base_url": "https://example.invalid/v1", "schedule_display": "once",
        "deliver": "local", "enabled_toolsets": ["terminal"],
    }


@pytest.mark.parametrize("outcome", [
    "accepted", "archive_refused", "selection_unprepared", "invalid_later",
    "mcp_refused", "bundle_accepted", "plugin_accepted", "requirement_refused",
])
def test_full_staged_cron_capture_commit_dispatch_and_archive(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "producer.py").write_text('print("{\\"wakeAgent\\": false, \\"source\\": \\"fixture\\"}")\n')
    (tmp_path / ".env").write_text("")
    config = "agent:\n  max_turns: 1\nmodel:\n  context_length: 131072\nprefill_messages_file: prefill.json\n"
    if outcome == "mcp_refused":
        config += "mcp_servers:\n  fixture:\n    command: python\n"
    (tmp_path / "config.yaml").write_text(config)
    (tmp_path / "prefill.json").write_text(
        json.dumps([{"role": "user", "content": "prefill fixture context"}])
    )

    from cron import jobs, scheduler
    from hermes_cli import plugins
    import hermes_state

    monkeypatch.setattr(jobs, "HERMES_DIR", tmp_path)
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    predecessor = jobs.OUTPUT_DIR / "123456abcdef"
    predecessor.mkdir(parents=True)
    (predecessor / "prior.md").write_text("prior archive context")
    skill_dir = tmp_path / "skills" / "fixture-skill"
    skill_dir.mkdir(parents=True)
    skill_frontmatter = "prerequisites:\n  env_vars: [MISSING_PREPARATION_KEY]\n" if outcome == "requirement_refused" else ""
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fixture-skill\ndescription: fixture\n"
        + skill_frontmatter + "---\nfixture skill instructions"
    )
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path / "skills")
    usage_path = tmp_path / "skills" / ".usage.json"
    usage_before = b'{"fixture-skill":{"use_count":7}}\n'
    usage_path.write_bytes(usage_before)
    manager = plugins.PluginManager()
    context = plugins.PluginContext(SimpleNamespace(name="fixture", key="fixture"), manager)
    job = _job()
    if outcome == "bundle_accepted":
        bundles_dir = tmp_path / "skill-bundles"
        bundles_dir.mkdir()
        (bundles_dir / "fixture-bundle.yaml").write_text(
            "name: fixture-bundle\nskills:\n  - fixture-skill\n"
        )
        job["skills"] = ["fixture-bundle"]
    if outcome == "plugin_accepted":
        context.register_skill("fixture-skill", skill_dir / "SKILL.md")
        job["skills"] = ["fixture:fixture-skill"]
    if outcome == "requirement_refused":
        monkeypatch.setattr("tools.skills_tool._secret_capture_callback",
                            lambda *a, **k: pytest.fail("secret capture during preparation"))
        monkeypatch.setattr("tools.env_passthrough.register_env_passthrough",
                            lambda *a, **k: pytest.fail("environment registration during preparation"))
    observed = {"script": 0, "prepare": 0, "commit": 0, "sends": [], "archive": [], "marks": [], "metadata_probes": []}

    def blocked_get(*args, **kwargs):
        observed["metadata_probes"].append(args)
        raise AssertionError("network metadata request during native cron preparation")

    monkeypatch.setattr("requests.get", blocked_get)

    class Provider:
        def prepare(self, *, job, script_output):
            observed["prepare"] += 1
            assert job["id"] == "abcdef123456"
            assert '"wakeAgent": false' in script_output
            variants = tuple(
                CronNativeVariant(
                    form_id=f"budget-{i}",
                    script_stdout=json.dumps({"wakeAgent": False, "form": f"budget-{i}"}),
                    owner_binding="source-fixture",
                ) for i in range(10)
            ) + (CronNativeVariant(
                form_id="" if outcome == "invalid_later" else "selected-current",
                script_stdout=json.dumps({"wakeAgent": False, "form": "selected-current"}),
                owner_binding="source-fixture",
            ),)

            def commit(records):
                observed["commit"] += 1
                assert len(records) == 11
                assert len({r.form_id for r in records}) == 11
                assert len({r.preparation_token for r in records}) == 11
                assert all(r.request.body_bytes > 0 for r in records)
                assert all(r.profile.model == "test-model" for r in records)
                assert records[-1].form_id == "selected-current"
                selected = records[-1]
                return CronNativeSelection(
                    form_id=selected.form_id,
                    preparation_token=("unprepared" if outcome == "selection_unprepared"
                                       else selected.preparation_token),
                    owner_binding=selected.owner_binding,
                    archive_callback=archive,
                    delegate_factory=lambda: httpx.MockTransport(send),
                )

            return CronNativePlan(variants, commit)

    def send(request):
        body = request.read()
        assert b"selected-current" in body
        assert b"budget-9" not in body
        assert b"prior archive context" in body
        assert b"fixture skill instructions" in body
        assert b"prefill fixture context" in body
        assert b"scheduled cron job" in body
        observed["sends"].append(request)
        return _terminal(request)

    def archive(item):
        assert isinstance(item.archive_path, Path) and item.archive_path.exists()
        assert item.archive_sha256 == hashlib.sha256(item.archive_path.read_bytes()).hexdigest()
        assert item.attempt.provider_response_id == "resp_cron_native"
        assert item.form_id == "selected-current"
        assert item.outer_callback_count == 1 and item.outer_retry_count == 0
        observed["archive"].append(item)
        return {"fixture_sidecar": True} if outcome in (
            "accepted", "bundle_accepted", "plugin_accepted"
        ) else None

    context.register_cron_native_preparation_provider("fixture", Provider())
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_: {"provider": "openai-codex", "api_key": "fixture-only",
                    "base_url": "https://example.invalid/v1", "api_mode": "codex_responses"},
    )
    original_script = scheduler._run_job_script

    def script_once(path):
        observed["script"] += 1
        return original_script(path)

    monkeypatch.setattr(scheduler, "_run_job_script", script_once)
    if outcome == "mcp_refused":
        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools",
                            lambda: pytest.fail("MCP discovery during refused preparation"))
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(
        scheduler, "mark_job_run",
        lambda jid, success, error=None, delivery_error=None:
            observed["marks"].append((jid, success, error)),
    )

    assert scheduler.run_one_job(job) is (outcome != "archive_refused")
    assert observed["script"] == observed["prepare"] == 1
    refused = ("invalid_later", "mcp_refused", "requirement_refused")
    assert observed["commit"] == (0 if outcome in refused else 1)
    expected_sends = 0 if outcome in ("selection_unprepared", *refused) else 1
    assert len(observed["sends"]) == len(observed["archive"]) == expected_sends
    assert observed["marks"][0][:2] == (
        "abcdef123456", outcome in ("accepted", "bundle_accepted", "plugin_accepted")
    )
    if outcome == "archive_refused":
        assert "archive callback refused" in observed["marks"][0][2]
    assert usage_path.read_bytes() == usage_before
    assert observed["metadata_probes"] == []


def test_missing_provider_and_no_agent_never_run_script_or_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from cron import scheduler
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
    runs = []
    monkeypatch.setattr(scheduler, "_run_job_script", lambda path: runs.append(path))
    job = _job()
    assert scheduler.run_job(job)[0] is False
    job["no_agent"] = True
    assert scheduler.run_job(job)[0] is False
    assert runs == []


def test_protected_script_failure_never_prepares_or_dispatches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from cron import scheduler
    from hermes_cli import plugins

    calls = []
    provider = SimpleNamespace(prepare=lambda **kw: calls.append("prepare"))
    manager = plugins.PluginManager()
    plugins.PluginContext(SimpleNamespace(name="fixture", key="fixture"), manager).register_cron_native_preparation_provider("fixture", provider)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
    monkeypatch.setattr(scheduler, "_run_job_script", lambda path: (False, "failed"))

    success, _, _, error = scheduler.run_job(_job())
    assert success is False and error == "protected native script failed"
    assert calls == []


def test_invalid_native_forms_fail_before_agent_or_commit():
    from cron.native import CronNativePlan, prepare_and_commit
    from agent.native_request import NativeAdmissionError

    calls = []
    with pytest.raises(NativeAdmissionError, match="cron_native_form_invalid"):
        prepare_and_commit(
            job=_job(),
            plan=CronNativePlan((CronNativeVariant("", "source", "owner"),),
                                lambda records: calls.append("commit")),
            build_prompt=lambda job, cached: "prompt",
            make_agent=lambda: calls.append("agent"),
            persist_user_message="source",
        )
    assert calls == []


def test_cold_custom_endpoint_constructor_refuses_before_metadata_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  max_turns: 1\n")
    (tmp_path / ".env").write_text("")
    from agent.native_request import NativeAdmissionError
    from run_agent import AIAgent
    probes = []

    def blocked_get(*args, **kwargs):
        probes.append(args)
        raise AssertionError("metadata network request before native preparation")

    monkeypatch.setattr("requests.get", blocked_get)
    with pytest.raises(NativeAdmissionError, match="cron_native_constructor_config_drift"):
        AIAgent(
            model="cold-custom", provider="custom", api_mode="codex_responses",
            api_key="fixture-only", base_url="https://example.invalid/v1",
            preparation_context_length=131072, quiet_mode=True, skip_memory=True,
        )
    assert probes == []


@pytest.mark.parametrize("entitled", [False, True])
@pytest.mark.parametrize("outcome", ["selected", "invalid_later", "commit_refused"])
def test_real_resolved_codex_credential_is_frozen_after_resolution(tmp_path, monkeypatch, outcome, entitled):
    """Real configuration freezes Nous guidance before SDK preparation."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    token_body = base64.urlsafe_b64encode(
        json.dumps({"exp": int(time.time()) + 86400}).encode()
    ).decode().rstrip("=")
    access_token = f"eyJhbGciOiJub25lIn0.{token_body}.fixture"
    auth_path = home / "auth.json"
    nous_token_body = base64.urlsafe_b64encode(json.dumps({
        "exp": int(time.time()) + 86400, "paid_access": entitled,
        "sub": "fixture-account",
    }).encode()).decode().rstrip("=")
    nous_token = f"eyJhbGciOiJub25lIn0.{nous_token_body}.fixture"
    auth_data = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {"auth_mode": "chatgpt", "tokens": {
            "access_token": access_token, "refresh_token": "synthetic-refresh",
        }}},
        "credential_pool": {"openai-codex": [{
            "id": "stale-device", "source": "device_code", "auth_type": "oauth",
            "label": "fixture", "priority": 0, "access_token": "stale-token",
            "refresh_token": "stale-refresh",
        }]},
    }
    if entitled:
        auth_data["providers"]["nous"] = {"access_token": nous_token}
    auth_path.write_text(json.dumps(auth_data))
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5.5\n  provider: openai-codex\n"
        "  context_length: 272000\n  max_tokens: 8192\n"
        "agent:\n  max_turns: 1\ncompression:\n  enabled: false\n"
        "tools:\n  tool_search:\n    enabled: off\n"
        "prefill_messages_file: prefill.json\n"
    )
    (home / "prefill.json").write_text(json.dumps([
        {"role": "user", "content": "prefill fixture context"},
    ]))
    (home / ".env").write_text("")
    scripts = home / "scripts"
    scripts.mkdir()
    (scripts / "producer.py").write_text('print("source fixture")\n')
    predecessor = home / "cron" / "output" / "123456abcdef"
    predecessor.mkdir(parents=True)
    (predecessor / "prior.md").write_text("prior archive context")
    skill_dir = home / "skills" / "fixture-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fixture-skill\ndescription: fixture\n---\nfixture skill instructions"
    )
    project_dir = home / "project"
    project_dir.mkdir()
    (project_dir / "AGENTS.md").write_text("fixture project instructions")

    from cron import jobs, scheduler
    from hermes_cli import plugins, runtime_provider
    from agent import credential_pool, native_request
    from hermes_cli import nous_account, nous_subscription
    import run_agent
    import hermes_state

    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    manager = plugins.PluginManager()
    context = plugins.PluginContext(SimpleNamespace(name="fixture", key="fixture"), manager)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
    observed = {"phase1": False, "resolver": 0, "pool_loads": 0,
                "pool_writes": 0, "commit": 0, "sends": 0, "marks": [],
                "nous_builds": 0, "nous_prompt": None, "captures": [], "drift_auth": None}
    post_resolution = {}
    real_resolve = runtime_provider.resolve_runtime_provider
    real_load_pool = credential_pool.load_pool
    real_write_pool = credential_pool.write_credential_pool
    real_nous_info = nous_account.get_nous_portal_account_info
    real_nous_builder = run_agent.build_nous_subscription_prompt
    real_capture = native_request.CaptureTransport.handle_request

    def observed_nous_info(*args, **kwargs):
        if observed["phase1"]:
            pytest.fail("Nous account manager re-entered after configuration freeze")
        return real_nous_info(*args, **kwargs)

    def observed_nous_builder(valid_names):
        assert not observed["phase1"]
        rendered = real_nous_builder(valid_names)
        observed["nous_builds"] += 1
        observed["nous_prompt"] = rendered
        observed["phase1"] = True
        return rendered

    def observed_capture(transport, request):
        body = request.read()
        observed["captures"].append(body)
        if len(observed["captures"]) == 1:
            # A real temp-store entitlement flip after the configuration
            # boundary cannot change any later provisional or selected prompt.
            changed = json.loads(auth_path.read_text())
            if entitled:
                changed["providers"].pop("nous")
            else:
                changed["providers"]["nous"] = {"access_token": nous_token.replace(
                    nous_token_body,
                    base64.urlsafe_b64encode(json.dumps({
                        "exp": int(time.time()) + 86400, "paid_access": True,
                        "sub": "late-fixture-account",
                    }).encode()).decode().rstrip("="),
                )}
            auth_path.write_text(json.dumps(changed))
            observed["drift_auth"] = auth_path.read_bytes()
        return real_capture(transport, request)

    def observed_load_pool(provider):
        if observed["phase1"]:
            pytest.fail("credential pool loaded after resolved handoff")
        observed["pool_loads"] += 1
        return real_load_pool(provider)

    def observed_write_pool(*args, **kwargs):
        if observed["phase1"]:
            pytest.fail("credential pool persisted after resolved handoff")
        observed["pool_writes"] += 1
        return real_write_pool(*args, **kwargs)

    monkeypatch.setattr(runtime_provider, "load_pool", observed_load_pool)
    monkeypatch.setattr(credential_pool, "load_pool", observed_load_pool)
    monkeypatch.setattr(credential_pool, "write_credential_pool", observed_write_pool)
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", observed_nous_info)
    monkeypatch.setattr(nous_subscription, "get_nous_portal_account_info", observed_nous_info)
    monkeypatch.setattr(run_agent, "build_nous_subscription_prompt", observed_nous_builder)
    monkeypatch.setattr(native_request.CaptureTransport, "handle_request", observed_capture)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("portal network request"))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", home / "skills")

    def observed_resolver(**kwargs):
        if observed["phase1"]:
            pytest.fail("provider resolved again after handoff")
        resolved = real_resolve(**kwargs)
        observed["resolver"] += 1
        assert resolved["provider"] == "openai-codex"
        assert resolved["api_mode"] == "codex_responses"
        assert resolved["api_key"] == access_token
        assert resolved["base_url"] == "https://chatgpt.com/backend-api/codex"
        post_resolution["auth"] = auth_path.read_bytes()
        post_resolution["config"] = (home / "config.yaml").read_bytes()
        return resolved

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", observed_resolver)
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("metadata GET"))

    class Provider:
        def prepare(self, *, job, script_output):
            assert not observed["phase1"]
            assert "source fixture" in script_output
            variants = tuple(CronNativeVariant(
                f"budget-{i}", f"source form budget-{i}", "source-fixture",
            ) for i in range(10)) + (CronNativeVariant(
                "" if outcome == "invalid_later" else "selected-current",
                "source form selected-current", "source-fixture",
            ),)

            def commit(records):
                observed["commit"] += 1
                assert observed["phase1"] and len(records) == 11
                assert len(observed["captures"]) == 11
                assert all(record.request.body_sha256 == hashlib.sha256(body).hexdigest()
                           for record, body in zip(records, observed["captures"]))
                assert all(r.profile.model == "gpt-5.5" for r in records)
                assert all(r.profile.context_length == 272000 for r in records)
                assert all(r.profile.max_tokens == 8192 for r in records)
                selected = records[-1]
                return CronNativeSelection(
                    selected.form_id,
                    "refused" if outcome == "commit_refused" else selected.preparation_token,
                    selected.owner_binding,
                    lambda item: {"receipt": True},
                    lambda: httpx.MockTransport(send),
                )

            return CronNativePlan(variants, commit)

    def send(request):
        observed["sends"] += 1
        assert observed["phase1"] and outcome == "selected"
        assert request.headers["Authorization"] == f"Bearer {access_token}"
        assert b"selected-current" in request.read()
        assert request.read() == observed["captures"][-1]
        assert b"fixture skill instructions" in request.read()
        assert b"prior archive context" in request.read()
        assert b"prefill fixture context" in request.read()
        assert b"fixture project instructions" in request.read()
        return _terminal(request, model="gpt-5.5")

    context.register_cron_native_preparation_provider("fixture", Provider())
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda jid, success, error=None, delivery_error=None:
                        observed["marks"].append((jid, success, error)))
    job = _job()
    job["model"] = "gpt-5.5"
    job.pop("base_url")
    job["enabled_toolsets"] = ["terminal"]
    job["workdir"] = str(project_dir)
    scheduler.run_one_job(job)
    assert observed["resolver"] == 1
    assert observed["nous_builds"] == 1
    assert isinstance(observed["nous_prompt"], str)
    assert bool(observed["nous_prompt"]) is entitled
    if entitled:
        assert "# Nous Subscription" in observed["nous_prompt"]
    assert len(observed["captures"]) == (10 if outcome == "invalid_later" else 11)
    system_prompts = []
    for body in observed["captures"]:
        payload = json.loads(body)
        content = [payload.get("instructions", "")]
        content.extend(item.get("content", "") for item in payload.get("input", [])
                       if isinstance(item, dict))
        content = [item for item in content if isinstance(item, str)]
        if entitled:
            matches = [item for item in content if observed["nous_prompt"] in item]
            assert len(matches) == 1
            system_prompts.append(matches[0])
        else:
            assert not any("# Nous Subscription" in item for item in content)
            system_prompts.append(content[0])
        assert b"fixture skill instructions" in body
        assert b"prior archive context" in body
        assert b"prefill fixture context" in body
        assert b"fixture project instructions" in body
    assert len(set(system_prompts)) == 1
    assert observed["pool_loads"] >= 1 and observed["pool_writes"] >= 1
    assert observed["commit"] == (0 if outcome == "invalid_later" else 1)
    assert observed["sends"] == (1 if outcome == "selected" else 0)
    assert observed["marks"][0][1] is (outcome == "selected")
    assert observed["drift_auth"] != post_resolution["auth"]
    assert auth_path.read_bytes() == observed["drift_auth"]
    assert (home / "config.yaml").read_bytes() == post_resolution["config"]
