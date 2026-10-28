"""Native construction primitive shares constructor semantics; no admission claim."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.prompt_inspection import inspect_cron_prompt, native_prompt_helpers
from agent.prompt_state import apply_prompt_configuration, apply_prompt_tools, construct_cron_prompt_state
from agent.system_prompt import build_system_prompt_parts

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {}}}}]


def inputs(**changes):
    value = dict(config={"agent": {"environment_probe": False}}, model="gpt-6.1-sol",
                 provider="openai-codex", tools=deepcopy(TOOLS), context_length=272000,
                 prefill_messages=[],
                 session_id="cron_aaaaaaaaaaaa_20261028_210000", workdir_present=False)
    value.update(changes)
    return value


def test_config_and_tool_state_match_the_shared_constructor_operations():
    cfg = {"agent": {"tool_use_enforcement": True, "task_completion_guidance": False,
                     "parallel_tool_call_guidance": False, "environment_probe": False},
           "platform_hints": {"cron": {"append": "Keep this native scheduling preference."}}}
    normal = SimpleNamespace()
    apply_prompt_configuration(normal, cfg)
    apply_prompt_tools(normal, deepcopy(TOOLS))
    inspected = construct_cron_prompt_state(**inputs(config=cfg))
    for name in vars(normal):
        assert getattr(inspected, name) == getattr(normal, name)
    assert inspected.tools == TOOLS
    assert inspected.valid_tool_names == {"read_file"}
    assert inspected.skip_memory is True
    assert inspected.skip_context_files is True
    assert inspected.load_soul_identity is True
    assert construct_cron_prompt_state(**inputs(workdir_present=True)).skip_context_files is False


def test_inspection_does_not_replace_nonempty_tools_or_share_mutable_input():
    value = inputs()
    state = construct_cron_prompt_state(**value)
    value["tools"][0]["function"]["name"] = "changed"
    value["config"]["agent"]["environment_probe"] = True
    assert state.valid_tool_names == {"read_file"}
    assert state._environment_probe is False
    assert state.tools[0]["function"]["name"] == "read_file"


def test_native_soul_and_guidance_survive_public_builder_inspection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "SOUL.md").write_text("# Native identity\nPreserve the operator's native identity and guidance.\n")
    value = inputs(config={"agent": {"environment_probe": False},
                           "platform_hints": {"cron": {"append": "Preserve configured cron guidance."}}})
    # Optional entitlement resolution is outside this pure state test. Both
    # paths use the same native implementation dependency under one control.
    with patch("agent.prompt_builder.build_nous_subscription_prompt", return_value="native entitlement guidance"):
        normal = build_system_prompt_parts(construct_cron_prompt_state(**value), helpers=native_prompt_helpers())
        inspected = inspect_cron_prompt(**value)
    assert inspected["parts"] == normal
    assert "Native identity" in inspected["instructions"]
    assert "native entitlement guidance" in inspected["instructions"]
    assert "Preserve configured cron guidance." in inspected["instructions"]
    assert "parallel" in inspected["instructions"].lower()
    assert inspected["tools"] == TOOLS
    assert inspected["scope"] == "native_resolved_prompt_construction_not_source_or_request_attestation"


def test_denied_dependency_is_not_downgraded_to_empty_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    (tmp_path / "SOUL.md").write_text("Preserve native identity.\n")

    class Denied(BaseException):
        pass

    # Native optional dependencies deliberately catch Exception. An isolated
    # inspector's mandatory denial must survive, not fabricate empty parity.
    with patch("agent.prompt_builder.build_nous_subscription_prompt", side_effect=Denied):
        with pytest.raises(Denied):
            inspect_cron_prompt(**inputs())


def test_unresolved_constructor_inputs_refuse():
    for changes in [{"context_length": 0}, {"context_length": None}, {"provider": ""},
                    {"model": ""}, {"tools": None}, {"workdir_present": None}]:
        with pytest.raises(ValueError, match="native_prompt_state_unresolved"):
            construct_cron_prompt_state(**inputs(**changes))


def test_resolved_prefill_is_retained_not_silently_discarded():
    history = [{"role": "assistant", "content": "Native prefill must remain visible to admission."}]
    state = construct_cron_prompt_state(**inputs(prefill_messages=history))
    assert state.prefill_messages == history
    assert state.prefill_messages is not history


def test_real_constructor_matches_inspection_without_inspection_side_effects(tmp_path):
    """Real resolved native schemas and native SOUL; no fake inspection witness.

    Runtime baseline initialization may create isolated fixture state. Only the
    subsequent inspection interval has the strict no-side-effect assertion.
    The test grants no source provenance or production request authority.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    home = tmp_path / "home"
    hermes = home / "hermes"
    hermes.mkdir(parents=True)
    (hermes / "config.yaml").write_text("model:\n  default: gpt-6.1-sol\n  provider: openai-codex\n  context_length: 272000\nagent:\n  environment_probe: false\n")
    (hermes / "SOUL.md").write_text("# Native SOUL\nRetain this native identity and every configured guideline.\n")
    code = r'''
import os,sys,json,hashlib
from pathlib import Path
from run_agent import AIAgent
from hermes_cli.config import load_config
from agent.prompt_inspection import inspect_cron_prompt
from agent.prompt_state import cron_prompt_options
root=Path(os.environ["HERMES_HOME"])
agent=AIAgent(model="gpt-6.1-sol",api_key="inspect-only",base_url="https://chatgpt.com/backend-api/codex",
    provider="openai-codex",api_mode="codex_responses",enabled_toolsets=["file"],disabled_toolsets=[],
    quiet_mode=True,**cron_prompt_options(workdir_present=False),
    session_id="cron_aaaaaaaaaaaa_20261028_210000",api_max_attempts=3,max_run_seconds=600)
normal=agent._build_system_prompt()
config=load_config()
assert agent.tools and "read_file" in agent.valid_tool_names
snapshot=lambda:{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
before=snapshot()
class Denied(BaseException):pass
denials=[]
inspection_active=True
def audit(event,args):
    if not inspection_active:return
    bad=None
    if event.startswith("socket.") or event in {"subprocess.Popen","os.system","os.posix_spawn"}:bad="network_or_process"
    if event in {"os.rename","os.remove","os.rmdir","os.link","os.symlink","os.chmod","os.truncate","os.utime","sqlite3.connect"}:bad="write_intent"
    # EEXIST on an existing directory is not a persistent mutation. Reject
    # actual creation and separately compare the full fixture postimage.
    if event=="os.mkdir" and not Path(args[0]).is_dir():bad="directory_creation"
    if event=="open":
        path,mode,flags=args
        name=str(path).replace("\\","/").lower()
        if any(token in name for token in ["/.env","/auth.json","/credentials","/secrets/","/.codex/"]):bad="credential_read"
        if (isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):bad="write_intent"
    if bad:
        denials.append({"event":event,"kind":bad})
        raise Denied(bad)
sys.addaudithook(audit)
result=inspect_cron_prompt(config=config,model=agent.model,provider=agent.provider,tools=agent.tools,
    context_length=agent.context_compressor.context_length,session_id=agent.session_id,workdir_present=False,
    prefill_messages=agent.prefill_messages)
assert result["instructions"]==normal
assert result["tools"]==agent.tools
assert "Native SOUL" in result["instructions"]
assert snapshot()==before
inspection_active=False
print(json.dumps({"native_instructions_equal":True,"native_nonempty_tools_equal":True,"denials":denials,
    "persistent_file_changes":0,"scope":"warm native process; explicit context config; no environment probe; no credentials configured; no production attestation"}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[2],
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
                                 "HERMES_HOME": str(hermes), "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["native_instructions_equal"]
    assert report["native_nonempty_tools_equal"]
    assert report["denials"] == []
    assert report["persistent_file_changes"] == 0


def test_inspection_never_seeds_missing_soul_or_accepts_a_different_fallback(tmp_path, monkeypatch):
    from agent.prompt_inspection_scope import PromptInspectionUnavailable

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(PromptInspectionUnavailable, match="native_instruction_inspection_unavailable"):
        inspect_cron_prompt(**inputs())
    assert not (tmp_path / "SOUL.md").exists()


def test_inspection_scope_restores_normal_profile_preparation_after_refusal(tmp_path, monkeypatch):
    from agent.prompt_inspection_scope import PromptInspectionUnavailable, prompt_source_inspection_active
    from hermes_cli.config import ensure_hermes_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(PromptInspectionUnavailable):
        inspect_cron_prompt(**inputs())
    assert prompt_source_inspection_active() is False
    ensure_hermes_home()
    assert (tmp_path / "SOUL.md").is_file()
    assert (tmp_path / "sessions").is_dir()


def test_read_only_inspection_skips_only_optional_skills_cache_persistence(tmp_path, monkeypatch):
    from agent.prompt_builder import _write_skills_snapshot
    from agent.prompt_inspection_scope import prompt_source_inspection

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("agent.prompt_builder.atomic_json_write") as writer:
        with prompt_source_inspection():
            _write_skills_snapshot(tmp_path, {}, [], {})
        writer.assert_not_called()
        _write_skills_snapshot(tmp_path, {}, [], {})
        writer.assert_called_once()


@pytest.mark.parametrize("dependency", ["network", "credential", "credential_alias", "write"])
def test_public_inspector_enforces_its_mandatory_native_boundary(tmp_path, monkeypatch, dependency):
    from agent.prompt_inspection_scope import PromptInspectionUnavailable, prompt_source_inspection_active

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "SOUL.md").write_text("Keep the native identity.\n")
    credential = tmp_path / ".env"
    credential.write_text("disposable fixture, never opened by inspection\n")
    alias = tmp_path / "ordinary-note.md"
    alias.symlink_to(credential)
    destination = tmp_path / "unwanted-write"

    def unavailable(*_args, **_kwargs):
        if dependency == "network":
            import socket
            return socket.socket()
        if dependency == "credential":
            return credential.read_text()
        if dependency == "credential_alias":
            return alias.read_text()
        destination.write_text("must never be written")
        return ""

    with patch("agent.prompt_builder.build_nous_subscription_prompt", side_effect=unavailable):
        with pytest.raises(PromptInspectionUnavailable, match="^native_instruction_inspection_unavailable$"):
            inspect_cron_prompt(**inputs())
    assert not destination.exists()
    assert prompt_source_inspection_active() is False


def test_cold_native_inspection_refuses_unavailable_dependencies_without_profile_writes(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    home = tmp_path / "home"
    hermes = home / "hermes"
    hermes.mkdir(parents=True)
    (hermes / "SOUL.md").write_text("# Native identity\nKeep this identity.\n")
    (hermes / "config.yaml").write_text("agent:\n  environment_probe: false\n")
    code = r'''
import json,os,hashlib
from pathlib import Path
from agent.prompt_inspection import inspect_cron_prompt
from agent.prompt_inspection_scope import PromptInspectionUnavailable
root=Path(os.environ["HERMES_HOME"])
snapshot=lambda:{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
before=snapshot()
try:
    result=inspect_cron_prompt(config={"agent":{"environment_probe":False}},model="gpt-6.1-sol",
        provider="openai-codex",context_length=272000,tools=[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}],
        session_id="cron_aaaaaaaaaaaa_20261028_210000",workdir_present=False,prefill_messages=[])
    status="constructed"
except PromptInspectionUnavailable as error:
    assert str(error)=="native_instruction_inspection_unavailable"
    status="unavailable"
assert before==snapshot()
print(json.dumps({"status":status,"persistent_file_changes":0}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[2],
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
                                 "HERMES_HOME": str(hermes), "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status"] in {"constructed", "unavailable"}
    assert report["persistent_file_changes"] == 0
