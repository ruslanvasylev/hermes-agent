"""Run-scoped controls must survive native retry/recovery without global edits."""
from types import SimpleNamespace
from unittest.mock import patch
import threading
import time

import httpx
import pytest

from agent.api_run_budget import ApiRunBudget, ApiRunBudgetExceeded, run_with_budget
from run_agent import AIAgent


def make_agent(**kwargs):
    with patch('run_agent.OpenAI'), patch('hermes_cli.config.load_config', return_value={'agent': {'api_max_retries': 30}}):
        return AIAgent(api_key='test-key', base_url='https://example.invalid/v1',
                       provider='openrouter', model='test/model', quiet_mode=True,
                       skip_context_files=True, skip_memory=True, enabled_toolsets=[], **kwargs)


def test_scoped_attempts_do_not_lower_global_config():
    limited = make_agent(api_max_attempts=3)
    ordinary = make_agent()
    assert limited._api_max_retries == ordinary._api_max_retries == 30
    assert limited._api_run_budget_limits == (3, None)
    assert ordinary._api_run_budget_limits == (None, None)


def test_native_retry_loop_stops_at_three_dispatches_even_when_recovery_would_reset():
    agent = make_agent(api_max_attempts=3)
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadError('synthetic transport failure')
    with patch.object(agent, '_interruptible_streaming_api_call', side_effect=fail), \
         patch.object(agent, '_interruptible_api_call', side_effect=fail), \
         patch.object(agent, '_try_recover_primary_transport', return_value=True), \
         patch('agent.conversation_loop.jittered_backoff', return_value=0):
        result = agent.run_conversation('Synthetic test request')
    assert len(calls) == 3
    assert result['failed'] is True and result['completed'] is False
    assert result['error_type'] == 'ApiRunBudgetExceeded'
    assert result['error_code'] == 'api_max_attempts'
    assert result['api_run_budget']['application_attempts'] == 3


def test_retry_backoff_is_interrupted_by_elapsed_budget():
    agent = make_agent(api_max_attempts=3, max_run_seconds=0.1)
    with patch.object(agent, '_interruptible_streaming_api_call', side_effect=httpx.ReadError('synthetic')), \
         patch.object(agent, '_interruptible_api_call', side_effect=httpx.ReadError('synthetic')), \
         patch('agent.conversation_loop.jittered_backoff', return_value=30):
        start = time.monotonic()
        result = agent.run_conversation('Synthetic test request')
    assert time.monotonic() - start < 3
    assert result['failed'] is True and result['error_code'] == 'max_run_seconds'
    assert result['api_run_budget']['application_attempts'] <= 1


def test_monotonic_deadline_caps_timeout_and_cannot_be_refunded():
    budget = ApiRunBudget(3, 0.05)
    assert 0 < budget.cap_timeout(900) <= 0.05
    for _ in range(3):
        budget.claim_attempt()
    with pytest.raises(ApiRunBudgetExceeded):
        budget.claim_attempt()
    assert budget.attempts == 3


def test_deadline_marks_partial_success_incomplete_and_emits_metadata_only():
    events = []
    stopped = threading.Event()
    agent = SimpleNamespace(_api_run_budget_limits=(3, 0.03), session_id='synthetic-session',
                            interrupt=lambda _: stopped.set(), event_callback=lambda *args: events.append(args))
    def call():
        assert stopped.wait(1)
        return {'completed': True, 'final_response': 'partial'}
    result = run_with_budget(agent, call)
    assert result['completed'] is False and result['failed'] is True
    assert result['error_code'] == 'max_run_seconds'
    assert events[0][0] == 'api_run_budget_exhausted'
    assert 'final_response' not in events[0][1]


@pytest.mark.parametrize('attempts,seconds', [(0,None),(True,None),(1.2,None),(None,0),(None,float('nan')),(None,True)])
def test_invalid_limits_fail_before_agent_initialization(attempts, seconds):
    with pytest.raises(ValueError):
        make_agent(api_max_attempts=attempts, max_run_seconds=seconds)


@pytest.mark.parametrize("progress", [False, True])
def test_real_native_http_read_is_cancelled_at_deadline(tmp_path, progress):
    """Exercise SDK + Hermes worker/poller against a stalled local SSE read."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    received = threading.Event()
    release = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.flush()
            received.set()
            if progress:
                import json
                chunk = {"id": "synthetic", "object": "chat.completion.chunk", "created": 0,
                         "model": "test/model", "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
                for _ in range(500):
                    if release.wait(0.01):
                        break
                    try:
                        self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
            else:
                release.wait(5)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch('hermes_cli.config.load_config', return_value={'agent': {'api_max_retries': 30}}):
            agent = AIAgent(api_key='synthetic-key',
                            base_url=f'http://127.0.0.1:{server.server_port}/v1',
                            provider='custom', model='test/model', quiet_mode=True,
                            skip_context_files=True, skip_memory=True, enabled_toolsets=[],
                            api_max_attempts=3, max_run_seconds=0.5)
        start = time.monotonic()
        result = agent.run_conversation('Synthetic local transport control')
        elapsed = time.monotonic() - start
        assert received.is_set(), 'control must actually enter a blocked provider read'
        assert elapsed < 3
        assert result['failed'] and not result['completed']
        assert result['error_code'] == 'max_run_seconds'
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_fallback_retry_counter_reset_cannot_escape_application_ceiling():
    agent = make_agent(api_max_attempts=3)
    agent._fallback_chain = [{'provider': 'custom', 'model': 'test/fallback'}]
    agent._fallback_index = 0
    with patch.object(agent, '_interruptible_streaming_api_call', side_effect=httpx.ReadTimeout('synthetic')), \
         patch.object(agent, '_interruptible_api_call', side_effect=httpx.ReadTimeout('synthetic')), \
         patch.object(agent, '_try_activate_fallback', return_value=True) as fallback, \
         patch('agent.conversation_loop.jittered_backoff', return_value=0):
        result = agent.run_conversation('Synthetic fallback control')
    assert fallback.called, 'control must traverse a native retry_count reset'
    assert result['api_run_budget']['application_attempts'] == 3
    assert result['error_code'] == 'api_max_attempts'


def test_codex_inner_retry_refreshes_remaining_request_timeout():
    from agent.codex_runtime import run_codex_stream
    from unittest.mock import MagicMock
    budget = ApiRunBudget(3, 1)
    agent = SimpleNamespace(_api_run_budget=budget, _interrupt_requested=False,
                            _client_log_context=lambda: 'synthetic')
    client = MagicMock()
    seen = []
    def create(**kwargs):
        seen.append(kwargs['timeout'])
        if len(seen) == 1:
            time.sleep(0.01)
            raise httpx.ReadTimeout('synthetic')
        return MagicMock()
    client.responses.create.side_effect = create
    with patch('agent.codex_runtime._consume_codex_event_stream', return_value=SimpleNamespace(status='completed')):
        run_codex_stream(agent, {'model': 'test/model', 'timeout': 900}, client=client)
    assert len(seen) == 2
    assert 0 < seen[1] < seen[0] <= 1
    # Native stream retries are wire operations inside one application dispatch.
    assert budget.attempts == 0


def test_codex_app_server_rejects_dispatch_ceiling_before_fastpath():
    agent = make_agent(api_max_attempts=3)
    agent.api_mode = 'codex_app_server'
    with patch.object(agent, '_run_codex_app_server_turn') as delegate:
        with pytest.raises(ValueError, match='api_max_attempts is not supported by codex_app_server'):
            agent.run_conversation('Synthetic unsupported transport control')
    delegate.assert_not_called()


def test_codex_app_server_without_dispatch_ceiling_retains_fastpath():
    agent = make_agent()
    agent.api_mode = 'codex_app_server'
    expected = {'completed': True, 'final_response': 'synthetic'}
    with patch.object(agent, '_run_codex_app_server_turn', return_value=expected) as delegate:
        result = agent.run_conversation('Synthetic existing transport control')
    delegate.assert_called_once()
    assert result == expected
