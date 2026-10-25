"""Native job store/scheduler propagation and script gate controls."""
from unittest.mock import MagicMock, patch
import threading
import time

import pytest


def test_native_job_api_roundtrips_limits_and_old_jobs_omit_them(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from cron.jobs import create_job, get_job, update_job
    old = create_job('synthetic', 'every 5m', model='test/model', provider='custom')
    limited = create_job('synthetic', 'every 5m', model='test/model', provider='custom',
                         api_max_attempts=3, max_run_seconds=180)
    assert 'api_max_attempts' not in old and 'max_run_seconds' not in old
    assert get_job(limited['id'])['api_max_attempts'] == 3
    assert get_job(limited['id'])['max_run_seconds'] == 180
    with pytest.raises(ValueError):
        update_job(limited['id'], {'api_max_attempts': 0})
    assert update_job(limited['id'], {'api_max_attempts': None})['api_max_attempts'] is None


def run_native_job(tmp_path, job, agent, session_db=None):
    from cron.scheduler import run_job
    with patch('cron.scheduler._hermes_home', tmp_path), \
         patch('cron.scheduler._resolve_origin', return_value=None), \
         patch('dotenv.load_dotenv'), patch('hermes_state.SessionDB', return_value=session_db or MagicMock()), \
         patch('hermes_cli.runtime_provider.resolve_runtime_provider', return_value={
             'api_key': 'synthetic', 'base_url': 'https://example.invalid/v1',
             'provider': 'custom', 'api_mode': 'chat_completions'}), \
         patch('run_agent.AIAgent', return_value=agent) as constructor:
        result = run_job(job)
        return result, constructor.call_args


def test_scheduler_passes_explicit_limits_to_native_agent(tmp_path):
    agent = MagicMock()
    agent.run_conversation.return_value = {'completed': True, 'final_response': 'done'}
    result, args = run_native_job(tmp_path, {'id': 'test-budget', 'prompt': 'synthetic',
                                          'model': 'test/model', 'provider': 'custom',
                                          'api_max_attempts': 3, 'max_run_seconds': 180}, agent)
    assert result[0] is True
    assert args.kwargs['api_max_attempts'] == 3
    assert args.kwargs['max_run_seconds'] == 180


def test_scheduler_elapsed_deadline_ignores_recent_activity_and_requests_cancel(tmp_path):
    stop = threading.Event()
    agent = MagicMock()
    agent.interrupt.side_effect = lambda *_: stop.set()
    agent.get_activity_summary.return_value = {'seconds_since_activity': 0}
    def wait(*args, **kwargs):
        stop.wait(3)
        return {'completed': False, 'final_response': ''}
    agent.run_conversation.side_effect = wait
    start = time.monotonic()
    result, _ = run_native_job(tmp_path, {'id': 'test-deadline', 'prompt': 'synthetic',
                                        'model': 'test/model', 'provider': 'custom',
                                        'max_run_seconds': 0.05}, agent)
    assert time.monotonic() - start < 2
    assert result[0] is False
    assert 'max_run_seconds' in result[3]
    assert stop.is_set()


def test_script_gate_does_not_construct_model_agent_despite_limits(tmp_path):
    from cron.scheduler import run_job
    with patch('cron.scheduler._run_job_script', return_value=(True, '{"wakeAgent": false}')), \
         patch('run_agent.AIAgent') as constructor:
        result = run_job({'id': 'script-gate', 'prompt': 'synthetic', 'script': 'synthetic.sh',
                          'api_max_attempts': 3, 'max_run_seconds': 180})
    assert result[0] is True
    constructor.assert_not_called()


def test_deadline_preserves_resources_and_sequential_workdir_slot_until_worker_exit(tmp_path, monkeypatch):
    import os
    import cron.scheduler as scheduler
    release = threading.Event()
    cancelled = threading.Event()
    second_started = threading.Event()
    observed = []
    workdir = tmp_path / 'work'
    workdir.mkdir()
    other_workdir = tmp_path / 'other-work'
    other_workdir.mkdir()
    monkeypatch.setenv('TERMINAL_CWD', 'original-sentinel')
    agent = MagicMock()
    database = MagicMock()
    agent.interrupt.side_effect = lambda *_: cancelled.set()
    def conversation(*args, **kwargs):
        assert cancelled.wait(2)
        observed.append(os.environ.get('TERMINAL_CWD'))
        assert release.wait(3)
        observed.append(os.environ.get('TERMINAL_CWD'))
        return {'completed': False, 'final_response': ''}
    agent.run_conversation.side_effect = conversation
    agent.get_activity_summary.return_value = {'seconds_since_activity': 0}
    job = {'id': 'pending-cancellation', 'prompt': 'synthetic', 'model': 'test/model',
           'provider': 'custom', 'max_run_seconds': 0.03, 'workdir': str(workdir)}
    pool = scheduler._get_sequential_pool()
    try:
        first = pool.submit(run_native_job, tmp_path, job, agent, database)
        assert cancelled.wait(2)
        second_agent = MagicMock()
        other_observed = []
        def second_conversation(*args, **kwargs):
            other_observed.append(os.environ.get('TERMINAL_CWD'))
            return {'completed': True, 'final_response': 'done'}
        second_agent.run_conversation.side_effect = second_conversation
        def second_workdir_job():
            second_started.set()
            return run_native_job(tmp_path, {**job, 'id': 'other-workdir-job',
                                            'max_run_seconds': None,
                                            'workdir': str(other_workdir)}, second_agent)
        second = pool.submit(second_workdir_job)
        assert not first.done()
        assert not second_started.wait(0.05)
        assert os.environ['TERMINAL_CWD'] == str(workdir)
        database.close.assert_not_called()
        database.end_session.assert_not_called()
        agent.close.assert_not_called()
        release.set()
        result, _ = first.result(timeout=2)
        second_result, _ = second.result(timeout=2)
        assert second_result[0] is True
        assert other_observed == [str(other_workdir)]
        assert result[0] is False and 'max_run_seconds' in result[3]
        assert observed == [str(workdir), str(workdir)]
        assert os.environ['TERMINAL_CWD'] == 'original-sentinel'
        database.close.assert_called_once()
        agent.close.assert_called_once()
    finally:
        release.set()
        scheduler._shutdown_parallel_pool()
