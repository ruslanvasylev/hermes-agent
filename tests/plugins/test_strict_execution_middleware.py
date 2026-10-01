"""The opt-in LLM execution chain must prove one successful downstream call."""

import pytest

from hermes_cli import middleware


def _run(monkeypatch, callbacks, *, terminal=None, strict=True, wrapper=None):
    monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda kind: callbacks)
    calls = []

    def send(request):
        calls.append(request)
        if terminal is not None:
            return terminal(request)
        return {"sent": request}

    run = wrapper or middleware.run_llm_execution_middleware
    return lambda: run({"prompt": "original"}, send, require_single_execution=strict), calls


def test_strict_single_call_and_payload_rewrite(monkeypatch):
    def rewrite(request, next_call, **_):
        return next_call({"prompt": "rewritten"})

    run, calls = _run(monkeypatch, [rewrite])
    assert run() == {"sent": {"prompt": "rewritten"}}
    assert calls == [{"prompt": "rewritten"}]


@pytest.mark.parametrize("catch_repeat", [False, True])
def test_strict_duplicate_fails_even_when_caught(monkeypatch, catch_repeat):
    def duplicate(request, next_call, **_):
        first = next_call(request)
        if catch_repeat:
            try:
                next_call(request)
            except RuntimeError:
                pass
        else:
            next_call(request)
        return first

    run, calls = _run(monkeypatch, [duplicate])
    with pytest.raises(RuntimeError, match="more than once"):
        run()
    assert calls == [{"prompt": "original"}]


def test_strict_swallowed_downstream_error_propagates_original(monkeypatch):
    failure = ValueError("terminal failed")

    def swallow(request, next_call, **_):
        try:
            next_call(request)
        except Exception:
            return "false success"

    def fail(_request):
        raise failure

    run, calls = _run(monkeypatch, [swallow], terminal=fail)
    with pytest.raises(ValueError) as caught:
        run()
    assert caught.value is failure
    assert calls == [{"prompt": "original"}]


def test_strict_short_circuit_cannot_report_success(monkeypatch):
    def skip(**_):
        return "false success"

    run, calls = _run(monkeypatch, [skip])
    with pytest.raises(RuntimeError, match="exactly once"):
        run()
    assert calls == []


def test_strict_callback_error_after_success_is_fatal(monkeypatch):
    failure = ValueError("callback failed")

    def fail_after_send(request, next_call, **_):
        next_call(request)
        raise failure

    run, calls = _run(monkeypatch, [fail_after_send])
    with pytest.raises(ValueError) as caught:
        run()
    assert caught.value is failure
    assert calls == [{"prompt": "original"}]


def test_strict_nested_chain_enforces_each_frame(monkeypatch):
    def outer(request, next_call, **_):
        return next_call(request)

    def inner(request, next_call, **_):
        first = next_call(request)
        try:
            next_call(request)
        except RuntimeError:
            pass
        return first

    run, calls = _run(monkeypatch, [outer, inner])
    with pytest.raises(RuntimeError, match="more than once"):
        run()
    assert calls == [{"prompt": "original"}]


def test_strict_compatibility_wrapper_forwards_option(monkeypatch):
    def skip(**_):
        return "false success"

    run, calls = _run(monkeypatch, [skip], wrapper=middleware.run_api_execution_middleware)
    with pytest.raises(RuntimeError, match="exactly once"):
        run()
    assert calls == []


def test_default_duplicate_callback_keeps_first_result(monkeypatch):
    def duplicate(request, next_call, **_):
        next_call(request)
        next_call(request)

    run, calls = _run(monkeypatch, [duplicate], strict=False)
    assert run() == {"sent": {"prompt": "original"}}
    assert calls == [{"prompt": "original"}]
