"""Trusted cron provider registration has one registry owner and fails closed."""

from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginContext, PluginManager


def test_cron_native_provider_registration_and_rediscovery(monkeypatch):
    manager = PluginManager()
    context = PluginContext(SimpleNamespace(name="fixture", key="fixture"), manager)
    provider = SimpleNamespace(prepare=lambda **kwargs: None)

    with pytest.raises(ValueError):
        context.register_cron_native_preparation_provider(" ", provider)
    with pytest.raises(ValueError):
        context.register_cron_native_preparation_provider("proof", object())

    context.register_cron_native_preparation_provider("proof", provider)
    assert manager.get_cron_native_preparation_provider("proof") is provider
    with pytest.raises(ValueError):
        context.register_cron_native_preparation_provider("proof", provider)

    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)
    manager.discover_and_load(force=True)
    assert manager.get_cron_native_preparation_provider("proof") is None
