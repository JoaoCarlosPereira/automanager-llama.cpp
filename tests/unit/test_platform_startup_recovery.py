from types import SimpleNamespace

import pytest

import llama_manager
from platform_manager import (
    clear_platform_listing_registry,
    platform_provider_for_listing,
)


class _RunningEvent:
    def __init__(self):
        self.waits = []

    def is_set(self):
        return False

    def wait(self, delay):
        self.waits.append(delay)
        return False


def test_ollama_cloud_auto_start_retries_transient_failures(monkeypatch):
    attempts = []

    class _PlatformManager:
        def start_backend(self, backend_id, _sidecar):
            attempts.append(backend_id)
            if len(attempts) < 3:
                raise RuntimeError("network not ready")
            return {"status": "running", "sidecar_port": None}

    event = _RunningEvent()
    monkeypatch.setattr(llama_manager, "platform_manager", _PlatformManager())
    monkeypatch.setattr(llama_manager, "shutdown_event", event)
    monkeypatch.setattr(
        llama_manager, "_OLLAMA_CLOUD_AUTO_START_RETRY_DELAYS", (1.0, 2.0)
    )

    assert llama_manager._auto_start_platform_with_retry(
        "platform:ollama-cloud"
    ) is True
    assert attempts == ["platform:ollama-cloud"] * 3
    assert event.waits == [1.0, 2.0]


def test_cli_platform_auto_start_does_not_retry(monkeypatch):
    attempts = []

    class _PlatformManager:
        def start_backend(self, backend_id, _sidecar):
            attempts.append(backend_id)
            raise RuntimeError("authentication required")

    event = _RunningEvent()
    monkeypatch.setattr(llama_manager, "platform_manager", _PlatformManager())
    monkeypatch.setattr(llama_manager, "shutdown_event", event)

    assert llama_manager._auto_start_platform_with_retry("platform:codex") is False
    assert attempts == ["platform:codex"]
    assert event.waits == []


@pytest.mark.asyncio
async def test_cloud_catalog_discovers_with_no_active_cloud_instance(monkeypatch):
    account = SimpleNamespace(id="account-1", api_key="secret")

    class _AccountManager:
        def get_accounts(self):
            return [account]

    class _Provider:
        def __init__(self, selected):
            assert selected is account

        async def list_models(self):
            return [{"id": "nemotron-3-ultra", "object": "model"}]

        async def close(self):
            return None

    clear_platform_listing_registry()
    monkeypatch.setattr(llama_manager, "ollama_cloud_manager", _AccountManager())
    monkeypatch.setattr(llama_manager, "OllamaCloudProvider", _Provider)
    monkeypatch.setattr(llama_manager, "_ollama_cloud_model_catalog", {})
    monkeypatch.setattr(llama_manager, "_ollama_cloud_model_catalog_loaded_at", 0.0)

    await llama_manager._ensure_ollama_cloud_model_registry(
        instances=[{"backend_type": "local", "status": "running"}],
        force=True,
    )

    assert "nemotron-3-ultra" in llama_manager._ollama_cloud_model_catalog
    assert platform_provider_for_listing("nemotron-3-ultra") == "ollama-cloud"
    clear_platform_listing_registry()


@pytest.mark.asyncio
async def test_empty_cloud_discovery_is_not_cached(monkeypatch):
    account = SimpleNamespace(id="account-1", api_key="secret")

    class _AccountManager:
        def get_accounts(self):
            return [account]

    class _Provider:
        def __init__(self, _selected):
            pass

        async def list_models(self):
            return []

        async def close(self):
            return None

    monkeypatch.setattr(llama_manager, "ollama_cloud_manager", _AccountManager())
    monkeypatch.setattr(llama_manager, "OllamaCloudProvider", _Provider)
    monkeypatch.setattr(llama_manager, "_ollama_cloud_model_catalog", {})
    monkeypatch.setattr(llama_manager, "_ollama_cloud_model_catalog_loaded_at", 0.0)

    await llama_manager._ensure_ollama_cloud_model_registry(force=True)

    assert llama_manager._ollama_cloud_model_catalog == {}
    assert llama_manager._ollama_cloud_model_catalog_loaded_at == 0.0
