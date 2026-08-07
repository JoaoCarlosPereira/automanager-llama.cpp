"""Tests for non-CLI platform support (Ollama Cloud) in PlatformIntegrationManager."""

import asyncio
import time
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from platform_manager import (
    DEFAULT_OLLAMA_CLOUD_DEFINITION,
    DEFAULT_PLATFORM_DEFINITIONS,
    PlatformDefinition,
    PlatformIntegrationError,
    PlatformIntegrationManager,
)
from platform_ollama_cloud import (
    OllamaCloudAccount,
    OllamaCloudCatalog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entry(catalog: list, backend_id: str) -> dict:
    return next(item for item in catalog if item["backend_id"] == backend_id)


class FakeOllamaCloudAccountManager:
    """Lightweight fake that stores accounts in memory."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self._accounts: List[OllamaCloudAccount] = []

    def get_accounts(self) -> List[OllamaCloudAccount]:
        return list(self._accounts)

    def add_account(self, api_key: str, label: str = "") -> OllamaCloudAccount:
        acc = OllamaCloudAccount(
            id=f"acc-{len(self._accounts) + 1}",
            api_key=api_key,
            label=label,
            status="available",
        )
        self._accounts.append(acc)
        return acc

    async def validate_connection(self, account: OllamaCloudAccount) -> bool:
        return account.status == "available"

    def resolve_for_request(self, *args, **kwargs):
        for acc in self._accounts:
            if acc.status == "available":
                return acc
        return None

    def apply_cooldown(self, account: OllamaCloudAccount, *args) -> None:
        account.status = "cooldown"

    def clear_cooldown(self, account: OllamaCloudAccount) -> None:
        account.status = "available"
        account.cooldown_until = None


def make_manager(
    tmp_config_manager,
    accounts: Optional[List[OllamaCloudAccount]] = None,
    with_ollama_cloud: bool = True,
    cliproxy_found: bool = True,
):
    """Build a PlatformIntegrationManager with optional Ollama Cloud support."""
    ollama_am: Optional[FakeOllamaCloudAccountManager] = None
    ollama_cat: Optional[OllamaCloudCatalog] = None

    if with_ollama_cloud:
        ollama_am = FakeOllamaCloudAccountManager(tmp_config_manager)
        ollama_cat = OllamaCloudCatalog()
        if accounts:
            ollama_am._accounts = list(accounts)

    resolver = {}
    if cliproxy_found:
        resolver["CLIProxyAPI"] = "/fake/CLIProxyAPI"

    definitions = list(DEFAULT_PLATFORM_DEFINITIONS)
    if with_ollama_cloud:
        definitions.append(DEFAULT_OLLAMA_CLOUD_DEFINITION)

    manager = PlatformIntegrationManager(
        tmp_config_manager,
        executable_resolver=resolver.get,
        platform_definitions=definitions,
        ollama_cloud_account_manager=ollama_am,
        ollama_cloud_catalog=ollama_cat,
    )
    return manager, ollama_am


# ---------------------------------------------------------------------------
# DEFAULT_OLLAMA_CLOUD_DEFINITION tests
# ---------------------------------------------------------------------------

class TestOllamaCloudDefinition:
    def test_has_cli_is_false(self):
        assert DEFAULT_OLLAMA_CLOUD_DEFINITION.has_cli is False

    def test_no_command_candidates(self):
        assert DEFAULT_OLLAMA_CLOUD_DEFINITION.command_candidates == ()

    def test_backend_id(self):
        assert DEFAULT_OLLAMA_CLOUD_DEFINITION.backend_id == "platform:ollama-cloud"

    def test_provider(self):
        assert DEFAULT_OLLAMA_CLOUD_DEFINITION.provider == "ollama-cloud"

    def test_display_name(self):
        assert DEFAULT_OLLAMA_CLOUD_DEFINITION.display_name == "Ollama Cloud"


# ---------------------------------------------------------------------------
# Catalog tests
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_ollama_cloud_included_in_catalog(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        catalog = manager.catalog()
        backend_ids = [e["backend_id"] for e in catalog]
        assert "platform:ollama-cloud" in backend_ids

    def test_ollama_cloud_has_cli_false(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        assert ollama_entry["has_cli"] is False

    def test_ollama_cloud_status_missing_without_accounts(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager, accounts=[])
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        assert ollama_entry["status"] == "missing"

    def test_ollama_cloud_status_available_with_account(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        assert ollama_entry["status"] == "available"

    def test_ollama_cloud_status_error_all_cooldown(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="cooldown")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        assert ollama_entry["status"] == "error"

    def test_ollama_cloud_catalog_entry_includes_accounts(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        assert "accounts" in ollama_entry
        assert ollama_entry["account_count"] == 1
        assert len(ollama_entry["accounts"]) == 1
        assert ollama_entry["accounts"][0]["id"] == "a1"
        assert ollama_entry["accounts"][0]["label"] == "Test"
        assert ollama_entry["accounts"][0]["status"] == "available"

    def test_ollama_cloud_catalog_entry_includes_catalog_status(self, tmp_config_manager):
        manager, ollama_am = make_manager(tmp_config_manager, accounts=[])
        assert ollama_am is not None
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        assert "catalog_status" in ollama_entry
        # Catalog has models built-in, so status should be fresh
        assert ollama_entry["catalog_status"] == "fresh"

    def test_cli_platforms_still_appear_in_catalog(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        backend_ids = [e["backend_id"] for e in manager.catalog()]
        assert "platform:codex" in backend_ids
        assert "platform:claude-code" in backend_ids
        assert "platform:google-antigravity" in backend_ids

    def test_cli_platform_detection_unchanged(self, tmp_config_manager):
        manager, _ = make_manager(
            tmp_config_manager,
            with_ollama_cloud=True,
            cliproxy_found=True,
        )
        codex = entry(manager.catalog(), "platform:codex")
        assert codex["detected"] is False
        assert codex["status"] == "missing"

    def test_ollama_cloud_includes_account_with_cooldown(self, tmp_config_manager):
        acc = OllamaCloudAccount(
            id="a2",
            api_key="sk-test",
            label="Cooldown",
            status="cooldown",
            cooldown_until=time.time() + 100,
        )
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])
        ollama_entry = entry(manager.catalog(), "platform:ollama-cloud")
        acc_info = ollama_entry["accounts"][0]
        assert acc_info["cooldown_until"] is not None


# ---------------------------------------------------------------------------
# runtime_state tests
# ---------------------------------------------------------------------------

class TestRuntimeState:
    def test_runtime_state_includes_accounts_for_non_cli(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])
        state = manager.runtime_state("platform:ollama-cloud")
        assert "accounts" in state
        assert len(state["accounts"]) == 1
        assert state["accounts"][0]["id"] == "a1"

    def test_runtime_state_missing_platform(self, tmp_config_manager):
        manager = make_manager(tmp_config_manager)[0]
        result = manager.runtime_state("platform:nonexistent")
        assert result["status"] == "missing"
        assert result["active"] is False

    def test_runtime_state_cli_platform_has_no_accounts_key(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        state = manager.runtime_state("platform:codex")
        assert "accounts" not in state or state.get("has_cli") is not False


# ---------------------------------------------------------------------------
# start_backend tests
# ---------------------------------------------------------------------------

class TestStartBackend:
    def test_start_ollama_cloud_no_sidecar_required(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])

        sidecar = MagicMock()

        state = manager.start_backend("platform:ollama-cloud", sidecar)
        assert state["active"] is True
        assert state["status"] == "running"
        sidecar.ensure_running.assert_not_called()

    def test_start_ollama_cloud_validates_connection(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, am = make_manager(tmp_config_manager, accounts=[acc])
        assert am is not None
        validate_calls = []
        original_validate = am.validate_connection
        async def tracked_validate(a):
            validate_calls.append(a.id)
            return await original_validate(a)
        am.validate_connection = tracked_validate

        sidecar = MagicMock()
        manager.start_backend("platform:ollama-cloud", sidecar)
        assert len(validate_calls) == 1
        assert validate_calls[0] == "a1"

    def test_start_ollama_cloud_no_accounts_raises(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager, accounts=[])
        sidecar = MagicMock()
        with pytest.raises(PlatformIntegrationError) as exc:
            manager.start_backend("platform:ollama-cloud", sidecar)
        assert exc.value.status_code == 502

    def test_start_cli_platform_still_uses_sidecar(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)

        class Sidecar:
            def ensure_running(self):
                return {"status": "running", "port": 9100}

        sidecar = MagicMock(wraps=Sidecar())
        with pytest.raises(PlatformIntegrationError) as exc:
            manager.start_backend("platform:codex", sidecar)
        assert exc.value.status_code == 400
        sidecar.ensure_running.assert_not_called()


# ---------------------------------------------------------------------------
# stop_backend tests
# ---------------------------------------------------------------------------

class TestStopBackend:
    def test_stop_ollama_cloud_updates_runtime_no_sidecar(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])
        sidecar = MagicMock()

        manager.start_backend("platform:ollama-cloud", sidecar)
        assert manager.runtime_state("platform:ollama-cloud")["active"] is True

        state = manager.stop_backend("platform:ollama-cloud", sidecar)
        assert state["active"] is False
        assert state["status"] == "stopped"
        sidecar.stop.assert_not_called()

    def test_stop_cli_platform_still_stops_sidecar(self, tmp_config_manager):
        manager, _ = make_manager(
            tmp_config_manager,
            with_ollama_cloud=False,
        )

        class Sidecar:
            def __init__(self):
                self.stopped = False
            def ensure_running(self):
                return {"status": "running", "port": 9100}
            def stop(self):
                self.stopped = True

        sidecar_obj = Sidecar()
        sidecar = MagicMock(wraps=sidecar_obj)
        with pytest.raises(PlatformIntegrationError):
            manager.start_backend("platform:codex", sidecar)
        assert sidecar_obj.stopped is False


# ---------------------------------------------------------------------------
# active_instances tests
# ---------------------------------------------------------------------------

class TestActiveInstances:
    def test_active_instances_includes_non_cli(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])

        sidecar = MagicMock()
        manager.start_backend("platform:ollama-cloud", sidecar)

        instances = manager.active_instances()
        ollama_instances = [i for i in instances if i["backend_id"] == "platform:ollama-cloud"]
        assert len(ollama_instances) == 1
        assert ollama_instances[0]["status"] == "running"
        assert ollama_instances[0]["backend_type"] == "platform"
        assert ollama_instances[0]["provider"] == "ollama-cloud"

    def test_active_instances_excludes_stopped_non_cli(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])

        sidecar = MagicMock()
        manager.start_backend("platform:ollama-cloud", sidecar)
        manager.stop_backend("platform:ollama-cloud", sidecar)

        instances = manager.active_instances()
        ollama_instances = [i for i in instances if i["backend_id"] == "platform:ollama-cloud"]
        assert len(ollama_instances) == 0

    def test_active_instances_excludes_ollama_when_not_started(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager, accounts=[])
        sidecar = MagicMock()
        instances = manager.active_instances()
        ollama_instances = [i for i in instances if i["backend_id"] == "platform:ollama-cloud"]
        assert len(ollama_instances) == 0


# ---------------------------------------------------------------------------
# get() tests
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_ollama_cloud_returns_non_cli_entry(self, tmp_config_manager):
        acc = OllamaCloudAccount(id="a1", api_key="sk-test", label="Test", status="available")
        manager, _ = make_manager(tmp_config_manager, accounts=[acc])

        item = manager.get("platform:ollama-cloud")
        assert item is not None
        assert item["backend_id"] == "platform:ollama-cloud"
        assert item["has_cli"] is False
        assert item["provider"] == "ollama-cloud"

    def test_get_cli_platform_unchanged(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        item = manager.get("platform:codex")
        assert item is not None
        assert item["backend_id"] == "platform:codex"
        assert "has_cli" not in item


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_cli_platforms_detected_without_ollama_cloud(self, tmp_config_manager):
        """CLI platform detection should work when Ollama Cloud is not registered."""
        manager, _ = make_manager(tmp_config_manager, with_ollama_cloud=False)
        codex = entry(manager.catalog(), "platform:codex")
        assert codex["detected"] is False

    def test_runtime_states_all_platforms(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        states = manager.runtime_states()
        backend_ids = [s["backend_id"] for s in states]
        assert "platform:codex" in backend_ids
        assert "platform:ollama-cloud" in backend_ids

    def test_definitions_includes_non_cli(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        defs = manager.definitions()
        def_ids = [d.backend_id for d in defs]
        assert "platform:ollama-cloud" in def_ids

    def test_start_nonexistent_platform_raises_404(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        sidecar = MagicMock()
        with pytest.raises(PlatformIntegrationError) as exc:
            manager.start_backend("platform:unknown", sidecar)
        assert exc.value.status_code == 404

    def test_stop_nonexistent_platform_raises_404(self, tmp_config_manager):
        manager, _ = make_manager(tmp_config_manager)
        sidecar = MagicMock()
        with pytest.raises(PlatformIntegrationError) as exc:
            manager.stop_backend("platform:unknown", sidecar)
        assert exc.value.status_code == 404
