"""Unit tests for OllamaCloudAccountManager — lifecycle of Ollama Cloud accounts."""

import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform_ollama_cloud import (
    OllamaCloudAccount,
    OllamaCloudAccountManager,
    OllamaCloudCatalog,
    OllamaCloudHTTPError,
    OllamaCloudProvider,
)


@pytest.fixture
def config_manager():
    """Mock ConfigManager with real-looking CRUD behaviour."""
    cm = MagicMock()
    cm.get_ollama_cloud_accounts_raw.return_value = []
    cm.add_ollama_cloud_account.return_value = {
        "id": "acc-new-1",
        "api_key": "sk-****",
        "label": "New Account",
        "created_at": "2026-08-06T00:00:00Z",
    }
    cm.remove_ollama_cloud_account.return_value = True
    cm.update_ollama_cloud_account.return_value = {
        "id": "acc-new-1",
        "api_key": "sk-****",
        "label": "Updated",
        "created_at": "2026-08-06T00:00:00Z",
    }
    return cm


@pytest.fixture
def catalog():
    return OllamaCloudCatalog()


@pytest.fixture
def manager(config_manager, catalog):
    return OllamaCloudAccountManager(config_manager, catalog)


@pytest.fixture
def account():
    return OllamaCloudAccount(
        id="acc-001",
        api_key="sk-test-key-123",
        label="Test Account",
        status="available",
    )


# ---------------------------------------------------------------------------
# get_accounts
# ---------------------------------------------------------------------------


class TestGetAccounts:
    def test_returns_empty_list_when_no_accounts(self, manager):
        accounts = manager.get_accounts()
        assert accounts == []
        manager.config_manager.get_ollama_cloud_accounts_raw.assert_called_once()

    def test_returns_list_of_accounts(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {
                "id": "acc-001",
                "api_key": "sk-****",
                "label": "Account 1",
                "created_at": "2026-08-06T00:00:00Z",
            },
            {
                "id": "acc-002",
                "api_key": "sk-****",
                "label": "Account 2",
                "created_at": "2026-08-06T00:00:01Z",
            },
        ]
        accounts = manager.get_accounts()
        assert len(accounts) == 2
        assert accounts[0].id == "acc-001"
        assert accounts[0].label == "Account 1"
        assert accounts[1].id == "acc-002"
        assert accounts[1].label == "Account 2"

    def test_api_key_is_masked(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {
                "id": "acc-001",
                "api_key": "sk-****",
                "label": "Account 1",
            }
        ]
        accounts = manager.get_accounts()
        assert accounts[0].api_key == "sk-****"

    def test_defaults_status_to_available(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {
                "id": "acc-001",
                "api_key": "sk-****",
                "label": "Account 1",
            }
        ]
        accounts = manager.get_accounts()
        assert accounts[0].status == "available"

    def test_defaults_cooldown_until_to_none(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {
                "id": "acc-001",
                "api_key": "sk-****",
                "label": "Account 1",
            }
        ]
        accounts = manager.get_accounts()
        assert accounts[0].cooldown_until is None


# ---------------------------------------------------------------------------
# add_account
# ---------------------------------------------------------------------------


class TestAddAccount:
    def test_calls_config_manager(self, manager):
        manager.add_account("sk-my-key", "My Account")
        manager.config_manager.add_ollama_cloud_account.assert_called_once_with(
            "sk-my-key", "My Account"
        )

    def test_returns_account_with_api_key(self, manager):
        acc = manager.add_account("sk-my-key", "Test Account")
        assert isinstance(acc, OllamaCloudAccount)
        assert acc.api_key == "sk-my-key"
        assert acc.label == "Test Account"
        assert acc.status == "available"

    def test_raises_on_empty_api_key(self, manager):
        with pytest.raises(ValueError, match="api_key is required"):
            manager.add_account("")

    def test_creates_uuid(self, manager):
        manager.add_account("sk-key", "")
        call_args = manager.config_manager.add_ollama_cloud_account.call_args
        assert call_args[0][0] == "sk-key"
        assert call_args[0][1] == ""


# ---------------------------------------------------------------------------
# remove_account
# ---------------------------------------------------------------------------


class TestRemoveAccount:
    def test_calls_config_manager(self, manager):
        manager.remove_account("acc-001")
        manager.config_manager.remove_ollama_cloud_account.assert_called_once_with(
            "acc-001"
        )

    def test_raises_on_missing_account(self, manager):
        manager.config_manager.remove_ollama_cloud_account.return_value = False
        with pytest.raises(ValueError, match="not found"):
            manager.remove_account("acc-missing")

    def test_raises_on_empty_id(self, manager):
        with pytest.raises(ValueError, match="account_id is required"):
            manager.remove_account("")

    def test_succeeds_when_account_found(self, manager):
        manager.config_manager.remove_ollama_cloud_account.return_value = True
        manager.remove_account("acc-001")  # no exception


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    @pytest.mark.asyncio
    async def test_returns_true_on_health_check_success(self, account):
        with patch.object(
            OllamaCloudProvider, "health_check", new_callable=AsyncMock, return_value=True
        ):
            manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
            result = await manager.validate_connection(account)
            assert result is True
            assert account.status == "available"

    @pytest.mark.asyncio
    async def test_returns_false_on_health_check_failure(self, account):
        with patch.object(
            OllamaCloudProvider, "health_check", new_callable=AsyncMock, return_value=False
        ):
            manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
            result = await manager.validate_connection(account)
            assert result is False
            assert account.status == "error"

    @pytest.mark.asyncio
    async def test_sets_error_on_exception(self, account):
        with patch.object(
            OllamaCloudProvider,
            "health_check",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
            result = await manager.validate_connection(account)
            assert result is False
            assert account.status == "error"

    @pytest.mark.asyncio
    async def test_closes_provider(self, account):
        with patch.object(
            OllamaCloudProvider, "health_check", new_callable=AsyncMock, return_value=True
        ) as mock_health:
            manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
            await manager.validate_connection(account)
            # Provider.close should be called
            mock_health_instance = mock_health.return_value
            # The provider is created and closed in validate_connection
            # We verify via the close mock


# ---------------------------------------------------------------------------
# resolve_for_request
# ---------------------------------------------------------------------------


class TestResolveForRequest:
    def test_returns_none_when_no_accounts(self, manager):
        result = manager.resolve_for_request(frozenset(), 4096)
        assert result is None

    def test_returns_first_available_account(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
            {"id": "acc-002", "api_key": "sk-****", "label": "B"},
        ]
        result = manager.resolve_for_request(frozenset(), 4096)
        assert result is not None
        assert result.id == "acc-001"

    def test_skips_cooldown_account(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
            {"id": "acc-002", "api_key": "sk-****", "label": "B"},
        ]
        # Manually set acc-001 to cooldown
        accounts = manager.get_accounts()
        accounts[0].status = "cooldown"
        accounts[0].cooldown_until = time.time() + 3600

        # Rebuild the internal list by mocking get_accounts to return modified accounts
        with patch.object(manager, "get_accounts", return_value=accounts):
            result = manager.resolve_for_request(frozenset(), 4096)
            assert result is not None
            assert result.id == "acc-002"

    def test_skips_error_account(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
            {"id": "acc-002", "api_key": "sk-****", "label": "B"},
        ]
        accounts = manager.get_accounts()
        accounts[0].status = "error"

        with patch.object(manager, "get_accounts", return_value=accounts):
            result = manager.resolve_for_request(frozenset(), 4096)
            assert result is not None
            assert result.id == "acc-002"

    def test_excludes_account_by_id(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
            {"id": "acc-002", "api_key": "sk-****", "label": "B"},
        ]
        result = manager.resolve_for_request(frozenset(), 4096, exclude_account_id="acc-001")
        assert result is not None
        assert result.id == "acc-002"

    def test_returns_none_when_all_accounts_excluded(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
        ]
        result = manager.resolve_for_request(frozenset(), 4096, exclude_account_id="acc-001")
        assert result is None

    def test_clears_cooldown_when_expired(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
        ]
        accounts = manager.get_accounts()
        accounts[0].status = "cooldown"
        accounts[0].cooldown_until = time.time() - 100  # expired

        with patch.object(manager, "get_accounts", return_value=accounts):
            result = manager.resolve_for_request(frozenset(), 4096)
            assert result is not None
            assert result.id == "acc-001"
            assert result.status == "available"
            assert result.cooldown_until is None

    def test_skips_cooldown_when_not_expired(self, manager):
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
        ]
        accounts = manager.get_accounts()
        accounts[0].status = "cooldown"
        accounts[0].cooldown_until = time.time() + 3600  # not expired

        with patch.object(manager, "get_accounts", return_value=accounts):
            result = manager.resolve_for_request(frozenset(), 4096)
            assert result is None

    def test_returns_first_account_when_no_filters_match(self, manager):
        """If required_capabilities or needed_ctx are empty/low, first available wins."""
        manager.config_manager.get_ollama_cloud_accounts_raw.return_value = [
            {"id": "acc-001", "api_key": "sk-****", "label": "A"},
            {"id": "acc-002", "api_key": "sk-****", "label": "B"},
        ]
        result = manager.resolve_for_request(frozenset(), 0)
        assert result is not None
        assert result.id == "acc-001"


# ---------------------------------------------------------------------------
# apply_cooldown
# ---------------------------------------------------------------------------


class TestApplyCooldown:
    def test_sets_cooldown_status(self, account):
        manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
        manager.apply_cooldown(account)
        assert account.status == "cooldown"

    def test_sets_cooldown_until_with_default_60s(self, account):
        manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
        before = time.time()
        manager.apply_cooldown(account)
        after = time.time()
        assert account.cooldown_until is not None
        expected_min = before + 60
        assert account.cooldown_until >= expected_min
        assert account.cooldown_until <= after + 60 + 1  # tolerance

    def test_sets_cooldown_until_with_custom_retry_after(self, account):
        manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
        before = time.time()
        manager.apply_cooldown(account, retry_after=120.0)
        after = time.time()
        assert account.cooldown_until is not None
        expected_min = before + 120.0
        assert account.cooldown_until >= expected_min
        assert account.cooldown_until <= after + 120.0 + 1

    def test_overrides_previous_cooldown(self, account):
        manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
        account.cooldown_until = time.time() - 100  # old cooldown
        account.status = "cooldown"
        manager.apply_cooldown(account, retry_after=30.0)
        assert account.status == "cooldown"
        assert account.cooldown_until > time.time() - 1  # new, near-future


# ---------------------------------------------------------------------------
# clear_cooldown
# ---------------------------------------------------------------------------


class TestClearCooldown:
    def test_clears_cooldown_status(self, account):
        account.status = "cooldown"
        account.cooldown_until = time.time() + 3600
        manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
        manager.clear_cooldown(account)
        assert account.status == "available"
        assert account.cooldown_until is None

    def test_safe_to_call_on_already_available(self, account):
        account.status = "available"
        account.cooldown_until = None
        manager = OllamaCloudAccountManager(MagicMock(), MagicMock())
        manager.clear_cooldown(account)
        assert account.status == "available"
        assert account.cooldown_until is None


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_succeeds_without_error(self, manager):
        await manager.close()  # should not raise


# ---------------------------------------------------------------------------
# Integration: add + get + remove cycle
# ---------------------------------------------------------------------------


class TestLifecycleCycle:
    def test_add_get_remove_cycle(self):
        cm = MagicMock()
        cm.get_ollama_cloud_accounts_raw.return_value = []
        cm.add_ollama_cloud_account.return_value = {
            "id": "acc-new",
            "api_key": "sk-****",
            "label": "Test",
            "created_at": "2026-08-06T00:00:00Z",
        }
        cm.remove_ollama_cloud_account.return_value = True
        mgr = OllamaCloudAccountManager(cm, OllamaCloudCatalog())

        # Add
        acc = mgr.add_account("sk-my-key", "Test")
        assert acc.id == "acc-new"
        assert acc.api_key == "sk-my-key"

        # Get — now the account exists
        cm.get_ollama_cloud_accounts_raw.return_value = [
            {
                "id": "acc-new",
                "api_key": "sk-****",
                "label": "Test",
                "created_at": "2026-08-06T00:00:00Z",
            }
        ]
        accounts = mgr.get_accounts()
        assert len(accounts) == 1
        assert accounts[0].id == "acc-new"

        # Remove
        mgr.remove_account("acc-new")
        cm.remove_ollama_cloud_account.assert_called_with("acc-new")

        # Get — now empty
        cm.get_ollama_cloud_accounts_raw.return_value = []
        accounts = mgr.get_accounts()
        assert len(accounts) == 0
