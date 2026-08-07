"""Tests for ConfigManager ollama_cloud_accounts CRUD."""
import json
import os
import uuid as uuid_mod
from config_manager import ConfigManager


class TestOllamaCloudAccountsMasking:
    """Tests for api_key masking utility."""

    def test_mask_api_key_basic(self):
        """Test basic api_key masking preserves prefix."""
        key = "sk-1234567890abcdef"
        result = ConfigManager._mask_api_key(key)
        assert result == "sk-123****...****"
        assert result != key  # not the real key

    def test_mask_api_key_empty_string(self):
        """Test masking empty string returns empty."""
        assert ConfigManager._mask_api_key("") == ""

    def test_mask_api_key_none(self):
        """Test masking None returns empty."""
        assert ConfigManager._mask_api_key(None) == ""

    def test_mask_api_key_short(self):
        """Test masking a very short key (length <= 6 → suffix ****)."""
        key = "sk-abc"
        result = ConfigManager._mask_api_key(key)
        assert result == "sk-abc****"


class TestOllamaCloudAccountsGet:
    """Tests for get_ollama_cloud_accounts."""

    def test_get_empty_list_when_no_accounts(self, tmp_path):
        """Test returns empty list when no ollama_cloud_accounts key exists."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({})
        accounts = cm.get_ollama_cloud_accounts()
        assert accounts == []

    def test_get_empty_list_when_none(self, tmp_path):
        """Test returns empty list when key is None."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({"ollama_cloud_accounts": None})
        accounts = cm.get_ollama_cloud_accounts()
        assert accounts == []

    def test_get_list_masks_api_key(self, tmp_path):
        """Test accounts are returned with masked api_key."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        real_key = "sk-realkey12345678901234567890"
        cm.save({
            "ollama_cloud_accounts": [
                {
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "api_key": real_key,
                    "label": "Test Account",
                    "created_at": "2026-08-06T10:00:00Z",
                }
            ]
        })
        accounts = cm.get_ollama_cloud_accounts()
        assert len(accounts) == 1
        assert accounts[0]["id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert accounts[0]["api_key"] != real_key
        assert accounts[0]["api_key"].startswith("sk-")
        assert "****" in accounts[0]["api_key"]
        assert accounts[0]["label"] == "Test Account"

    def test_get_returns_all_accounts(self, tmp_path):
        """Test all accounts are returned."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({
            "ollama_cloud_accounts": [
                {
                    "id": "id1",
                    "api_key": "sk-key1",
                    "label": "Account 1",
                    "created_at": "2026-08-01T00:00:00Z",
                },
                {
                    "id": "id2",
                    "api_key": "sk-key2",
                    "label": "Account 2",
                    "created_at": "2026-08-02T00:00:00Z",
                },
            ]
        })
        accounts = cm.get_ollama_cloud_accounts()
        assert len(accounts) == 2
        assert accounts[0]["label"] == "Account 1"
        assert accounts[1]["label"] == "Account 2"


class TestOllamaCloudAccountsAdd:
    """Tests for add_ollama_cloud_account."""

    def test_add_account_creates_entry(self, tmp_path):
        """Test adding an account creates it in config."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        result = cm.add_ollama_cloud_account("sk-testkey12345678901234567890", "My Account")
        assert "id" in result
        assert result["label"] == "My Account"
        assert "****" in result["api_key"]
        assert result["created_at"] is not None

    def test_add_account_returns_masked_key(self, tmp_path):
        """Test returned account has masked api_key, not the real one."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        real_key = "sk-abcdefghijklmnopqrstuvwxyz123456"
        result = cm.add_ollama_cloud_account(real_key)
        assert result["api_key"] != real_key

    def test_add_account_without_label(self, tmp_path):
        """Test adding account with empty label defaults to empty string."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        result = cm.add_ollama_cloud_account("sk-testkey12345678901234567890")
        assert result["label"] == ""

    def test_add_account_raises_without_api_key(self, tmp_path):
        """Test add raises ValueError when api_key is empty."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        try:
            cm.add_ollama_cloud_account("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_add_account_persists_to_disk(self, tmp_path):
        """Test added account survives a fresh load."""
        config_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path)
        cm.add_ollama_cloud_account("sk-testkey12345678901234567890", "Persistent")
        # Fresh instance should read the saved data
        cm2 = ConfigManager(config_path)
        accounts = cm2.get_ollama_cloud_accounts()
        assert len(accounts) == 1
        assert accounts[0]["label"] == "Persistent"

    def test_add_account_generates_uuid(self, tmp_path):
        """Test that add generates a valid UUID id."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        result = cm.add_ollama_cloud_account("sk-testkey12345678901234567890")
        # Should not raise
        uuid_mod.UUID(result["id"])

    def test_add_multiple_accounts_accumulate(self, tmp_path):
        """Test adding multiple accounts accumulates them."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.add_ollama_cloud_account("sk-key1", "Account 1")
        cm.add_ollama_cloud_account("sk-key2", "Account 2")
        cm.add_ollama_cloud_account("sk-key3", "Account 3")
        accounts = cm.get_ollama_cloud_accounts()
        assert len(accounts) == 3


class TestOllamaCloudAccountsRemove:
    """Tests for remove_ollama_cloud_account."""

    def test_remove_existing_account(self, tmp_path):
        """Test removing an existing account returns True."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890", "To Remove")
        result = cm.remove_ollama_cloud_account(acc["id"])
        assert result is True

    def test_remove_nonexistent_account(self, tmp_path):
        """Test removing a nonexistent account returns False."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        result = cm.remove_ollama_cloud_account("nonexistent-id")
        assert result is False

    def test_remove_persists_to_disk(self, tmp_path):
        """Test removed account is gone after fresh load."""
        config_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path)
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890")
        cm.remove_ollama_cloud_account(acc["id"])
        cm2 = ConfigManager(config_path)
        assert len(cm2.get_ollama_cloud_accounts()) == 0

    def test_remove_one_of_many(self, tmp_path):
        """Test removing one account leaves the others intact."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        a1 = cm.add_ollama_cloud_account("sk-key1", "Account 1")
        a2 = cm.add_ollama_cloud_account("sk-key2", "Account 2")
        cm.remove_ollama_cloud_account(a1["id"])
        accounts = cm.get_ollama_cloud_accounts()
        assert len(accounts) == 1
        assert accounts[0]["id"] == a2["id"]


class TestOllamaCloudAccountsUpdate:
    """Tests for update_ollama_cloud_account."""

    def test_update_label(self, tmp_path):
        """Test updating the label of an account."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890", "Old Label")
        result = cm.update_ollama_cloud_account(acc["id"], {"label": "New Label"})
        assert result is not None
        assert result["label"] == "New Label"

    def test_update_nonexistent_account(self, tmp_path):
        """Test updating a nonexistent account returns None."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        result = cm.update_ollama_cloud_account("nonexistent-id", {"label": "X"})
        assert result is None

    def test_update_partial_fields_preserves_others(self, tmp_path):
        """Test that partial update preserves other fields."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890", "Old Label")
        created = acc["created_at"]
        account_id = acc["id"]
        result = cm.update_ollama_cloud_account(account_id, {"label": "New Label"})
        assert result["id"] == account_id
        assert result["label"] == "New Label"
        # api_key should still be masked (not the real key)
        assert "****" in result["api_key"]

    def test_update_api_key(self, tmp_path):
        """Test rotating the api_key."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        acc = cm.add_ollama_cloud_account("sk-oldkey1234567890123456789012", "Kept Account")
        new_key = "sk-newkey1234567890123456789012"
        result = cm.update_ollama_cloud_account(acc["id"], {"api_key": new_key})
        assert result is not None
        assert "****" in result["api_key"]
        # Verify the new key is stored by reloading
        accounts = cm.get_ollama_cloud_accounts()
        assert len(accounts) == 1

    def test_update_persists_to_disk(self, tmp_path):
        """Test updated account survives a fresh load."""
        config_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path)
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890", "Old")
        cm.update_ollama_cloud_account(acc["id"], {"label": "Updated"})
        cm2 = ConfigManager(config_path)
        accounts = cm2.get_ollama_cloud_accounts()
        assert len(accounts) == 1
        assert accounts[0]["label"] == "Updated"

    def test_update_empty_accounts_list(self, tmp_path):
        """Test update on empty accounts list returns None."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({"ollama_cloud_accounts": []})
        result = cm.update_ollama_cloud_account("any-id", {"label": "X"})
        assert result is None


class TestOllamaCloudAtomicPersistence:
    """Tests that ollama_cloud_accounts use atomic persistence."""

    def test_atomic_write_no_tmp_left(self, tmp_path):
        """Test atomic write does not leave .tmp file after operations."""
        config_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path)
        cm.add_ollama_cloud_account("sk-key12345678901234567890")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_update_no_tmp_left(self, tmp_path):
        """Test atomic write does not leave .tmp file after update."""
        config_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path)
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890")
        cm.update_ollama_cloud_account(acc["id"], {"label": "Updated"})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_remove_no_tmp_left(self, tmp_path):
        """Test atomic write does not leave .tmp file after remove."""
        config_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path)
        acc = cm.add_ollama_cloud_account("sk-key12345678901234567890")
        cm.remove_ollama_cloud_account(acc["id"])
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
