"""Extended tests for config_manager.py covering auth, atomic writes, and edge cases."""
import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch
from config_manager import ConfigManager, TokenManager, AuthManager


class TestConfigManagerAtomicWrites:
    """Tests for atomic write functionality via save/load."""

    def test_save_and_load_preserves_data(self, tmp_path):
        """Test save creates config file and load retrieves it."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({"key1": "value1", "key2": "value2"})
        config = cm.get_config()
        assert config["key1"] == "value1"
        assert config["key2"] == "value2"

    def test_save_overwrites_previous_data(self, tmp_path):
        """Test save overwrites existing data, not appends."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({"key": "old_value"})
        cm.save({"key": "new_value", "extra": True})
        config = cm.get_config()
        assert config["key"] == "new_value"
        assert config["extra"] is True
        assert "old_value" not in str(config)

    def test_save_empty_dict(self, tmp_path):
        """Test save with empty dict loads as empty."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({})
        config = cm.get_config()
        assert config == {}

    def test_load_from_nonexistent_file(self, tmp_path):
        """Test load returns empty dict when file does not exist."""
        cm = ConfigManager(str(tmp_path / "nonexistent.json"))
        config = cm.get_config()
        assert config == {}

    def test_json_corruption_handled(self, tmp_path):
        """Test load returns empty dict on corrupted JSON."""
        config_path = str(tmp_path / "config.json")
        with open(config_path, "w") as f:
            f.write("{invalid json!!!")
        cm = ConfigManager(config_path)
        config = cm.get_config()
        assert config == {}

    def test_atomic_write_no_tmp_file_left(self, tmp_path):
        """Test atomic write does not leave .tmp file on success."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.save({"test": True})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestConfigManagerModelSettings:
    """Tests for per-model settings storage."""

    def test_update_and_retrieve_model_settings(self, tmp_path):
        """Test updating model settings and retrieving them."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.update_model_settings("/models/model.gguf", {"context_size": 8192})
        settings = cm.get_model_settings("/models/model.gguf")
        assert settings["context_size"] == 8192

    def test_get_model_settings_nonexistent_returns_empty_dict(self, tmp_path):
        """Test get_model_settings returns {} for nonexistent model (not None)."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        settings = cm.get_model_settings("/nonexistent/model.gguf")
        assert settings == {}

    def test_update_model_settings_merges_with_existing(self, tmp_path):
        """Test update_model_settings merges, not replaces entirely."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.update_model_settings("/models/model.gguf", {"context_size": 8192})
        cm.update_model_settings("/models/model.gguf", {"batch_size": 1024})
        settings = cm.get_model_settings("/models/model.gguf")
        assert settings["context_size"] == 8192
        assert settings["batch_size"] == 1024

    def test_multiple_models_have_independent_settings(self, tmp_path):
        """Test different models store separate settings."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.update_model_settings("/models/a.gguf", {"context_size": 4096})
        cm.update_model_settings("/models/b.gguf", {"context_size": 16384})
        settings_a = cm.get_model_settings("/models/a.gguf")
        settings_b = cm.get_model_settings("/models/b.gguf")
        assert settings_a["context_size"] == 4096
        assert settings_b["context_size"] == 16384


class TestConfigManagerDefaults:
    """Tests for default model management."""

    def test_get_default_models_empty_list(self, tmp_path):
        """Test get_default_models returns empty list with no defaults."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        models = cm.get_default_models()
        assert models == []

    def test_set_default_model_adds_to_list(self, tmp_path):
        """Test set_default_model adds model to defaults."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.set_default_model("/models/model.gguf")
        models = cm.get_default_models()
        assert len(models) == 1
        # normalize_model_path calls os.path.abspath which resolves to CWD on Windows
        assert models[0].endswith("models/model.gguf")

    def test_set_default_model_does_not_duplicate(self, tmp_path):
        """Test set_default_model does not add same model twice."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.set_default_model("/models/model.gguf")
        cm.set_default_model("/models/model.gguf")
        models = cm.get_default_models()
        assert len(models) == 1

    def test_set_default_model_with_add_false_removes(self, tmp_path):
        """Test set_default_model(path, add=False) removes model from defaults."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.set_default_model("/models/model.gguf")
        cm.set_default_model("/models/model.gguf", add=False)
        models = cm.get_default_models()
        assert "/models/model.gguf" not in models

    def test_set_default_model_none_path_leaves_others(self, tmp_path):
        """Test set_default_model(None) does not clear other defaults."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        cm.set_default_model("/models/model1.gguf")
        cm.set_default_model("/models/model2.gguf")
        cm.set_default_model(None)
        models = cm.get_default_models()
        assert len(models) == 2


class TestTokenManager:
    """Tests for TokenManager class."""

    def test_get_or_create_generates_token(self, tmp_path):
        """Test get_or_create generates a new token if none exists."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        token = tm.get_or_create()
        assert token.startswith("sk-")
        assert len(token) > 5

    def test_get_or_create_returns_same_token(self, tmp_path):
        """Test get_or_create returns the same token on subsequent calls."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        token1 = tm.get_or_create()
        token2 = tm.get_or_create()
        assert token1 == token2

    def test_renew_changes_token(self, tmp_path):
        """Test renew generates a new token."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        token1 = tm.get_or_create()
        token2 = tm.renew()
        assert token1 != token2
        assert token2.startswith("sk-")

    def test_validate_good_token(self, tmp_path):
        """Test validate accepts properly formed tokens."""
        tm = TokenManager(None)
        assert tm.validate("sk-abcdefghijklmnopqrstuvwxyz123456") is True

    def test_validate_bad_token(self, tmp_path):
        """Test validate rejects malformed tokens."""
        tm = TokenManager(None)
        assert tm.validate("short") is False
        assert tm.validate("sk-short") is False
        assert tm.validate(123) is False


class TestAuthManager:
    """Tests for AuthManager class."""

    def test_authenticate_default_password(self, tmp_path):
        """Test authenticate with default 'admin' password."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        am = AuthManager(cm, tm)
        result = am.authenticate("admin", "admin")
        assert result is not None
        assert "token" in result

    def test_authenticate_wrong_password(self, tmp_path):
        """Test authenticate rejects wrong password."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        am = AuthManager(cm, tm)
        result = am.authenticate("admin", "wrongpass")
        assert result is None

    def test_change_password_and_login(self, tmp_path):
        """Test changing password allows login with new password."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        am = AuthManager(cm, tm)
        am.change_password("admin", "newpass123")
        result = am.authenticate("admin", "newpass123")
        assert result is not None

    def test_change_password_wrong_old(self, tmp_path):
        """Test change_password rejects wrong old password."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        am = AuthManager(cm, tm)
        result = am.change_password("wrong", "newpass")
        assert result is False

    def test_logout_invalidates_session(self, tmp_path):
        """Test logout invalidates session token."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        am = AuthManager(cm, tm)
        result = am.authenticate("admin", "admin")
        token = result["token"]
        assert am.verify_session(token) is True
        am.logout(token)
        assert am.verify_session(token) is False

    def test_force_password_change_on_first_login(self, tmp_path):
        """Test initial auth returns force_password_change=True."""
        cm = ConfigManager(str(tmp_path / "config.json"))
        tm = TokenManager(cm)
        am = AuthManager(cm, tm)
        result = am.authenticate("admin", "admin")
        assert result["force_password_change"] is True
