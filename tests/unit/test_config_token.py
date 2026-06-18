import json
import os
from unittest.mock import patch, MagicMock

import pytest

from config_manager import ConfigManager, TokenManager, normalize_model_path


@pytest.fixture
def config_path(tmp_path):
    return str(tmp_path / "automanager_config.json")


@pytest.fixture
def config_manager(config_path):
    return ConfigManager(config_path)


@pytest.fixture
def token_manager(config_manager):
    return TokenManager(config_manager)


def test_config_load_missing_file_returns_empty_dict(config_manager):
    assert config_manager.load() == {}


def test_config_load_valid_json(config_manager, config_path):
    expected = {"default_model": "/models/llama.gguf", "enabled": True}
    with open(config_path, "w") as config_file:
        json.dump(expected, config_file)
    loaded = config_manager.load()
    assert loaded["enabled"] is True
    assert loaded["default_models"] == [normalize_model_path("/models/llama.gguf")]
    assert "default_model" not in loaded


def test_config_load_corrupted_json_returns_empty_dict(config_manager, config_path):
    with open(config_path, "w") as config_file:
        config_file.write("{invalid")
    assert config_manager.load() == {}


def test_config_save_and_load_round_trip(config_manager):
    expected = {"api_token": "sk-" + "a" * 32, "default_model": "/models/a.gguf"}
    config_manager.save(expected)
    loaded = config_manager.load()
    assert loaded["api_token"] == expected["api_token"]
    assert loaded["default_models"] == [normalize_model_path("/models/a.gguf")]


def test_config_get_model_settings_empty(config_manager):
    assert config_manager.get_model_settings("/models/missing.gguf") == {}


def test_config_get_model_settings_with_data(config_manager):
    model_path = "/models/llama.gguf"
    config_manager.save({"model_configs": {model_path: {"context_size": 8192}}})
    assert config_manager.get_model_settings(model_path) == {"context_size": 8192}


def test_config_update_model_settings(config_manager):
    model_path = "/models/llama.gguf"
    settings = {
        "context_size": 8192,
        "mmproj_path": "/models/llama.mmproj",
        "gpu_weights": [{"index": 0, "weight": 100, "name": "GPU", "active": True}],
    }
    config_manager.update_model_settings(model_path, settings)
    saved = config_manager.get_model_settings(model_path)
    assert saved["context_size"] == 8192
    assert saved["mmproj_path"] == "/models/llama.mmproj"
    assert saved["gpu_weights"] == settings["gpu_weights"]
    assert saved["mtp_enabled"] is False
    assert saved["mtp_draft_tokens"] == 3
    assert saved["thinking_enabled"] is True
    assert "last_started" in saved


def test_config_update_model_settings_engine_fields(config_manager):
    model_path = "/models/engine.gguf"
    settings = {
        "ubatch_size": 256,
        "cache_type_k": "q8_0",
        "cache_type_v": "q4_0",
        "numa_enabled": True,
        "threads": 8,
        "threads_batch": 4,
        "auto_balance_profile": True,
        "pinned_fields": {"cache_type": True, "threads": False},
        "gpu_weights": [
            {
                "index": 0,
                "weight": 100,
                "name": "GPU",
                "active": True,
                "pinned": True,
            }
        ],
    }
    config_manager.update_model_settings(model_path, settings)
    saved = config_manager.get_model_settings(model_path)
    assert saved["ubatch_size"] == 256
    assert saved["cache_type_k"] == "q8_0"
    assert saved["cache_type_v"] == "q4_0"
    assert saved["numa_enabled"] is True
    assert saved["threads"] == 8
    assert saved["threads_batch"] == 4
    assert saved["auto_balance_profile"] is True
    assert saved["pinned_fields"] == {"cache_type": True, "threads": False}
    assert saved["gpu_weights"][0]["pinned"] is True


def test_config_update_model_settings_mtp_fields(config_manager):
    model_path = "/models/mtp.gguf"
    config_manager.update_model_settings(
        model_path,
        {"mtp_enabled": True, "mtp_draft_tokens": 5},
    )
    saved = config_manager.get_model_settings(model_path)
    assert saved["mtp_enabled"] is True
    assert saved["mtp_draft_tokens"] == 5


def test_config_update_model_settings_thinking_fields(config_manager):
    model_path = "/models/qwen.gguf"
    config_manager.update_model_settings(
        model_path,
        {"thinking_enabled": False},
    )
    saved = config_manager.get_model_settings(model_path)
    assert saved["thinking_enabled"] is False


def test_config_partial_update_preserves_thinking_fields(config_manager):
    model_path = "/models/qwen.gguf"
    config_manager.update_model_settings(
        model_path,
        {"thinking_enabled": False},
    )
    config_manager.update_model_settings(model_path, {"context_size": 32768})
    saved = config_manager.get_model_settings(model_path)
    assert saved["thinking_enabled"] is False
    assert saved["context_size"] == 32768


def test_config_partial_update_preserves_llama_server_bin(config_manager):
    model_path = "/models/qwen.gguf"
    bin_path = "/opt/llama-cpp-turboquant/build/bin/llama-server"
    config_manager.update_model_settings(
        model_path,
        {"llama_server_bin": bin_path, "context_size": 65536},
    )
    config_manager.update_model_settings(model_path, {"context_size": 32768})
    saved = config_manager.get_model_settings(model_path)
    assert saved["llama_server_bin"] == bin_path
    assert saved["context_size"] == 32768


def test_config_update_model_settings_llama_bin_fields(config_manager):
    model_path = "/models/turbo.gguf"
    config_manager.update_model_settings(
        model_path,
        {
            "llama_server_bin": "/opt/turbo/bin/llama-server",
            "turboquant_preset": "recommended",
            "cache_type_k": "q8_0",
            "cache_type_v": "turbo3",
        },
    )
    saved = config_manager.get_model_settings(model_path)
    assert saved["llama_server_bin"] == "/opt/turbo/bin/llama-server"
    assert saved["turboquant_preset"] == "recommended"
    assert saved["cache_type_k"] == "q8_0"
    assert saved["cache_type_v"] == "turbo3"


def test_config_normalizes_model_path_keys(config_manager):
    model_path = "/models/qwen.gguf"
    config_manager.update_model_settings(
        "/models\\qwen.gguf",
        {"thinking_enabled": False},
    )
    saved = config_manager.get_model_settings(model_path)
    assert saved["thinking_enabled"] is False
    loaded = config_manager.load()["model_configs"]
    assert len(loaded) == 1
    assert normalize_model_path(next(iter(loaded.keys()))) == normalize_model_path(model_path)


def test_config_partial_update_preserves_mtp_fields(config_manager):
    model_path = "/models/mtp.gguf"
    config_manager.update_model_settings(
        model_path,
        {"mtp_enabled": True, "mtp_draft_tokens": 4},
    )
    config_manager.update_model_settings(model_path, {"context_size": 32768})
    saved = config_manager.get_model_settings(model_path)
    assert saved["mtp_enabled"] is True
    assert saved["mtp_draft_tokens"] == 4
    assert saved["context_size"] == 32768


def test_config_hardware_incapable_persisted_and_cleared(config_manager):
    model_path = "/models/huge.gguf"
    config_manager.update_model_settings(
        model_path,
        {
            "context_size": 65536,
            "hardware_incapable": True,
            "hardware_incapable_message": "VRAM insuficiente",
        },
    )
    saved = config_manager.get_model_settings(model_path)
    assert saved["hardware_incapable"] is True
    assert saved["hardware_incapable_message"] == "VRAM insuficiente"

    config_manager.update_model_settings(
        model_path,
        {"hardware_incapable": False, "hardware_incapable_message": None},
    )
    cleared = config_manager.get_model_settings(model_path)
    assert cleared["hardware_incapable"] is False
    assert cleared.get("hardware_incapable_message") is None


def test_config_partial_update_preserves_hardware_incapable(config_manager):
    model_path = "/models/huge.gguf"
    config_manager.update_model_settings(
        model_path,
        {
            "hardware_incapable": True,
            "hardware_incapable_message": "Nao cabe",
        },
    )
    config_manager.update_model_settings(model_path, {"context_size": 32768})
    saved = config_manager.get_model_settings(model_path)
    assert saved["hardware_incapable"] is True
    assert saved["hardware_incapable_message"] == "Nao cabe"
    assert saved["context_size"] == 32768


def test_normalize_model_path_maps_windows_drive_on_linux():
    """Skip on Windows since os.path.abspath resolves to Windows paths."""
    import sys
    if sys.platform == "win32":
        pytest.skip("Windows path resolution differs from POSIX")
    import config_manager as cm
    def _fake_abspath(p):
        return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
    mock_os = MagicMock(spec=os, name='posix_os', abspath=_fake_abspath)
    mock_os.name = 'posix'
    with patch.object(cm, 'os', mock_os):
        result = cm.normalize_model_path("Z:/media/docker/models/model.gguf")
        assert result == "/media/docker/models/model.gguf"


def test_config_migrates_windows_paths_and_invalid_defaults(tmp_path):
    import sys
    if sys.platform == "win32":
        pytest.skip("Windows path resolution differs from POSIX")
    cfg_path = tmp_path / "automanager_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "default_models": ["Z:/media/docker/models/model.gguf"],
                "model_configs": {
                    "Z:/media/docker/models/model.gguf": {"context_size": 4096}
                },
            }
        ),
        encoding="utf-8",
    )
    manager = ConfigManager(str(cfg_path))
    cfg_file = str(cfg_path)

    def exists_side_effect(path):
        if os.path.normpath(str(path)) == os.path.normpath(cfg_file):
            return True
        return False

    import config_manager as cm
    def _fake_abspath(p):
        return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
    mock_os = MagicMock(spec=os, name="posix_os", abspath=_fake_abspath)
    mock_os.name = "posix"
    mock_os.path.exists = exists_side_effect
    mock_os.path.normpath = os.path.normpath
    mock_os.path.isabs = os.path.isabs

    with patch.object(cm, "os", mock_os):
        loaded = manager.load()

    assert loaded["default_models"] == []
    assert "/media/docker/models/model.gguf" in loaded["model_configs"]
    assert "Z:/media/docker/models/model.gguf" not in loaded["model_configs"]


def test_config_set_and_get_default_model(config_manager):
    model_path = "/models/default.gguf"
    config_manager.set_default_model(model_path)
    assert config_manager.get_default_models() == [normalize_model_path(model_path)]


def test_config_clear_default_model(config_manager):
    # Setup initial default
    config_manager.set_default_model("/some/path")
    # Clear it
    config_manager.set_default_model("/some/path", add=False)
    assert config_manager.get_default_models() == []


def test_auth_session_expires_after_idle(auth_manager):
    from datetime import datetime, timedelta, timezone

    from config_manager import SESSION_IDLE_SECONDS

    result = auth_manager.authenticate("admin", "admin")
    assert result is not None
    if isinstance(result, dict):
        token = result["token"]
    else:
        token = result
    assert auth_manager.verify_session(token) is True

    stale = datetime.now(timezone.utc) - timedelta(seconds=SESSION_IDLE_SECONDS + 1)
    auth_manager._sessions[token] = stale
    assert auth_manager.verify_session(token) is False
    assert token not in auth_manager._sessions


def test_token_generate_format(token_manager):
    token = token_manager.generate()
    assert token.startswith("sk-")
    assert len(token) >= 35


def test_token_generate_is_unique(token_manager):
    assert token_manager.generate() != token_manager.generate()


def test_token_validate_accepts_valid_token(token_manager):
    assert token_manager.validate("sk-" + "a" * 32) is True


@pytest.mark.parametrize(
    "token",
    ["", "not-a-token", "pk-" + "a" * 32, "sk-" + "a" * 31, None],
)
def test_token_validate_rejects_invalid_tokens(token_manager, token):
    assert token_manager.validate(token) is False


def test_token_get_or_create_creates_and_persists_token(token_manager, config_manager):
    token = token_manager.get_or_create()
    assert token_manager.validate(token) is True
    assert config_manager.load()["api_token"] == token


def test_token_get_or_create_preserves_existing_valid_token(token_manager, config_manager):
    existing = "sk-" + "b" * 32
    config_manager.save({"api_token": existing})
    assert token_manager.get_or_create() == existing


def test_token_renew_replaces_existing_token(token_manager, config_manager):
    old_token = token_manager.get_or_create()
    new_token = token_manager.renew()
    assert token_manager.validate(new_token) is True
    assert new_token != old_token
    assert config_manager.load()["api_token"] == new_token
