import json

import pytest

from config_manager import ConfigManager, TokenManager


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
    assert config_manager.load() == expected


def test_config_load_corrupted_json_returns_empty_dict(config_manager, config_path):
    with open(config_path, "w") as config_file:
        config_file.write("{invalid")
    assert config_manager.load() == {}


def test_config_save_and_load_round_trip(config_manager):
    expected = {"api_token": "sk-" + "a" * 32, "default_model": "/models/a.gguf"}
    config_manager.save(expected)
    assert config_manager.load() == expected


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
    saved = config_manager.load()["model_configs"][model_path]
    assert saved["context_size"] == 8192
    assert saved["mmproj_path"] == "/models/llama.mmproj"
    assert saved["gpu_weights"] == settings["gpu_weights"]
    assert "last_started" in saved


def test_config_set_and_get_default_model(config_manager):
    model_path = "/models/default.gguf"
    config_manager.set_default_model(model_path)
    assert config_manager.get_default_model() == model_path


def test_config_clear_default_model(config_manager):
    config_manager.set_default_model(None)
    assert config_manager.get_default_model() is None


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
