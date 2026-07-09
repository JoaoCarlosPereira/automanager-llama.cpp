"""Testes da configuração do Modo Proxy Inteligente (task_01).

Cobre a chave global smart_proxy no ConfigManager, as flags por modelo
(proxy_eligible / max_parallel_requests) e a validação dos schemas.
"""
import json

import pytest
from pydantic import ValidationError

from config_manager import (
    DEFAULT_PLATFORM_CONFIG,
    DEFAULT_SMART_PROXY,
    ConfigManager,
    normalize_model_path,
)
from schemas import (
    DEFAULT_MAX_PARALLEL_REQUESTS,
    DEFAULT_PROXY_ELIGIBLE,
    ProxyConfigRequest,
    SetModelProxyRequest,
)


class TestSmartProxySettings:
    def test_defaults_when_key_missing(self, tmp_config_manager: ConfigManager):
        settings = tmp_config_manager.get_smart_proxy_settings()
        assert settings == {
            "enabled": False,
            "primary_model_path": None,
            "primary_backend_id": None,
            "ttl_minutes": 180,
            "max_wait_seconds": 30,
        }

    def test_partial_update_preserves_other_keys(
        self, tmp_config_manager: ConfigManager
    ):
        tmp_config_manager.update_smart_proxy_settings(
            {"ttl_minutes": 60, "max_wait_seconds": 10}
        )
        merged = tmp_config_manager.update_smart_proxy_settings({"enabled": True})
        assert merged["enabled"] is True
        assert merged["ttl_minutes"] == 60
        assert merged["max_wait_seconds"] == 10

    def test_primary_model_path_is_normalized(
        self, tmp_config_manager: ConfigManager
    ):
        raw = "models\\sub\\Qwen.gguf"
        merged = tmp_config_manager.update_smart_proxy_settings(
            {"primary_model_path": raw}
        )
        assert merged["primary_model_path"] == normalize_model_path(raw)

    def test_clearing_primary_model_path(self, tmp_config_manager: ConfigManager):
        tmp_config_manager.update_smart_proxy_settings(
            {"primary_model_path": "models/a.gguf"}
        )
        merged = tmp_config_manager.update_smart_proxy_settings(
            {"primary_model_path": None}
        )
        assert merged["primary_model_path"] is None

    def test_primary_backend_id_preserves_primary_model_path(
        self, tmp_config_manager: ConfigManager
    ):
        tmp_config_manager.update_smart_proxy_settings(
            {"primary_model_path": "models/a.gguf"}
        )
        merged = tmp_config_manager.update_smart_proxy_settings(
            {"primary_backend_id": "platform:codex"}
        )
        assert merged["primary_backend_id"] == "platform:codex"
        assert merged["primary_model_path"] == normalize_model_path("models/a.gguf")

    def test_setting_primary_model_path_clears_primary_backend_id(
        self, tmp_config_manager: ConfigManager
    ):
        tmp_config_manager.update_smart_proxy_settings(
            {"primary_backend_id": "platform:codex"}
        )
        merged = tmp_config_manager.update_smart_proxy_settings(
            {"primary_model_path": "models/a.gguf"}
        )
        assert merged["primary_backend_id"] is None
        assert merged["primary_model_path"] == normalize_model_path("models/a.gguf")

    def test_invalid_ints_fall_back_to_defaults(
        self, tmp_config_manager: ConfigManager
    ):
        merged = tmp_config_manager.update_smart_proxy_settings(
            {"ttl_minutes": 0, "max_wait_seconds": -5}
        )
        assert merged["ttl_minutes"] == DEFAULT_SMART_PROXY["ttl_minutes"]
        assert merged["max_wait_seconds"] == DEFAULT_SMART_PROXY["max_wait_seconds"]

    def test_persisted_to_disk_atomically(
        self, tmp_config_manager: ConfigManager, tmp_config_path
    ):
        tmp_config_manager.update_smart_proxy_settings({"enabled": True})
        with open(tmp_config_path) as f:
            raw = json.load(f)
        assert raw["smart_proxy"]["enabled"] is True

    def test_legacy_config_untouched_keys(self, tmp_config_manager: ConfigManager):
        tmp_config_manager.save({"api_token": "sk-" + "a" * 48, "model_configs": {}})
        tmp_config_manager.update_smart_proxy_settings({"enabled": True})
        config = tmp_config_manager.get_config()
        assert config["api_token"] == "sk-" + "a" * 48
        assert config["model_configs"] == {}


class TestPerModelProxyFlags:
    def test_defaults_for_unknown_model(self, tmp_config_manager: ConfigManager):
        settings = tmp_config_manager.get_model_settings("models/x.gguf")
        assert settings.get("proxy_eligible", DEFAULT_PROXY_ELIGIBLE) is True
        assert (
            settings.get("max_parallel_requests", DEFAULT_MAX_PARALLEL_REQUESTS) == 1
        )

    def test_update_and_reload_proxy_eligible(
        self, tmp_config_manager: ConfigManager
    ):
        path = "models/a.gguf"
        tmp_config_manager.update_model_settings(path, {"proxy_eligible": False})
        settings = tmp_config_manager.get_model_settings(path)
        assert settings["proxy_eligible"] is False
        assert settings["max_parallel_requests"] == DEFAULT_MAX_PARALLEL_REQUESTS

    def test_update_max_parallel_preserves_existing_fields(
        self, tmp_config_manager: ConfigManager
    ):
        path = "models/a.gguf"
        tmp_config_manager.update_model_settings(path, {"context_size": 32768})
        tmp_config_manager.update_model_settings(path, {"max_parallel_requests": 2})
        settings = tmp_config_manager.get_model_settings(path)
        assert settings["max_parallel_requests"] == 2
        assert settings["context_size"] == 32768
        assert settings["proxy_eligible"] is True


class TestPlatformProxyFlags:
    def test_defaults_for_empty_platform_configs(
        self, tmp_config_manager: ConfigManager
    ):
        configs = tmp_config_manager.get_platform_configs()
        assert configs["platform:codex"] == DEFAULT_PLATFORM_CONFIG
        assert configs["platform:claude-code"] == DEFAULT_PLATFORM_CONFIG
        assert configs["platform:google-antigravity"] == DEFAULT_PLATFORM_CONFIG

    def test_update_platform_settings_uses_backend_id_without_model_configs(
        self, tmp_config_manager: ConfigManager
    ):
        settings = tmp_config_manager.update_platform_settings(
            "platform:codex", {
                "proxy_eligible": True,
                "max_parallel_requests": 3,
                "auto_start": True,
            }
        )
        assert settings["proxy_eligible"] is True
        assert settings["max_parallel_requests"] == 3
        assert settings["auto_start"] is True
        config = tmp_config_manager.get_config()
        assert "platform:codex" in config["platform_configs"]
        assert "platform:codex" not in config.get("model_configs", {})

    def test_update_platform_settings_invalid_parallel_uses_default(
        self, tmp_config_manager: ConfigManager
    ):
        settings = tmp_config_manager.update_platform_settings(
            "platform:codex", {"max_parallel_requests": 0}
        )
        assert settings["max_parallel_requests"] == DEFAULT_MAX_PARALLEL_REQUESTS


class TestProxySchemas:
    def test_proxy_config_request_rejects_zero_ttl(self):
        with pytest.raises(ValidationError):
            ProxyConfigRequest(ttl_minutes=0)

    def test_proxy_config_request_rejects_zero_wait(self):
        with pytest.raises(ValidationError):
            ProxyConfigRequest(max_wait_seconds=0)

    def test_set_model_proxy_rejects_zero_parallel(self):
        with pytest.raises(ValidationError):
            SetModelProxyRequest(model_path="m.gguf", max_parallel_requests=0)

    def test_partial_fields_exclude_unset(self):
        req = ProxyConfigRequest(enabled=True)
        assert req.model_dump(exclude_unset=True) == {"enabled": True}

    def test_proxy_config_accepts_primary_backend_id(self):
        req = ProxyConfigRequest(primary_backend_id="platform:codex")
        assert req.model_dump(exclude_unset=True) == {
            "primary_backend_id": "platform:codex"
        }

    def test_set_model_proxy_accepts_backend_id(self):
        req = SetModelProxyRequest(
            backend_id="platform:codex", proxy_eligible=True, auto_start=True
        )
        assert req.model_dump(exclude_unset=True) == {
            "backend_id": "platform:codex",
            "proxy_eligible": True,
            "auto_start": True,
        }
