import pytest
from config_manager import ConfigManager

def test_generic_openai_config_crud(tmp_path):
    cm = ConfigManager(str(tmp_path / "config.json"))

    acc = cm.add_generic_openai_account("Test", "http://test/v1/", "sk-123")
    assert acc["base_url"] == "http://test/v1" # trailing slash removed
    assert "sk-" in acc["api_key"] and "***" in acc["api_key"]

    raw = cm.get_generic_openai_accounts_raw()
    assert raw[0]["api_key"] == "sk-123"

    cm.update_generic_openai_account(acc["id"], {"status": "cooldown"})
    updated = cm.get_generic_openai_accounts()
    assert updated[0]["status"] == "cooldown"

    cm.remove_generic_openai_account(acc["id"])
    assert len(cm.get_generic_openai_accounts()) == 0


def test_remove_platform_settings_cleans_smart_proxy_references(tmp_path):
    cm = ConfigManager(str(tmp_path / "config.json"))
    backend_id = "platform:generic-openai:account-1"
    cm.update_platform_settings(backend_id, {"proxy_eligible": True})
    cm.update_smart_proxy_settings({
        "primary_backend_id": backend_id,
        "custom_priority": ["platform:codex", backend_id],
    })

    assert cm.remove_platform_settings(backend_id) is True

    assert backend_id not in cm.get_config().get("platform_configs", {})
    proxy = cm.get_smart_proxy_settings()
    assert proxy["primary_backend_id"] is None
    assert backend_id not in proxy["custom_priority"]
