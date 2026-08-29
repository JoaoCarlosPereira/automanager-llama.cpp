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
