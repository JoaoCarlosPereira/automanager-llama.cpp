from unittest.mock import patch

import llama_manager
from fastapi.testclient import TestClient
from config_manager import ConfigManager


def test_card_settings_are_independent_from_physical_model_config(tmp_path):
    config_path = tmp_path / "config.json"
    manager = ConfigManager(str(config_path))
    model_path = str(tmp_path / "model.gguf")

    manager.update_model_settings(model_path, {"context_size": 4096})
    manager.update_model_settings(model_path, {"context_size": 8192}, card_id="clone-a")
    manager.update_model_settings(model_path, {"auto_start": True}, card_id="clone-a")

    saved = manager.load()
    assert manager.get_model_settings(model_path)["context_size"] == 4096
    assert saved["model_card_configs"]["clone-a"]["context_size"] == 8192
    assert saved["model_card_configs"]["clone-a"]["auto_start"] is True


def test_duplicate_and_remove_card_without_deleting_model(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    config = {"model_configs": {str(model_path): {"context_size": 4096}}}

    with patch.object(llama_manager.auth_manager, "check_auth", return_value=True), \
         patch.object(llama_manager.model_scanner, "models_dir", str(tmp_path)), \
         patch.object(llama_manager.config_manager, "load", return_value=config), \
         patch.object(llama_manager.config_manager, "save") as save:
        test_client = TestClient(llama_manager.app)
        duplicate = test_client.post(
            "/models/duplicate",
            json={"path": str(model_path)},
        )
        assert duplicate.status_code == 200
        card_id = duplicate.json()["card_id"]
        assert model_path.exists()
        assert config["model_card_configs"][card_id]["context_size"] == 4096

        removed = test_client.post(
            "/delete",
            json={"path": str(model_path), "card_id": card_id},
        )
        assert removed.status_code == 200
        assert removed.json()["file_deleted"] is False
        assert model_path.exists()
        assert save.call_count == 2
