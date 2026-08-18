"""Unit tests for model_manager download and projector scoping."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from model_manager import (
    DownloadManager,
    ModelScanner,
    _projector_paths_for_model,
    _mtp_paths_for_model,
    infer_model_family,
)


def test_mtp_draft_matches_quantized_gemma_model():
    model = "/models/gemma-4/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
    draft = "/models/mtp-gemma-4/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    assert _mtp_paths_for_model(model, [{"path": draft}]) == [draft]


def test_scanner_separates_mtp_draft_from_chat_models(tmp_path):
    model = tmp_path / "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
    draft = tmp_path / "mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    model.write_text("", encoding="utf-8")
    draft.write_text("", encoding="utf-8")
    config = MagicMock()
    config.load.return_value = {"model_configs": {}}
    result = ModelScanner(config, MagicMock(), models_dir=str(tmp_path)).scan()
    assert [item["name"] for item in result["models"]] == [model.name]
    assert [item["name"] for item in result["mtp_models"]] == [draft.name]
    assert result["models"][0]["mtp_candidates"] == [str(draft)]


class TestProjectorScoping:
    def test_projector_paths_for_model_same_directory_only(self, tmp_path):
        model_path = tmp_path / "llava" / "model.gguf"
        same_dir_proj = tmp_path / "llava" / "model-mmproj.gguf"
        other_dir_proj = tmp_path / "other" / "model-mmproj.gguf"
        model_path.parent.mkdir(parents=True)
        (tmp_path / "other").mkdir()
        model_path.write_text("", encoding="utf-8")
        same_dir_proj.write_text("", encoding="utf-8")
        other_dir_proj.write_text("", encoding="utf-8")

        projectors = [
            {"path": str(same_dir_proj), "name": same_dir_proj.name},
            {"path": str(other_dir_proj), "name": other_dir_proj.name},
        ]

        result = _projector_paths_for_model(str(model_path), projectors)

        assert result == [str(same_dir_proj)]


class TestDownloadManagerVision:
    def test_start_download_mmproj_into_model_directory(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llava"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "llava-7b.gguf"
        model_path.write_text("", encoding="utf-8")

        mgr = DownloadManager(models_dir=str(models_dir))
        download_id = mgr.start_download(
            "https://example.com/llava-mmproj.gguf",
            model_path=str(model_path),
        )

        with mgr._lock:
            entry = mgr._downloads[download_id]

        assert entry["filename"] == "llava-mmproj.gguf"
        assert entry["path"] == str(model_dir / "llava-mmproj.gguf")
        assert entry["model_path"] == str(model_path).replace("\\", "/")

    def test_start_download_mmproj_rejects_path_outside_models_dir(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        outside = tmp_path / "outside.gguf"
        outside.write_text("", encoding="utf-8")

        mgr = DownloadManager(models_dir=str(models_dir))

        with pytest.raises(HTTPException) as exc:
            mgr.start_download(
                "https://example.com/proj.mmproj",
                model_path=str(outside),
            )

        assert exc.value.status_code == 403

    def test_start_download_mmproj_missing_model(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        missing = models_dir / "missing.gguf"

        mgr = DownloadManager(models_dir=str(models_dir))

        with pytest.raises(HTTPException) as exc:
            mgr.start_download(
                "https://example.com/proj.mmproj",
                model_path=str(missing),
            )

        assert exc.value.status_code == 404

    def test_start_download_mtp_preserves_gguf_in_model_directory(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "gemma-4"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "gemma-4-26b.gguf"
        model_path.write_text("", encoding="utf-8")
        mgr = DownloadManager(models_dir=str(models_dir))
        download_id = mgr.start_download(
            "https://example.com/mtp-gemma-4-26b-Q8_0.gguf",
            model_path=str(model_path),
            asset_type="mtp",
        )
        with mgr._lock:
            entry = mgr._downloads[download_id]
        assert entry["filename"] == "mtp-gemma-4-26b-Q8_0.gguf"
        assert entry["path"] == str(model_dir / entry["filename"])
        assert entry["asset_type"] == "mtp"


class TestModelScannerSameDirectory:
    @pytest.fixture
    def config_manager_cls(self):
        manager = MagicMock()
        manager.load.return_value = {"model_configs": {}}
        return MagicMock(return_value=manager)

    def test_scan_mmproj_candidates_ignore_other_directories(
        self,
        tmp_path,
        config_manager_cls,
    ):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "vision"
        other_dir = models_dir / "other"
        model_dir.mkdir(parents=True)
        other_dir.mkdir()

        model_path = model_dir / "llava-7b.gguf"
        local_proj = model_dir / "llava-7b-mmproj.gguf"
        remote_proj = other_dir / "llava-7b-mmproj.gguf"
        for path in (model_path, local_proj, remote_proj):
            path.write_text("", encoding="utf-8")

        with patch("model_manager.MODELS_DIR", str(models_dir)), patch(
            "model_manager.ConfigManager",
            config_manager_cls,
        ):
            result = ModelScanner(
                config_manager_cls.return_value,
                MagicMock(),
                models_dir=str(models_dir),
            ).scan()

        llava = next(m for m in result["models"] if m["name"] == "llava-7b.gguf")
        assert llava["mmproj_candidates"] == [str(local_proj)]
        assert llava["auto_mmproj"] == str(local_proj)


class TestModelScannerDelete:
    def test_delete_model_removes_file_config_and_stops_instance(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "llama-7b.gguf"
        model_path.write_text("gguf", encoding="utf-8")

        config_manager = MagicMock()
        config_manager.load.return_value = {
            "default_model": str(model_path),
            "default_models": [str(model_path)],
            "model_configs": {
                str(model_path): {"context_size": 4096},
            },
        }

        process_manager = MagicMock()
        process_manager.get_status.return_value = {
            "instances": [
                {
                    "port": 8085,
                    "status": "running",
                    "model_path": str(model_path),
                }
            ],
            "recovery": {},
        }

        scanner = ModelScanner(
            config_manager,
            process_manager,
            models_dir=str(models_dir),
        )
        scanner.delete_model(str(model_path))

        assert not model_path.exists()
        process_manager.stop.assert_called_once_with(8085)
        saved = config_manager.save.call_args[0][0]
        assert saved["default_model"] is None
        assert saved["default_models"] == []
        assert str(model_path) not in saved["model_configs"]


class TestModelScannerRename:
    def test_rename_model_updates_file_and_config(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "llama-7b.gguf"
        model_path.write_text("gguf", encoding="utf-8")

        config_manager = MagicMock()
        config_manager.load.return_value = {
            "default_model": str(model_path),
            "default_models": [str(model_path)],
            "model_configs": {
                str(model_path): {"context_size": 4096},
            },
        }

        process_manager = MagicMock()
        process_manager.get_status.return_value = {
            "instances": [],
            "recovery": {},
        }

        scanner = ModelScanner(
            config_manager,
            process_manager,
            models_dir=str(models_dir),
        )
        new_path = scanner.rename_model(str(model_path), "llama-13b")

        expected = str(model_dir / "llama-13b.gguf")
        assert new_path == expected.replace("\\", "/")
        assert not model_path.exists()
        assert os.path.exists(expected)
        saved = config_manager.save.call_args[0][0]
        assert saved["default_model"] == expected.replace("\\", "/")
        assert saved["default_models"] == [expected.replace("\\", "/")]
        assert expected.replace("\\", "/") in saved["model_configs"]

    def test_rename_model_blocks_when_instance_running(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "llama-7b.gguf"
        model_path.write_text("gguf", encoding="utf-8")

        process_manager = MagicMock()
        process_manager.get_status.return_value = {
            "instances": [
                {
                    "port": 8085,
                    "status": "running",
                    "model_path": str(model_path),
                }
            ],
            "recovery": {},
        }

        scanner = ModelScanner(
            MagicMock(),
            process_manager,
            models_dir=str(models_dir),
        )

        with pytest.raises(HTTPException) as exc:
            scanner.rename_model(str(model_path), "llama-13b")

        assert exc.value.status_code == 400
        assert model_path.exists()


class TestInferModelFamily:
    def test_qwen_family(self):
        assert infer_model_family("Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf") == "Qwen3.6"

    def test_llama_family(self):
        assert infer_model_family("Llama-3.3-70B-Instruct-Q4_K_M.gguf") == "Llama-3.3"

    def test_meta_llama_prefix_stripped(self):
        assert infer_model_family("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") == "Llama-3.1"

    def test_family_from_hf_repo(self):
        url = "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/file.gguf"
        assert infer_model_family("file.gguf", url) == "Llama-3.3"


class TestDownloadManagerFamilyAndCancel:
    def test_start_download_uses_family_directory(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        mgr = DownloadManager(models_dir=str(models_dir))
        download_id = mgr.start_download(
            "https://example.com/Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf"
        )
        with mgr._lock:
            entry = mgr._downloads[download_id]
        assert entry["family"] == "Qwen3.6"
        assert entry["path"].startswith(str(models_dir / "Qwen3.6"))
        assert entry["path"].endswith("Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")

    def test_cancel_queued_download_removes_partial_file(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        mgr = DownloadManager(models_dir=str(models_dir))
        download_id = mgr.start_download("https://example.com/Llama-3.3-70B.gguf")
        with mgr._lock:
            path = mgr._downloads[download_id]["path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"partial")
        assert mgr.cancel_download(download_id) is True
        assert not os.path.exists(path)
        assert mgr.get_progress()[download_id]["status"] == "cancelled"
