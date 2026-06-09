"""Unit tests for model_manager download and projector scoping."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from model_manager import (
    DownloadManager,
    ModelScanner,
    _projector_paths_for_model,
)


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
