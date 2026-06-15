"""Extended tests for model_manager.py covering SSRF, path traversal, download, and edge cases."""
import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
from fastapi import HTTPException


class TestSSRFPrevention:
    """Tests for SSRF prevention in download URL validation."""

    def test_valid_http_url(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http://example.com/model.gguf") is True

    def test_valid_https_url(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("https://huggingface.co/model.gguf") is True

    def test_valid_huggingface_url(self):
        from model_manager import _validate_download_url
        assert _validate_download_url(
            "https://huggingface.co/user/model/resolve/main/model.gguf"
        ) is True

    def test_block_localhost_hostname(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http://localhost/model.gguf") is False
        assert _validate_download_url("https://localhost/model.gguf") is False

    def test_block_private_ip_127(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http://127.0.0.1/model.gguf") is False

    def test_block_private_ip_10(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http://10.0.0.1/model.gguf") is False

    def test_block_private_ip_192_168(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http://192.168.1.1/model.gguf") is False

    def test_block_ipv6_loopback(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http://[::1]/model.gguf") is False

    def test_block_file_protocol(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("file:///path/to/model.gguf") is False

    def test_block_empty_host(self):
        from model_manager import _validate_download_url
        assert _validate_download_url("http:///path/to/model.gguf") is False

    def test_block_google_metadata(self):
        from model_manager import _validate_download_url
        assert _validate_download_url(
            "http://metadata.google.internal/model.gguf"
        ) is False


class TestModelScannerRenameDelete:
    """Tests for ModelScanner.rename_model and delete_model methods."""

    def test_rename_model_success(self, tmp_path):
        """Test ModelScanner.rename_model renames file correctly."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager
        from process_manager import ProcessManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock(spec=ProcessManager)
        pm.get_status.return_value = {"instances": []}

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        old_path = os.path.join(models_dir, "old_name.gguf")
        with open(old_path, "w") as f:
            f.write("test")

        scanner = ModelScanner(config, pm, models_dir)
        new_path = scanner.rename_model(old_path, "new_name")

        assert not os.path.exists(old_path)
        assert os.path.exists(new_path)
        assert new_path.endswith("new_name.gguf")

    def test_rename_model_adds_gguf_extension(self, tmp_path):
        """Test rename_model adds .gguf if missing."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()
        pm.get_status.return_value = {"instances": []}

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        old_path = os.path.join(models_dir, "model.gguf")
        with open(old_path, "w") as f:
            f.write("test")

        scanner = ModelScanner(config, pm, models_dir)
        new_path = scanner.rename_model(old_path, "no_ext")
        assert new_path.endswith(".gguf")

    def test_rename_model_raises_if_running(self, tmp_path):
        """Test rename_model raises HTTPException when model is running."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager
        from config_manager import normalize_model_path

        config = ConfigManager(str(tmp_path / "config.json"))
        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "running.gguf")
        with open(model_path, "w") as f:
            f.write("test")

        pm = MagicMock()
        norm = normalize_model_path(model_path)
        pm.get_status.return_value = {
            "instances": [{"status": "running", "model_path": norm}]
        }

        scanner = ModelScanner(config, pm, models_dir)
        with pytest.raises(HTTPException):
            scanner.rename_model(model_path, "renamed")

    def test_rename_model_raises_if_dest_exists(self, tmp_path):
        """Test rename_model raises if destination already exists."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()
        pm.get_status.return_value = {"instances": []}

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        src = os.path.join(models_dir, "source.gguf")
        dst_file = os.path.join(models_dir, "dest.gguf")
        with open(src, "w") as f:
            f.write("source")
        with open(dst_file, "w") as f:
            f.write("dest")

        scanner = ModelScanner(config, pm, models_dir)
        with pytest.raises(HTTPException):
            scanner.rename_model(src, "dest")

    def test_delete_model_success(self, tmp_path):
        """Test ModelScanner.delete_model removes file and config refs."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()
        pm.get_status.return_value = {"instances": []}

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "model.gguf")
        with open(model_path, "w") as f:
            f.write("test")

        scanner = ModelScanner(config, pm, models_dir)
        scanner.delete_model(model_path)

        assert not os.path.exists(model_path)

    def test_delete_model_raises_for_path_traversal(self, tmp_path):
        """Test delete_model raises HTTPException for path traversal."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        scanner = ModelScanner(config, pm, models_dir)

        with pytest.raises(HTTPException):
            scanner.delete_model("/etc/passwd")

    def test_rename_model_raises_for_path_traversal(self, tmp_path):
        """Test rename_model raises HTTPException for path traversal."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()
        pm.get_status.return_value = {"instances": []}

        models_dir = str(tmp_path / "models")
        scanner = ModelScanner(config, pm, models_dir)

        with pytest.raises(HTTPException):
            scanner.rename_model("/etc/passwd", "evil")


class TestDownloadManager:
    """Tests for DownloadManager functionality."""

    def test_start_download_returns_download_id(self, tmp_path):
        """Test start_download returns a UUID string."""
        from model_manager import DownloadManager

        dm = DownloadManager(str(tmp_path))
        download_id = dm.start_download("https://example.com/model.gguf")
        assert isinstance(download_id, str)
        assert len(download_id) > 0

    def test_start_download_raises_for_private_ip(self, tmp_path):
        """Test start_download raises HTTPException for private IPs."""
        from model_manager import DownloadManager
        from fastapi import HTTPException

        dm = DownloadManager(str(tmp_path))
        with pytest.raises(HTTPException):
            dm.start_download("http://192.168.1.1/model.gguf")

    def test_start_download_raises_for_localhost(self, tmp_path):
        from model_manager import DownloadManager
        from fastapi import HTTPException

        dm = DownloadManager(str(tmp_path))
        with pytest.raises(HTTPException):
            dm.start_download("http://localhost/model.gguf")

    def test_start_download_creates_family_dir(self, tmp_path):
        """Test start_download creates model family directory."""
        from model_manager import DownloadManager

        dm = DownloadManager(str(tmp_path))
        dm.start_download("https://example.com/qwen3.6-7b.gguf")
        family_dirs = [d for d in os.listdir(str(tmp_path))
                       if os.path.isdir(os.path.join(str(tmp_path), d))]
        assert "qwen3.6" in family_dirs

    def test_get_progress_empty(self):
        from model_manager import DownloadManager

        dm = DownloadManager()
        progress = dm.get_progress()
        assert progress == {}

    def test_get_progress_with_download_entry(self, tmp_path):
        """Test get_progress returns entries with status info."""
        from model_manager import DownloadManager

        dm = DownloadManager(str(tmp_path))
        download_id = dm.start_download("https://example.com/model.gguf")
        progress = dm.get_progress()
        assert download_id in progress
        entry = progress[download_id]
        assert entry["status"] == "downloading"
        assert "progress" in entry
        assert "url" in entry

    def test_cancel_download_returns_true(self, tmp_path):
        """Test cancel_download returns True for active download."""
        from model_manager import DownloadManager

        dm = DownloadManager(str(tmp_path))
        download_id = dm.start_download("https://example.com/model.gguf")
        result = dm.cancel_download(download_id)
        assert result is True

    def test_cancel_download_returns_false_for_unknown_id(self, tmp_path):
        """Test cancel_download returns False for unknown download_id."""
        from model_manager import DownloadManager

        dm = DownloadManager(str(tmp_path))
        result = dm.cancel_download("nonexistent-id")
        assert result is False

    def test_cancel_download_returns_false_if_already_done(self, tmp_path):
        """Test cancel_download returns False if download already cancelled."""
        from model_manager import DownloadManager

        dm = DownloadManager(str(tmp_path))
        download_id = dm.start_download("https://example.com/model.gguf")
        dm.cancel_download(download_id)
        result = dm.cancel_download(download_id)
        assert result is False

    def test_clear_completed_returns_count(self):
        from model_manager import DownloadManager

        dm = DownloadManager()
        dm._downloads["id1"] = {"status": "completed", "progress": 100}
        dm._downloads["id2"] = {"status": "cancelled"}
        dm._downloads["id3"] = {"status": "downloading"}
        count = dm.clear_completed()
        assert count == 2
        assert "id1" not in dm._downloads
        assert "id2" not in dm._downloads
        assert "id3" in dm._downloads


class TestModelScanner:
    """Tests for ModelScanner.scan method."""

    def test_scan_returns_models_projector_storage(self, tmp_path):
        """Test scan returns dict with models, projectors, and storage."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        scanner = ModelScanner(config, pm, models_dir)

        # Create test files
        with open(os.path.join(models_dir, "model.gguf"), "w") as f:
            f.write("test")
        with open(os.path.join(models_dir, "mmproj.mmproj"), "w") as f:
            f.write("test")

        result = scanner.scan()
        assert "models" in result
        assert "projectors" in result
        assert "storage" in result

    def test_scan_finds_gguf_files(self, tmp_path):
        """Test scan discovers .gguf model files."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        scanner = ModelScanner(config, pm, models_dir)
        with open(os.path.join(models_dir, "model1.gguf"), "w") as f:
            f.write("test")
        with open(os.path.join(models_dir, "model2.gguf"), "w") as f:
            f.write("test")

        result = scanner.scan()
        assert len(result["models"]) == 2

    def test_scan_finds_projectors(self, tmp_path):
        """Test scan discovers .mmproj projector files."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        scanner = ModelScanner(config, pm, models_dir)
        with open(os.path.join(models_dir, "mmproj-model.mmproj"), "w") as f:
            f.write("test")

        result = scanner.scan()
        assert len(result["projectors"]) == 1

    def test_scan_ignores_non_gguf_non_mmproj(self, tmp_path):
        """Test scan only picks up .gguf and projector files."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        scanner = ModelScanner(config, pm, models_dir)
        with open(os.path.join(models_dir, "readme.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(models_dir, "data.json"), "w") as f:
            f.write("{}")

        result = scanner.scan()
        assert len(result["models"]) == 0
        assert len(result["projectors"]) == 0

    def test_scan_finds_nested_models(self, tmp_path):
        """Test scan finds models in subdirectories."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        os.makedirs(os.path.join(models_dir, "subdir"), exist_ok=True)

        scanner = ModelScanner(config, pm, models_dir)
        with open(os.path.join(models_dir, "subdir", "nested.gguf"), "w") as f:
            f.write("test")

        result = scanner.scan()
        assert len(result["models"]) == 1
        assert "nested.gguf" in result["models"][0]["name"]

    def test_scan_includes_mmproj_candidates(self, tmp_path):
        """Test scan includes mmproj_candidates for each model."""
        from model_manager import ModelScanner
        from config_manager import ConfigManager

        config = ConfigManager(str(tmp_path / "config.json"))
        pm = MagicMock()

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)

        scanner = ModelScanner(config, pm, models_dir)
        model_path = os.path.join(models_dir, "model.gguf")
        mmproj_path = os.path.join(models_dir, "mmproj.mmproj")
        with open(model_path, "w") as f:
            f.write("test")
        with open(mmproj_path, "w") as f:
            f.write("test")

        result = scanner.scan()
        assert len(result["models"]) == 1
        assert "mmproj_candidates" in result["models"][0]
        assert len(result["models"][0]["mmproj_candidates"]) == 1


class TestModelHelpers:
    """Tests for model utility functions."""

    def test_is_projector_filename_mmproj(self):
        from model_manager import _is_projector_filename
        assert _is_projector_filename("mmproj-model.gguf") is True
        assert _is_projector_filename("clip-vision.gguf") is True

    def test_is_projector_filename_no_match(self):
        from model_manager import _is_projector_filename
        assert _is_projector_filename("model.gguf") is False
        assert _is_projector_filename("llama-Q4_K_M.gguf") is False

    def test_infer_model_family_from_name(self):
        from model_manager import infer_model_family
        family = infer_model_family("qwen3.6-8b-instruct.Q4_K_M.gguf")
        assert "qwen3.6" in family

    def test_infer_model_family_from_url(self):
        from model_manager import infer_model_family
        family = infer_model_family(
            "",
            "https://huggingface.co/user/llama-3.3-model/resolve/main/model.gguf"
        )
        assert "Llama" in family or "3" in family

    def test_get_repository_storage(self, tmp_path):
        from model_manager import get_repository_storage

        models_dir = str(tmp_path / "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "file.bin"), "wb") as f:
            f.write(b"x" * 1024)

        storage = get_repository_storage(models_dir)
        assert "used_gb" in storage
        assert "total_gb" in storage
        assert "path" in storage
        assert storage["path"] == models_dir
