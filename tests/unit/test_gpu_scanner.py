import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gpu_manager import GPUDetector, GPUManager
from model_manager import ModelScanner


class TestGPUDetectorDetectGpus:
    def test_detect_gpus_uses_nvidia_smi_fallback(self):
        detector = GPUDetector()
        smi_output = (
            "0, NVIDIA RTX 4090, 24564\n"
            "1, NVIDIA RTX 3090, 24576\n"
        )

        with patch(
            "gpu_manager.subprocess.check_output",
            side_effect=[b"", smi_output.encode()],
        ) as mock_check_output:
            result = detector.detect_gpus()

        assert result == [
            {"index": 0, "name": "NVIDIA RTX 4090", "vram": 24564},
            {"index": 1, "name": "NVIDIA RTX 3090", "vram": 24576},
        ]
        assert mock_check_output.call_count == 2
        mock_check_output.assert_any_call(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        )

    def test_detect_gpus_returns_empty_for_empty_nvidia_smi_output(self):
        detector = GPUDetector()

        with patch(
            "gpu_manager.subprocess.check_output",
            side_effect=[b"", b"\n\n"],
        ):
            result = detector.detect_gpus()

        assert result == []

    def test_detect_gpus_returns_empty_on_subprocess_error(self):
        detector = GPUDetector()

        with patch(
            "gpu_manager.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            result = detector.detect_gpus()

        assert result == []


class TestGPUDetectorGetMetrics:
    def test_get_metrics_parses_valid_nvidia_smi_output(self):
        detector = GPUDetector()
        smi_output = (
            "0, 42, 1024, 8192, 65, 125.50\n"
            "1, 0, 0, 24576, 40, 70\n"
        )
        virtual_memory = MagicMock(percent=37.5)

        with patch(
            "gpu_manager.subprocess.check_output",
            return_value=smi_output.encode(),
        ) as mock_check_output, patch(
            "gpu_manager.psutil.cpu_percent",
            return_value=12.3,
        ) as mock_cpu_percent, patch(
            "gpu_manager.psutil.virtual_memory",
            return_value=virtual_memory,
        ):
            result = detector.get_metrics()

        assert result == {
            "cpu": 12.3,
            "ram": 37.5,
            "gpus": [
                {
                    "index": 0,
                    "util": "42",
                    "mem_used": "1024",
                    "mem_total": "8192",
                    "vram_pct": 12.5,
                    "temp": "65",
                    "power": "125",
                },
                {
                    "index": 1,
                    "util": "0",
                    "mem_used": "0",
                    "mem_total": "24576",
                    "vram_pct": 0.0,
                    "temp": "40",
                    "power": "70",
                },
            ],
        }
        mock_check_output.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        )
        mock_cpu_percent.assert_called_once_with(interval=0.1)

    def test_get_metrics_returns_zero_metrics_on_error(self):
        detector = GPUDetector()

        with patch(
            "gpu_manager.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            result = detector.get_metrics()

        assert result == {"cpu": 0, "ram": 0, "gpus": []}


class TestModelScannerScan:
    @pytest.fixture
    def config_manager_cls(self):
        manager = MagicMock()
        manager.load.return_value = {"model_configs": {}}
        return MagicMock(return_value=manager)

    def test_scan_classifies_models_projectors_and_mmproj_candidates(
        self,
        tmp_path,
        config_manager_cls,
    ):
        models_dir = tmp_path / "models"
        vision_dir = models_dir / "vision"
        text_dir = models_dir / "text"
        vision_dir.mkdir(parents=True)
        text_dir.mkdir(parents=True)

        model_path = vision_dir / "llava-7b.gguf"
        matching_projector_path = vision_dir / "llava-7b-mmproj.gguf"
        other_projector_path = text_dir / "clip-projector.mmproj"
        text_model_path = text_dir / "mistral.gguf"
        for file_path in [
            model_path,
            matching_projector_path,
            other_projector_path,
            text_model_path,
        ]:
            file_path.write_text("", encoding="utf-8")

        with patch("model_manager.MODELS_DIR", str(models_dir)), patch(
            "model_manager.ConfigManager",
            config_manager_cls,
        ):
            result = ModelScanner(
                config_manager_cls.return_value,
                MagicMock(),
                models_dir=str(models_dir),
            ).scan()

        model_names = {item["name"] for item in result["models"]}
        projector_names = {item["name"] for item in result["projectors"]}

        assert model_names == {"llava-7b.gguf", "mistral.gguf"}
        assert projector_names == {"llava-7b-mmproj.gguf", "clip-projector.mmproj"}

        llava_model = next(
            item for item in result["models"] if item["name"] == "llava-7b.gguf"
        )
        mistral_model = next(
            item for item in result["models"] if item["name"] == "mistral.gguf"
        )

        assert llava_model["path"] == str(model_path)
        assert llava_model["dir"] == "vision"
        assert llava_model["last_config"] is None
        assert llava_model["mmproj_candidates"] == [str(matching_projector_path)]
        assert llava_model["auto_mmproj"] == str(matching_projector_path)

        assert mistral_model["mmproj_candidates"] == []
        assert mistral_model["auto_mmproj"] is None

    def test_scan_attaches_saved_last_config(
        self,
        tmp_path,
        config_manager_cls,
    ):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "saved.gguf"
        projector_path = models_dir / "saved-mmproj.mmproj"
        model_path.write_text("", encoding="utf-8")
        projector_path.write_text("", encoding="utf-8")

        config_manager_cls.return_value.load.return_value = {
            "model_configs": {
                str(model_path): {"context_size": 4096},
                str(projector_path): {"mmproj_path": str(projector_path)},
            }
        }

        with patch("model_manager.MODELS_DIR", str(models_dir)), patch(
            "model_manager.ConfigManager",
            config_manager_cls,
        ):
            result = ModelScanner(
                config_manager_cls.return_value,
                MagicMock(),
                models_dir=str(models_dir),
            ).scan()

        assert result["models"][0]["last_config"] == {"context_size": 4096}
        assert result["projectors"][0]["last_config"] == {
            "mmproj_path": str(projector_path)
        }

    def test_scan_returns_empty_lists_for_empty_directory(
        self,
        tmp_path,
        config_manager_cls,
    ):
        with patch(
            "model_manager.ConfigManager",
            config_manager_cls,
        ):
            result = ModelScanner(
                config_manager_cls.return_value,
                MagicMock(),
                models_dir=str(tmp_path),
            ).scan()

        assert result == {"models": [], "projectors": []}
