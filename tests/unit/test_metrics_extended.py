"""Tests for extended /metrics response (cpu_name, ram_total_mb, ram_used_mb)."""

import subprocess
from unittest.mock import MagicMock, patch

from gpu_manager import CPUInfo, GPUDetector


class TestGetMetricsExtended:
    """Tests for GPUDetector.get_metrics() with CPU name and RAM details."""

    def test_get_metrics_includes_cpu_name(self):
        """Verify cpu_name field is present and comes from detect_cpu_info."""
        fake_vm = MagicMock(
            total=16_000_000_000,
            available=12_000_000_000,
            percent=25.0,
        )
        fake_cpu = CPUInfo(name="Intel Core i9-13900K", ram_total_mb=15258, ram_used_mb=7629)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=35.2), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        assert result["cpu_name"] == "Intel Core i9-13900K"

    def test_get_metrics_includes_ram_total_mb(self):
        """Verify ram_total_mb field is present with correct value."""
        fake_vm = MagicMock(
            total=34_359_738_368,
            available=34_359_738_368 - 8_589_934_592,
            percent=24.2,
        )
        fake_cpu = CPUInfo(name="AMD Ryzen 9 7950X", ram_total_mb=32768, ram_used_mb=8192)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n1, 7, 1024, 24576, 45, 95.25\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=12.0), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        assert result["ram_total_mb"] == 32768

    def test_get_metrics_includes_ram_used_mb(self):
        """Verify ram_used_mb field is present with correct value."""
        fake_vm = MagicMock(
            total=34_359_738_368,
            available=34_359_738_368 - 8_589_934_592,
            percent=24.2,
        )
        fake_cpu = CPUInfo(name="AMD Ryzen 9 7950X", ram_total_mb=32768, ram_used_mb=8192)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=12.0), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        assert result["ram_used_mb"] == 8192

    def test_get_metrics_contains_all_expected_fields(self):
        """Full contract: all expected keys must be present."""
        fake_vm = MagicMock(
            total=16_000_000_000,
            available=12_000_000_000,
            percent=67.3,
        )
        fake_cpu = CPUInfo(name="Test CPU", ram_total_mb=15258, ram_used_mb=7629)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=35.2), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        expected_keys = {
            "cpu",
            "cpu_name",
            "cpu_temp",
            "cpu_power",
            "ram",
            "ram_total_mb",
            "ram_used_mb",
            "gpus",
        }
        assert set(result.keys()) == expected_keys

    def test_get_metrics_error_returns_fallback_with_all_fields(self):
        """When nvidia-smi fails, return safe defaults with all fields."""
        with patch("gpu_manager.subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")), \
             patch("gpu_manager.psutil.cpu_percent", side_effect=RuntimeError("cpu down")), \
             patch("gpu_manager.psutil.virtual_memory", side_effect=RuntimeError("ram down")):
            result = GPUDetector().get_metrics()

        assert result["cpu_name"] == "Unknown CPU"
        assert result["ram_total_mb"] == 0
        assert result["ram_used_mb"] == 0
        assert result["gpus"] == []
        assert result["cpu"] == 0
        assert result["ram"] == 0

    def test_get_metrics_ram_still_works_when_nvidia_smi_fails(self):
        """GPU metrics may fail independently; host RAM must still update."""
        total = 16_000_000_000
        used = 4_000_000_000
        fake_vm = MagicMock(total=total, available=total - used, percent=25.0)

        with patch("gpu_manager.subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")), \
             patch("gpu_manager.psutil.cpu_percent", return_value=10.0), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=CPUInfo(name="Host CPU", ram_total_mb=15258, ram_used_mb=3814)):
            result = GPUDetector().get_metrics()

        assert result["gpus"] == []
        assert result["ram"] == 25.0
        assert result["ram_used_mb"] == round(used / (1024 * 1024))
        assert result["ram_total_mb"] == round(total / (1024 * 1024))

    def test_get_metrics_gpu_data_unchanged(self):
        """Verify GPU data is still correctly parsed after the change."""
        fake_vm = MagicMock(
            total=16_000_000_000,
            available=12_000_000_000,
            percent=45.0,
        )
        fake_cpu = CPUInfo(name="Test CPU", ram_total_mb=15258, ram_used_mb=7629)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n1, 7, 1024, 24576, 45, 95.25\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=12.0), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        assert len(result["gpus"]) == 2
        assert result["gpus"][0]["index"] == 0
        assert result["gpus"][0]["mem_used"] == "8192"
        assert result["gpus"][0]["mem_total"] == "24564"
        assert result["gpus"][0]["vram_pct"] == 33.3
        assert result["gpus"][0]["temp"] == "61"
        assert result["gpus"][0]["power"] == "240"

    def test_get_metrics_returns_ram_percent(self):
        """Verify ram field is the psutil virtual_memory percent."""
        fake_vm = MagicMock(
            total=16_000_000_000,
            available=12_000_000_000,
            percent=67.3,
        )
        fake_cpu = CPUInfo(name="Test CPU", ram_total_mb=15258, ram_used_mb=7629)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=35.2), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        assert result["ram"] == 67.3

    def test_get_metrics_cpu_percent_field_uses_psutil(self):
        """Verify the cpu field is psutil.cpu_percent(interval=0.1) value."""
        fake_vm = MagicMock(
            total=16_000_000_000,
            available=12_000_000_000,
            percent=45.0,
        )
        fake_cpu = CPUInfo(name="Test CPU", ram_total_mb=15258, ram_used_mb=7629)

        with patch("gpu_manager.subprocess.check_output", return_value=b"0, 42, 8192, 24564, 61, 240.50\n"), \
             patch("gpu_manager.psutil.cpu_percent", return_value=42.7), \
             patch("gpu_manager.psutil.virtual_memory", return_value=fake_vm), \
             patch("gpu_manager.GPUDetector.detect_cpu_info", return_value=fake_cpu):
            result = GPUDetector().get_metrics()

        assert result["cpu"] == 42.7
