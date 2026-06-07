"""Tests for CPUInfo detection (detect_cpu_info)."""

from unittest.mock import MagicMock, patch

import pytest

from gpu_manager import CPUInfo, GPUDetector, _sanitize_cpu_name


def _make_proc_mock(lines):
    """Create a context-manager mock that yields lines when iterated."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=iter(lines))
    ctx.__exit__ = MagicMock(return_value=None)
    return ctx


class TestGPUDetectorDetectCpuInfo:
    """Tests for GPUDetector.detect_cpu_info()."""

    def test_detect_cpu_info_linux_reads_first_model_name_from_proc_cpuinfo(self):
        """Ensure only the first 'model name' line is used."""
        lines = [
            "processor\t: 0\n",
            "model name\t: AMD EPYC 7763\n",
            "cpu MHz\t: 200\n",
            "processor\t: 1\n",
            "model name\t: AMD EPYC 7763 v2\n",
        ]
        virtual_memory = MagicMock(
            total=34_359_738_368,
            available=34_359_738_368 - 8_589_934_592,
        )

        with patch("gpu_manager.open", return_value=_make_proc_mock(lines)), patch(
            "gpu_manager.psutil.virtual_memory", return_value=virtual_memory
        ), patch("gpu_manager.os.name", "posix"):
            result = GPUDetector().detect_cpu_info()

        assert result.name == "AMD EPYC 7763"
        assert "7763 v2" not in result.name

    def test_sanitize_cpu_name_strips_intel_r_mark_and_clock(self):
        assert (
            _sanitize_cpu_name("Intel(R) Xeon(R) CPU E5-2676 v3 @ 2.40GHz")
            == "Xeon CPU E5-2676 v3"
        )
        assert _sanitize_cpu_name("Intel64 Family 6 Model 186") == "Intel64 Family 6 Model 186"

    def test_detect_cpu_info_strips_clock_frequency_suffix(self):
        lines = ["model name\t: Intel(R) Xeon(R) CPU E5-2676 v3 @ 2.40GHz\n"]
        virtual_memory = MagicMock(
            total=34_359_738_368,
            available=34_359_738_368 - 8_589_934_592,
        )

        with patch("gpu_manager.open", return_value=_make_proc_mock(lines)), patch(
            "gpu_manager.psutil.virtual_memory", return_value=virtual_memory
        ), patch("gpu_manager.os.name", "posix"):
            result = GPUDetector().detect_cpu_info()

        assert result.name == "Xeon CPU E5-2676 v3"
        assert "Intel" not in result.name
        assert "(R)" not in result.name
        assert "GHz" not in result.name
        assert "@" not in result.name

    def test_detect_cpu_info_linux_parses_proc_cpuinfo(self):
        lines = ["model name\t: AMD Ryzen 9 7950X\n"]
        virtual_memory = MagicMock(
            total=34_359_738_368,
            available=34_359_738_368 - 8_589_934_592,
        )

        with patch("gpu_manager.open", return_value=_make_proc_mock(lines)), patch(
            "gpu_manager.psutil.virtual_memory", return_value=virtual_memory
        ), patch("gpu_manager.os.name", "posix"):
            result = GPUDetector().detect_cpu_info()

        assert isinstance(result, CPUInfo)
        assert result.name == "AMD Ryzen 9 7950X"
        assert result.ram_total_mb == round(34_359_738_368 / (1024 * 1024))
        assert result.ram_used_mb == round(8_589_934_592 / (1024 * 1024))

    def test_detect_cpu_info_linux_fallback_when_proc_not_found(self):
        virtual_memory = MagicMock(
            total=16_000_000_000,
            available=16_000_000_000 - 4_000_000_000,
        )

        with patch("gpu_manager.open", side_effect=FileNotFoundError), patch(
            "gpu_manager.psutil.virtual_memory", return_value=virtual_memory
        ), patch("gpu_manager.os.name", "posix"), patch(
            "gpu_manager.platform.processor", return_value="x86_64"
        ):
            result = GPUDetector().detect_cpu_info()

        assert result.name == "x86_64"

    def test_detect_cpu_info_windows_platform_processor(self):
        virtual_memory = MagicMock(
            total=17_179_869_184,
            available=17_179_869_184 - 5_000_000_000,
        )

        with patch("gpu_manager.psutil.virtual_memory", return_value=virtual_memory), patch(
            "gpu_manager.os.name", "nt"
        ), patch("gpu_manager.platform.processor", return_value="Intel64 Family 6 Model 186"):
            result = GPUDetector().detect_cpu_info()

        assert isinstance(result, CPUInfo)
        assert result.name == "Intel64 Family 6 Model 186"

    def test_detect_cpu_info_windows_registry_fallback(self):
        virtual_memory = MagicMock(
            total=16_000_000_000,
            available=16_000_000_000 - 3_000_000_000,
        )

        mock_key = MagicMock()
        mock_key.__enter__ = MagicMock(return_value=mock_key)
        mock_key.__exit__ = MagicMock(return_value=None)

        mock_winreg = MagicMock()
        mock_winreg.OpenKey.return_value = mock_key
        mock_winreg.QueryValueEx.return_value = ("AMD Ryzen Threadripper PRO 5955WX", None)

        with patch("gpu_manager.psutil.virtual_memory", return_value=virtual_memory), patch(
            "gpu_manager.os.name", "nt"
        ), patch("gpu_manager.platform.processor", return_value=""), patch.dict(
            "sys.modules", {"winreg": mock_winreg}
        ):
            result = GPUDetector().detect_cpu_info()

        assert isinstance(result, CPUInfo)
        assert result.name == "AMD Ryzen Threadripper PRO 5955WX"

    def test_detect_cpu_info_windows_registry_failure_falls_back_to_machine(self):
        virtual_memory = MagicMock(
            total=16_000_000_000,
            available=16_000_000_000 - 3_000_000_000,
        )

        mock_key = MagicMock()
        mock_key.__enter__ = MagicMock(side_effect=PermissionError)
        mock_key.__exit__ = MagicMock(return_value=None)

        mock_winreg = MagicMock()
        mock_winreg.OpenKey.return_value = mock_key

        with patch("gpu_manager.psutil.virtual_memory", return_value=virtual_memory), patch(
            "gpu_manager.os.name", "nt"
        ), patch("gpu_manager.platform.processor", return_value=""), patch(
            "gpu_manager.platform.machine", return_value="AMD64"
        ), patch.dict("sys.modules", {"winreg": mock_winreg}):
            result = GPUDetector().detect_cpu_info()

        assert result.name == "AMD64"

    def test_detect_cpu_info_returns_unknown_cpu_on_total_failure(self):
        """When processor() and machine() are both empty and winreg fails, fallback is 'Unknown CPU'."""
        virtual_memory = MagicMock(
            total=16_000_000_000,
            available=16_000_000_000 - 3_000_000_000,
        )

        mock_key = MagicMock()
        mock_key.__enter__ = MagicMock(side_effect=PermissionError("access denied"))
        mock_key.__exit__ = MagicMock(return_value=None)

        mock_winreg = MagicMock()
        mock_winreg.OpenKey.return_value = mock_key

        with patch("gpu_manager.psutil.virtual_memory", return_value=virtual_memory), patch(
            "gpu_manager.platform.processor", return_value=""
        ), patch("gpu_manager.platform.machine", return_value=""), patch.dict(
            "sys.modules", {"winreg": mock_winreg}
        ):
            result = GPUDetector().detect_cpu_info()

        assert result.name == "Unknown CPU"

    def test_detect_cpu_info_ram_values_in_megabytes(self):
        """Verify RAM values are correctly converted from bytes to MB."""
        total_bytes = 8_589_934_592  # 8 GB
        used_bytes = 2_147_483_648   # 2 GB

        virtual_memory = MagicMock(total=total_bytes, available=total_bytes - used_bytes)

        with patch("gpu_manager.psutil.virtual_memory", return_value=virtual_memory), patch(
            "gpu_manager.os.name", "posix"
        ), patch("gpu_manager.open", side_effect=FileNotFoundError), patch(
            "gpu_manager.platform.processor", return_value="test_cpu"
        ):
            result = GPUDetector().detect_cpu_info()

        assert result.ram_total_mb == 8192
        assert result.ram_used_mb == 2048
