"""Tests for CPU temperature and power sensor helpers."""

from unittest.mock import MagicMock, mock_open, patch

from gpu_manager import (
    CPUInfo,
    GPUDetector,
    _format_metric_watts,
    _rapl_package_energy_uj,
    _read_cpu_temperature_c,
    _read_hwmon_cpu_power_w,
)


class TestCpuTemperature:
    def test_read_cpu_temperature_prefers_coretemp(self):
        sensors = {
            "acpitz": [MagicMock(current=40.0)],
            "coretemp": [MagicMock(current=62.0)],
        }
        with patch(
            "gpu_manager.psutil.sensors_temperatures",
            create=True,
            return_value=sensors,
        ):
            assert _read_cpu_temperature_c() == 62.0

    def test_read_cpu_temperature_falls_back_to_thermal_zone(self):
        with patch(
            "gpu_manager.psutil.sensors_temperatures",
            create=True,
            return_value={},
        ), patch(
            "gpu_manager.glob.glob",
            return_value=["/sys/class/thermal/thermal_zone0/temp"],
        ), patch("gpu_manager.os.name", "posix"), patch(
            "builtins.open",
            mock_open(read_data="48500\n"),
        ):
            assert _read_cpu_temperature_c() == 48.5


class TestCpuPower:
    def test_format_metric_watts_rounds_to_integer_string(self):
        assert _format_metric_watts(127.4) == "127"
        assert _format_metric_watts(None) is None

    def test_rapl_package_energy_reads_package_zero(self):
        with patch("gpu_manager.os.path.isdir", return_value=True), patch(
            "gpu_manager.os.listdir", return_value=["intel-rapl:0"]
        ), patch(
            "builtins.open",
            mock_open(read_data="package-0"),
        ) as mocked_open:
            mocked_open.side_effect = [
                mock_open(read_data="package-0\n").return_value,
                mock_open(read_data="9000000\n").return_value,
            ]
            assert _rapl_package_energy_uj() == 9_000_000

    def test_read_cpu_power_w_uses_hwmon_when_available(self):
        detector = GPUDetector()
        with patch("gpu_manager._read_hwmon_cpu_power_w", return_value=88.0):
            assert detector._read_cpu_power_w() == 88.0

    def test_read_cpu_power_w_computes_rapl_delta(self):
        detector = GPUDetector()
        with patch("gpu_manager._read_hwmon_cpu_power_w", return_value=None), patch(
            "gpu_manager._rapl_package_energy_uj",
            side_effect=[1_000_000, 1_200_000],
        ), patch("gpu_manager.time.monotonic", side_effect=[10.0, 10.5]):
            assert detector._read_cpu_power_w() is None
            assert detector._read_cpu_power_w() == 0.4

    def test_get_metrics_includes_cpu_temp_and_power(self):
        detector = GPUDetector()
        virtual_memory = MagicMock(
            percent=30.0,
            total=16_000_000_000,
            available=11_200_000_000,
        )
        with patch(
            "gpu_manager.subprocess.check_output",
            return_value=b"0, 10, 512, 8192, 55, 80.00\n",
        ), patch("gpu_manager.psutil.cpu_percent", return_value=22.0), patch(
            "gpu_manager.psutil.virtual_memory",
            return_value=virtual_memory,
        ), patch.object(
            detector,
            "detect_cpu_info",
            return_value=CPUInfo(name="Test CPU", ram_total_mb=0, ram_used_mb=0),
        ), patch(
            "gpu_manager._read_cpu_temperature_c",
            return_value=47.6,
        ), patch.object(
            detector,
            "_read_cpu_power_w",
            return_value=133.2,
        ):
            result = detector.get_metrics()

        assert result["cpu_temp"] == "48"
        assert result["cpu_power"] == "133"
