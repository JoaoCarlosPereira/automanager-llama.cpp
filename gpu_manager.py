"""GPU detection and tensor split management."""
import glob
import os
import platform
import re
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple


class OffloadPlan(NamedTuple):
    """Resolved layer/tensor split from active device weights."""

    n_gpu_layers: int
    n_cpu_layers: int
    gpu_pct: float
    cpu_pct: float
    tensor_split: List[str]
    # False when the model does not fit and CPU is disabled (block with alert).
    is_feasible: bool = True

import psutil

from llama_server_bin import get_llama_server_bin
from schemas import GPUWeight
from load_distributor import LoadDistributor, DistributionResult

# Default layer count used when model metadata cannot be read.
DEFAULT_TOTAL_LAYERS = 32
# llama-server clamps -ngl to the model layer count; use a high value for "all GPU".
ALL_GPU_LAYERS = 999

logger = logging.getLogger("automanager")


def _llama_server_cmd() -> str:
    return get_llama_server_bin()

# e.g. "Intel(R) Xeon(R) CPU E5-2676 v3 @ 2.40GHz" -> "Xeon CPU E5-2676 v3"
_CPU_FREQ_SUFFIX = re.compile(r"\s*@\s*[\d.]+\s*GHz\s*$", re.IGNORECASE)
_CPU_R_MARK = re.compile(r"\(R\)", re.IGNORECASE)
_INTEL_WORD = re.compile(r"\bIntel\b", re.IGNORECASE)
_CPU_MULTI_SPACE = re.compile(r"\s{2,}")


def _sanitize_cpu_name(name: str) -> str:
    """Normalize CPU model string for display (drop MHz suffix, Intel, (R))."""
    name = _CPU_FREQ_SUFFIX.sub("", name)
    name = _CPU_R_MARK.sub("", name)
    name = _INTEL_WORD.sub("", name)
    name = _CPU_MULTI_SPACE.sub(" ", name)
    return name.strip()


def _format_metric_watts(value: Optional[float]) -> Optional[str]:
    """Format power draw like nvidia-smi (integer watts string)."""
    if value is None:
        return None
    return str(int(round(value)))


def _read_cpu_temperature_c() -> Optional[float]:
    """Best-effort CPU temperature in Celsius (cross-platform)."""
    try:
        sensors_fn = getattr(psutil, "sensors_temperatures", None)
        temps = sensors_fn() if sensors_fn else {}
        if temps:
            preferred = ("coretemp", "k10temp", "zenpower", "acpitz", "cpu_thermal")
            for key in preferred:
                entries = temps.get(key)
                if entries:
                    current = entries[0].current
                    if current is not None:
                        return float(current)
            for entries in temps.values():
                for entry in entries:
                    if entry.current is not None:
                        return float(entry.current)
    except Exception:
        pass

    if os.name != "posix":
        return None

    try:
        for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
            with open(path, "r", encoding="utf-8") as f:
                raw = int(f.read().strip())
            if raw > 0:
                return raw / 1000.0
    except Exception:
        pass
    return None


def _rapl_package_energy_uj() -> Optional[int]:
    """Read Intel/AMD RAPL package energy counter (Linux powercap)."""
    base = "/sys/class/powercap/intel-rapl"
    if not os.path.isdir(base):
        return None
    for entry in os.listdir(base):
        name_path = os.path.join(base, entry, "name")
        energy_path = os.path.join(base, entry, "energy_uj")
        try:
            with open(name_path, "r", encoding="utf-8") as f:
                if f.read().strip() != "package-0":
                    continue
            with open(energy_path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _read_hwmon_cpu_power_w() -> Optional[float]:
    """Instantaneous CPU package power from hwmon (milliwatts), if exposed."""
    for power_path in sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input")):
        try:
            with open(power_path, "r", encoding="utf-8") as f:
                milliwatts = int(f.read().strip())
            if milliwatts > 0:
                return milliwatts / 1000.0
        except (OSError, ValueError):
            continue
    return None


def _system_ram_mb(vm=None) -> Tuple[int, int, float]:
    """
    RAM stats aligned with psutil virtual_memory().percent.

    Uses (total - available) for used bytes so MB text and % bar match.
    """
    vm = vm or psutil.virtual_memory()
    total_mb = round(vm.total / (1024 * 1024))
    used_mb = round((vm.total - vm.available) / (1024 * 1024))
    return total_mb, used_mb, vm.percent


@dataclass
class GPUInfo:
    index: int
    name: str
    vram: int


@dataclass
class CPUInfo:
    name: str
    ram_total_mb: int
    ram_used_mb: int


class GPUDetector:
    """Detects GPUs and parses metrics from nvidia-smi."""

    def __init__(self) -> None:
        self._rapl_prev: Optional[Tuple[float, int]] = None
        self._metrics_cache: Dict[str, Any] = {}
        self._metrics_cache_time: float = 0.0
        self._metrics_cache_ttl: float = 2.0  # seconds

    def _read_cpu_power_w(self) -> Optional[float]:
        """Estimate CPU package power (RAPL delta or hwmon instantaneous)."""
        hwmon_power = _read_hwmon_cpu_power_w()
        if hwmon_power is not None:
            return hwmon_power

        energy_uj = _rapl_package_energy_uj()
        if energy_uj is None:
            return None

        now = time.monotonic()
        prev = self._rapl_prev
        self._rapl_prev = (now, energy_uj)
        if prev is None:
            return None

        dt = now - prev[0]
        if dt <= 0:
            return None

        delta_uj = energy_uj - prev[1]
        if delta_uj < 0:
            return None
        return delta_uj / dt / 1_000_000

    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs using llama-server --help first, fallback to nvidia-smi."""
        try:
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            bin_path = _llama_server_cmd()
            output = subprocess.check_output(
                [bin_path, "--help"],
                env=env, timeout=10, stderr=subprocess.STDOUT,
            ).decode()
            pattern = r"Device (\d+): (.*?), compute capability.*?, VRAM: (\d+) MiB"
            matches = re.findall(pattern, output)
            gpus = []
            for match in matches:
                idx, name, vram = match
                gpus.append({
                    "index": int(idx),
                    "name": name.strip(),
                    "vram": int(vram),
                })
            if gpus:
                return gpus
        except Exception:
            pass

        # Fallback to nvidia-smi
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=10,
            ).decode()
            gpus = []
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram": int(parts[2]),
                    })
            return gpus
        except Exception as e:
            logger.error(f"GPU detection error: {e}")
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get real-time hardware metrics including CPU name and RAM details."""
        # Cache nvidia-smi calls to avoid excessive subprocess spawning
        now = time.monotonic()
        if (
            self._metrics_cache
            and (now - self._metrics_cache_time) < self._metrics_cache_ttl
        ):
            return self._metrics_cache

        gpus: List[Dict[str, Any]] = []
        try:
            output = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=10,
            ).decode()
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    mem_used = float(parts[2])
                    mem_total = float(parts[3])
                    gpus.append({
                        "index": int(parts[0]),
                        "util": parts[1],
                        "mem_used": parts[2],
                        "mem_total": parts[3],
                        "vram_pct": round(
                            (mem_used / mem_total) * 100, 1
                        ) if mem_total > 0 else 0,
                        "temp": parts[4],
                        "power": parts[5].split(".")[0] if "." in parts[5] else parts[5],
                    })
        except Exception as e:
            logger.error(f"GPU metrics error: {e}")

        try:
            vm = psutil.virtual_memory()
            ram_total_mb, ram_used_mb, ram_pct = _system_ram_mb(vm)
            cpu_name = self.detect_cpu_info().name
            cpu_temp_c = _read_cpu_temperature_c()
            cpu_temp = (
                str(int(round(cpu_temp_c))) if cpu_temp_c is not None else None
            )
            cpu_power = _format_metric_watts(self._read_cpu_power_w())
            result = {
                "cpu": psutil.cpu_percent(interval=0.1),
                "cpu_name": cpu_name,
                "cpu_temp": cpu_temp,
                "cpu_power": cpu_power,
                "ram": ram_pct,
                "ram_total_mb": ram_total_mb,
                "ram_used_mb": ram_used_mb,
                "gpus": gpus,
            }
            # Cache the result
            self._metrics_cache = result
            self._metrics_cache_time = now
            return result
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            cached = {
                "cpu": 0,
                "cpu_name": "Unknown CPU",
                "cpu_temp": None,
                "cpu_power": None,
                "ram": 0,
                "ram_total_mb": 0,
                "ram_used_mb": 0,
                "gpus": gpus,
            }
            self._metrics_cache = cached
            self._metrics_cache_time = now
            return cached

    def detect_cpu_info(self) -> CPUInfo:
        """Detect CPU name and RAM stats (cross-platform)."""
        # --- CPU name ---
        cpu_name = ""
        try:
            if os.name == "nt":
                # Windows: try platform.processor() first, fallback to registry
                cpu_name = platform.processor() or ""
                if not cpu_name:
                    try:
                        import winreg
                        with winreg.OpenKey(
                            winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                        ) as key:
                            cpu_name, _ = winreg.QueryValueEx(
                                key, "ProcessorNameString"
                            )
                    except Exception:
                        cpu_name = platform.machine()
            else:
                # Linux/Unix: parse /proc/cpuinfo for model name
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if line.startswith("model name"):
                                cpu_name = line.split(":", 1)[1].strip()
                                break
                except FileNotFoundError:
                    cpu_name = platform.processor() or platform.machine()
        except Exception:
            cpu_name = platform.machine() or "Unknown CPU"

        if not cpu_name:
            cpu_name = "Unknown CPU"

        cpu_name = _sanitize_cpu_name(cpu_name)

        # --- RAM (psutil, cross-platform) ---
        ram_total_mb, ram_used_mb, _ = _system_ram_mb()

        return CPUInfo(
            name=cpu_name,
            ram_total_mb=ram_total_mb,
            ram_used_mb=ram_used_mb,
        )



class GPUManager(GPUDetector):
    """GPU operations including strict tensor split enforcement."""

    # Estimated layer counts per parameter size (typical architectures).
    # Used to refine compute_n_gpu_layers() when the caller has model metadata.
    LAYERS_BY_PARAM: Dict[str, int] = {
        "0.5b": 24,
        "1b": 28,
        "1.5b": 28,
        "3b": 36,
        "7b": 32,
        "8b": 32,
        "10b": 40,
        "13b": 40,
        "14b": 40,
        "20b": 44,
        "30b": 60,
        "34b": 48,
        "65b": 80,
        "70b": 80,
        "72b": 80,
        "405b": 126,
    }

    @staticmethod
    def active_gpus_with_weight(gpu_weights: List[GPUWeight]) -> List[GPUWeight]:
        """Active GPU entries with weight > 0, preserving request order."""
        return [
            w
            for w in gpu_weights
            if w.active and w.device == "gpu" and w.weight > 0
        ]

    @staticmethod
    def sum_active_weight(
        gpu_weights: List[GPUWeight], device: Optional[str] = None
    ) -> float:
        """Sum weights for active devices, optionally filtered by device type."""
        devices = [w for w in gpu_weights if w.active]
        if device:
            devices = [w for w in devices if w.device == device]
        return sum(w.weight for w in devices)

    @staticmethod
    def cpu_offload_active(gpu_weights: List[GPUWeight]) -> bool:
        """True when the user enabled CPU offload with weight > 0."""
        return GPUManager.sum_active_weight(gpu_weights, "cpu") > 0

    @staticmethod
    def normalize_gpu_weights(gpu_weights: List[GPUWeight]) -> List[GPUWeight]:
        """Drop stale weight on inactive devices; keep active plan intact."""
        normalized: List[GPUWeight] = []
        for w in gpu_weights:
            data = w.model_dump() if hasattr(w, "model_dump") else dict(w)
            if data.get("device") not in ("gpu", "cpu"):
                data["device"] = "gpu"
            if not data.get("active", True) or float(data.get("weight", 0) or 0) <= 0:
                data["active"] = False
                data["weight"] = 0.0
            normalized.append(GPUWeight(**data))
        return normalized

    def compute_tensor_split(self, gpu_weights: List[GPUWeight]) -> List[str]:
        """Relative split among active GPUs; preserves user % ratios."""
        active = self.active_gpus_with_weight(gpu_weights)
        if not active:
            return []
        total = sum(w.weight for w in active) or 1.0
        return [f"{w.weight / total:.4f}" for w in active]

    def compute_offload_plan(
        self, gpu_weights: List[GPUWeight], total_layers: int = 32, cpu_enabled: Optional[bool] = None
    ) -> OffloadPlan:
        """Resolve ``-ngl``, CPU share and ``--tensor-split`` from active weights.

        Uses the unified LoadDistributor engine for GPU-first, CPU-minimum policy
        when cpu_enabled is explicitly True or False (new behavior).
        When cpu_enabled is None, uses the original proportional logic (backward compatible).

        Active devices must represent the user's distribution (sum ~100%).
        GPU weights map directly to ``--tensor-split``.
        """
        total_layers = max(1, total_layers)
        active_gpus = self.active_gpus_with_weight(gpu_weights)
        gpu_pct = self.sum_active_weight(gpu_weights, "gpu")
        cpu_pct = self.sum_active_weight(gpu_weights, "cpu")

        # Valve OFF: LoadDistributor forces GPU-only layers.
        if cpu_enabled is False:
            return self._compute_offload_plan_with_lu(
                gpu_weights, total_layers, False, active_gpus
            )

        # Valve ON with VRAM estimate: LoadDistributor spill-over policy.
        if cpu_enabled is True:
            model_vram_mb = getattr(self, "_cached_model_vram_mb", 0) or 0
            if model_vram_mb > 0:
                return self._compute_offload_plan_with_lu(
                    gpu_weights, total_layers, True, active_gpus
                )
            # Sem estimativa de VRAM — respeita pesos da UI (mesmo caminho abaixo).

        # Backward-compatible path: use weight proportions directly
        if self.cpu_offload_active(gpu_weights):
            n_gpu_layers = max(
                0, min(total_layers, int(round(gpu_pct / 100.0 * total_layers)))
            )
            n_cpu_layers = max(0, total_layers - n_gpu_layers)
        elif active_gpus:
            n_gpu_layers = ALL_GPU_LAYERS
            n_cpu_layers = 0
        else:
            n_gpu_layers = 0
            n_cpu_layers = total_layers

        return OffloadPlan(
            n_gpu_layers=n_gpu_layers,
            n_cpu_layers=n_cpu_layers,
            gpu_pct=gpu_pct,
            cpu_pct=cpu_pct,
            tensor_split=self.compute_tensor_split(gpu_weights),
        )

    @staticmethod
    def _build_priority_order(gpu_weights: List[GPUWeight]) -> List[int]:
        """GPU indices in priority order: main first, then by ascending index.

        Implements the cascade priority of ADR-001. Considers only active GPU
        entries with weight > 0.
        """
        gpu_entries = [
            w for w in gpu_weights
            if w.device == "gpu" and w.active and w.weight > 0
        ]
        ordered = sorted(int(w.index) for w in gpu_entries)
        main_idx = next((int(w.index) for w in gpu_entries if w.is_main), None)
        if main_idx is not None and main_idx in ordered:
            return [main_idx] + [i for i in ordered if i != main_idx]
        return ordered

    def _compute_offload_plan_with_lu(
        self,
        gpu_weights: List[GPUWeight],
        total_layers: int,
        cpu_enabled: bool,
        active_gpus: List[GPUWeight],
    ) -> OffloadPlan:
        """Compute the offload plan via the strict priority-fill cascade.

        Used when ``cpu_enabled`` is explicit (True/False). Delegates the
        distribution to :class:`LoadDistributor` (single source of truth) and
        maps the result to an :class:`OffloadPlan`. See ADR-003.
        """
        gpu_pct = self.sum_active_weight(gpu_weights, "gpu")
        cpu_pct = self.sum_active_weight(gpu_weights, "cpu")

        # Build GPU weight dict
        gpu_weight_dict = {}
        for w in gpu_weights:
            if w.device == "gpu" and w.active and w.weight > 0:
                gpu_weight_dict[int(w.index)] = int(round(w.weight))

        # Get total VRAM (MB) per GPU from live metrics. get_metrics() emits
        # ``mem_total`` (MiB, from nvidia-smi); accept legacy keys as fallback.
        metrics = self.get_metrics()
        vram_by_index = {}
        for gpu in metrics.get("gpus", []):
            try:
                idx = int(gpu.get("index"))
            except (TypeError, ValueError):
                continue
            raw = (
                gpu.get("mem_total")
                or gpu.get("vram_total_mb")
                or gpu.get("vram")
                or 0
            )
            try:
                vram_by_index[idx] = int(float(raw))
            except (TypeError, ValueError):
                vram_by_index[idx] = 0
        vram_dict = {
            int(w.index): vram_by_index.get(int(w.index), 0)
            for w in gpu_weights
            if w.device == "gpu"
        }

        model_vram_mb = 0
        if hasattr(self, '_cached_model_vram_mb') and self._cached_model_vram_mb:
            model_vram_mb = self._cached_model_vram_mb

        # Priority order for the strict cascade: main GPU first, then the
        # remaining active GPUs by ascending index (ADR-001).
        priority_order = self._build_priority_order(gpu_weights)

        result = LoadDistributor.distribute(
            gpu_vram=vram_dict,
            priority_order=priority_order,
            estimated_model_vram_mb=model_vram_mb,
            cpu_enabled=cpu_enabled,
        )

        if result.is_feasible and not cpu_enabled:
            n_gpu_layers = ALL_GPU_LAYERS
            n_cpu_layers = 0
            final_gpu_pct = float(sum(gpu_weight_dict.values()) or 100)
            final_cpu_pct = 0.0
        elif result.is_feasible and cpu_enabled:
            n_gpu_layers = LoadDistributor.compute_n_gpu_layers(
                total_layers, result.total_gpu_pct
            )
            n_cpu_layers = total_layers - n_gpu_layers
            final_gpu_pct = float(result.total_gpu_pct)
            final_cpu_pct = float(result.cpu_weight)
        else:
            n_gpu_layers = ALL_GPU_LAYERS if active_gpus else 0
            n_cpu_layers = 0 if active_gpus else total_layers
            final_gpu_pct = float(sum(gpu_weight_dict.values()) or 100)
            final_cpu_pct = 0.0

        return OffloadPlan(
            n_gpu_layers=n_gpu_layers,
            n_cpu_layers=n_cpu_layers,
            gpu_pct=final_gpu_pct,
            cpu_pct=final_cpu_pct,
            tensor_split=self.compute_tensor_split(gpu_weights),
            is_feasible=result.is_feasible,
        )

    def get_visible_devices(self, gpu_weights: List[GPUWeight]) -> Optional[str]:
        active = self.active_gpus_with_weight(gpu_weights)
        if not active:
            return None
        return ",".join(str(w.index) for w in active)

    def resolve_main_gpu_index(
        self, gpu_weights: List[GPUWeight]
    ) -> str:
        """Index of main GPU within the active GPU list (for ``--main-gpu``)."""
        active_gpus = self.active_gpus_with_weight(gpu_weights)
        if not active_gpus:
            return "0"
        main_gpu_obj = next((w for w in active_gpus if w.is_main), None)
        if not main_gpu_obj:
            return "0"
        for i, w in enumerate(active_gpus):
            if w.index == main_gpu_obj.index:
                return str(i)
        return "0"

    def validate_gpu_weights(self, gpu_weights: List[GPUWeight]) -> Tuple[bool, str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0 and w.device == "gpu"]
        if not active:
            return False, "No active GPUs selected. Enable at least one GPU with weight > 0."

        has_active_cpu = any(w.active and w.device == "cpu" for w in gpu_weights)
        if not has_active_cpu:
            gpu_total = self.sum_active_weight(gpu_weights, "gpu")
            if abs(gpu_total - 100.0) > 1.0:
                return False, (
                    f"Pesos das GPUs ativas somam {gpu_total:.1f}% (esperado ~100%). "
                    "Ajuste os pesos para somar 100% ou ative a CPU para offload."
                )
        return True, ""

    def compute_n_gpu_layers(
        self, gpu_weights: List[GPUWeight], total_layers: int = 32
    ) -> int:
        """Return GPU layer count derived from :meth:`compute_offload_plan`."""
        return self.compute_offload_plan(gpu_weights, total_layers).n_gpu_layers

    def detect_model_layers(self, model_path: str) -> int:
        """Detect the number of transformer layers in a GGUF model file.

        Uses ``llama-server --model-info`` to read ``n_layer`` from the model metadata.
        Falls back to :data:`DEFAULT_TOTAL_LAYERS` when detection fails.

        :param model_path: path to the GGUF model file.
        :returns: integer number of layers.
        """
        try:
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = ""
            bin_path = _llama_server_cmd()
            output = subprocess.check_output(
                [bin_path, "--model-info", model_path],
                env=env, timeout=15, stderr=subprocess.STDOUT,
            ).decode(errors="replace")
            match = re.search(r"n_layer\s*=\s*(\d+)", output)
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.warning(f"Could not detect model layers for {model_path}: {exc}")
        return DEFAULT_TOTAL_LAYERS

    def detect_model_mtp(self, model_path: str) -> bool:
        """Return True when GGUF metadata declares MTP heads (nextn_predict_layers > 0)."""
        try:
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = ""
            bin_path = _llama_server_cmd()
            output = subprocess.check_output(
                [bin_path, "--model-info", model_path],
                env=env, timeout=15, stderr=subprocess.STDOUT,
            ).decode(errors="replace")
            match = re.search(r"nextn_predict_layers\s*=\s*(\d+)", output)
            if match:
                return int(match.group(1)) > 0
        except Exception as exc:
            logger.warning(f"Could not detect MTP for {model_path}: {exc}")
        return False
