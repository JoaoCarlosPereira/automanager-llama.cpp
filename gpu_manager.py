"""GPU detection and tensor split management."""
import os
import platform
import re
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

from schemas import GPUWeight

# Default layer count used when model metadata cannot be read.
DEFAULT_TOTAL_LAYERS = 32

LLAMA_SERVER_BIN = "llama-server"
logger = logging.getLogger("automanager")


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

    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs using llama-server --help first, fallback to nvidia-smi."""
        try:
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            output = subprocess.check_output(
                f"{LLAMA_SERVER_BIN} --help 2>&1",
                shell=True, env=env, timeout=10,
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
        try:
            output = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=10,
            ).decode()
            gpus = []
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
            vm = psutil.virtual_memory()
            cpu = self.detect_cpu_info()
            return {
                "cpu": psutil.cpu_percent(interval=0.1),
                "cpu_name": cpu.name,
                "ram": vm.percent,
                "ram_total_mb": cpu.ram_total_mb,
                "ram_used_mb": cpu.ram_used_mb,
                "gpus": gpus,
            }
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {
                "cpu": 0,
                "cpu_name": "Unknown CPU",
                "ram": 0,
                "ram_total_mb": 0,
                "ram_used_mb": 0,
                "gpus": [],
            }

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

        # --- RAM (psutil, cross-platform) ---
        vm = psutil.virtual_memory()
        ram_total_mb = round(vm.total / (1024 * 1024))
        ram_used_mb = round(vm.used / (1024 * 1024))

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

    def compute_tensor_split(self, gpu_weights: List[GPUWeight]) -> List[str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0 and w.device == "gpu"]
        if not active:
            return []
        total = sum(w.weight for w in active) or 1.0
        return [f"{w.weight / total:.4f}" for w in active]

    def get_visible_devices(self, gpu_weights: List[GPUWeight]) -> Optional[str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0 and w.device == "gpu"]
        if not active:
            return None
        return ",".join(str(w.index) for w in active)

    def validate_gpu_weights(self, gpu_weights: List[GPUWeight]) -> Tuple[bool, str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0 and w.device == "gpu"]
        if not active:
            return False, "No active GPUs selected. Enable at least one GPU with weight > 0."
        return True, ""

    def compute_n_gpu_layers(
        self, gpu_weights: List[GPUWeight], total_layers: int = 32
    ) -> int:
        """Return the number of layers to offload to GPU based on weight sum.

        The sum of all active ``device="gpu"`` weights represents the fraction
        of the model to place on GPU.  The remainder falls to CPU.

        Examples::

            >>> weights = [GPUWeight(index=0, weight=70, name="A", device="gpu"),
            ...            GPUWeight(index=0, weight=30, name="C", device="cpu")]
            >>> mgr.compute_n_gpu_layers(weights, total_layers=32)  # 70% → 22
            22

        :param gpu_weights: list of GPUWeight (may include CPU entries).
        :param total_layers: total number of transformer layers in the model.
        :returns: integer number of layers, clamped to ``[0, total_layers]``.
        """
        gpu_pct = sum(
            w.weight for w in gpu_weights if w.active and w.device == "gpu"
        )
        return max(0, min(total_layers, int(round(gpu_pct / 100.0 * total_layers))))

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
            output = subprocess.check_output(
                f'{LLAMA_SERVER_BIN} --model-info "{model_path}" 2>&1',
                shell=True, env=env, timeout=15,
            ).decode(errors="replace")
            match = re.search(r"n_layer\s*=\s*(\d+)", output)
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.warning(f"Could not detect model layers for {model_path}: {exc}")
        return DEFAULT_TOTAL_LAYERS

    def validate_weights(self, gpu_weights: List[GPUWeight]) -> Tuple[bool, str]:
        """Validate that active device weights form a valid offload plan.

        Validation rules:

        1. **Sum == 100 %** (±1 % tolerance) across all active devices
           (GPU + CPU).
        2. **CPU weight <= 70 %** — the PRD flags high CPU offload as a
           performance risk (risk table entry about ``cpu_usage`` degrading
           interactive inference).

        :returns: ``(ok, error_message)`` — ``ok`` is ``True`` when all
                  rules pass; otherwise ``error_message`` explains the first
                  failing rule.
        """
        active = [w for w in gpu_weights if w.active]
        if not active:
            return False, "Nenhum dispositivo ativo selecionado."

        has_active_gpu = any(w.device == "gpu" for w in active)
        if not has_active_gpu:
            return False, (
                "Selecione pelo menos uma GPU ativa. "
                "Offload apenas em CPU não é suportado."
            )

        total = sum(w.weight for w in active)
        if abs(total - 100.0) > 1.0:
            return False, (
                f"Pesos ativos somam {total:.1f}% (esperado ~100%). "
                "Ajuste os pesos para somar 100%."
            )

        cpu_weight = sum(
            w.weight for w in active if w.device == "cpu"
        )
        if cpu_weight > 70.0:
            return False, (
                f"O peso da CPU ({cpu_weight:.1f}%) excede o limite máximo "
                "de 70%. Reduza o peso da CPU para manter performance "
                "aceitável de inferência."
            )

        return True, ""
