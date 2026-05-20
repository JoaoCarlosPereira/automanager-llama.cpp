"""GPU detection and tensor split management."""
import os
import re
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

from schemas import GPUWeight

LLAMA_SERVER_BIN = "llama-server"
logger = logging.getLogger("automanager")


@dataclass
class GPUInfo:
    index: int
    name: str
    vram: int


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
        """Get real-time hardware metrics."""
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
            return {
                "cpu": psutil.cpu_percent(interval=0.1),
                "ram": psutil.virtual_memory().percent,
                "gpus": gpus,
            }
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {"cpu": 0, "ram": 0, "gpus": []}



class GPUManager(GPUDetector):
    """GPU operations including strict tensor split enforcement."""

    def compute_tensor_split(self, gpu_weights: List[GPUWeight]) -> List[str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0]
        if not active:
            return []
        total = sum(w.weight for w in active) or 1.0
        return [f"{w.weight / total:.4f}" for w in active]

    def get_visible_devices(self, gpu_weights: List[GPUWeight]) -> Optional[str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0]
        if not active:
            return None
        return ",".join(str(w.index) for w in active)

    def validate_gpu_weights(self, gpu_weights: List[GPUWeight]) -> Tuple[bool, str]:
        active = [w for w in gpu_weights if w.active and w.weight > 0]
        if not active:
            return False, "No active GPUs selected. Enable at least one GPU with weight > 0."
        return True, ""
