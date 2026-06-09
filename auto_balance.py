"""Progressive GPU weight discovery and VRAM maximization for auto-balance."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas import GPUWeight

logger = logging.getLogger("automanager")

READY_PATTERNS = re.compile(
    r"(?i)(listening on|http server listening|server listening|"
    r"model loaded|loaded model|main loop)"
)
OOM_PATTERNS = re.compile(
    r"(?i)(out of memory|cuda error|malloc failed|c10\.Error)"
)

MAIN_REDUCTION_STEP = 5
MIN_MAIN_WEIGHT = 10
MIN_GPU_WEIGHT = 0
PROBE_TIMEOUT_SEC = 180
POLL_INTERVAL_SEC = 1.0
VRAM_SETTLE_SEC = 4.0
TARGET_VRAM_PCT = 93.0
DEVICE_BUDGET_TOTAL = 100
DEVICE_BUDGET_TOLERANCE = 1
FAILURE_HARDWARE_CAPACITY = "hardware_capacity_exceeded"

# CPU offload constants
CPU_OFFLOAD_STEP = 10  # CPU escalates in 10% increments
DEFAULT_N_GPU_LAYERS = 99  # Default llama-server --ngl value
# No MAX_CPU_WEIGHT_PCT — CPU uses whatever is needed as spill-over


class AutoBalanceCancelled(Exception):
    """Raised when the user cancels an in-progress auto-balance run."""


class AutoBalancePlanner:
    """Builds and adjusts weight maps (main-first spill, sum=100)."""

    @staticmethod
    def spill_order(main_index: int, active_indices: List[int]) -> List[int]:
        ordered = [main_index]
        for idx in sorted(active_indices):
            if idx != main_index:
                ordered.append(idx)
        return ordered

    @staticmethod
    def weights_for_active_count(
        spill_order: List[int],
        vram_by_index: Dict[int, int],
        active_count: int,
    ) -> Dict[int, int]:
        subset = spill_order[:active_count]
        total_vram = sum(max(1, vram_by_index.get(i, 1)) for i in subset)
        weights: Dict[int, int] = {}
        remaining = 100
        for pos, idx in enumerate(subset):
            if pos == len(subset) - 1:
                weights[idx] = remaining
            else:
                vram = max(1, vram_by_index.get(idx, 1))
                share = int(round(100 * vram / total_vram))
                share = max(1, min(share, remaining - (len(subset) - pos - 1)))
                weights[idx] = share
                remaining -= share
        return weights

    @staticmethod
    def reduce_main_weight(
        weights: Dict[int, int],
        spill_order: List[int],
        vram_by_index: Dict[int, int],
    ) -> Optional[Dict[int, int]]:
        main_idx = spill_order[0]
        main_w = weights.get(main_idx, 0)
        if main_w <= MIN_MAIN_WEIGHT:
            return None

        new_weights = dict(weights)
        freed = min(MAIN_REDUCTION_STEP, main_w - MIN_MAIN_WEIGHT)
        new_weights[main_idx] = main_w - freed

        others = [i for i in spill_order if i in new_weights and i != main_idx]
        if not others:
            return None

        total_vram = sum(max(1, vram_by_index.get(i, 1)) for i in others)
        distributed = 0
        for pos, idx in enumerate(others):
            if pos == len(others) - 1:
                add = freed - distributed
            else:
                vram = max(1, vram_by_index.get(idx, 1))
                add = int(round(freed * vram / total_vram))
                add = max(0, min(add, freed - distributed))
            new_weights[idx] = new_weights.get(idx, 0) + add
            distributed += add
        return new_weights

    @staticmethod
    def active_subset(weight_map: Dict[int, int]) -> List[int]:
        return [idx for idx, w in weight_map.items() if w > 0]

    @staticmethod
    def _gpu_only(weights: List[GPUWeight]) -> List[GPUWeight]:
        return [w for w in weights if w.device == "gpu"]

    @staticmethod
    def pinned_map_from_request(gpu_weights: List[GPUWeight]) -> Dict[int, int]:
        return {
            int(w.index): int(w.weight)
            for w in gpu_weights
            if w.active and w.pinned and w.device == "gpu"
        }

    @staticmethod
    def scale_weight_map(
        weight_map: Dict[int, int], target_total: int
    ) -> Dict[int, int]:
        """Scale GPU weights proportionally so they sum to ``target_total``."""
        if target_total <= 0:
            return {idx: 0 for idx in weight_map}
        current = sum(weight_map.values())
        if current <= 0:
            return dict(weight_map)
        if current == target_total:
            return dict(weight_map)

        scaled: Dict[int, int] = {}
        remaining = target_total
        indices = sorted(weight_map.keys())
        for pos, idx in enumerate(indices):
            if pos == len(indices) - 1:
                scaled[idx] = remaining
            else:
                share = int(round(weight_map[idx] * target_total / current))
                share = max(0, min(share, remaining - (len(indices) - pos - 1)))
                scaled[idx] = share
                remaining -= share
        if indices and sum(scaled.values()) != target_total:
            scaled[indices[0]] += target_total - sum(scaled.values())
        return scaled

    @staticmethod
    def distribute_unpinned(
        pinned_map: Dict[int, int],
        unpinned_indices: List[int],
        vram_by_index: Dict[int, int],
        spill_order: List[int],
        target_total: int = 100,
    ) -> Optional[Dict[int, int]]:
        """Keep pinned weights; split remainder across unpinned GPUs."""
        pinned_total = sum(pinned_map.values())
        if pinned_total > target_total:
            return None
        if not unpinned_indices:
            return dict(pinned_map) if pinned_total == target_total else None

        remainder = target_total - pinned_total
        if len(unpinned_indices) == 1:
            return {**pinned_map, unpinned_indices[0]: remainder}

        result = dict(pinned_map)
        total_vram = sum(max(1, vram_by_index.get(i, 1)) for i in unpinned_indices)
        ordered = sorted(unpinned_indices, key=lambda i: spill_order.index(i))
        left = remainder
        for pos, idx in enumerate(ordered):
            if pos == len(ordered) - 1:
                result[idx] = left
            else:
                vram = max(1, vram_by_index.get(idx, 1))
                share = int(round(remainder * vram / total_vram))
                share = max(0, min(share, left))
                result[idx] = share
                left -= share
        if sum(result.values()) != target_total:
            result[ordered[-1]] += target_total - sum(result.values())
        return result

    @staticmethod
    def apply_pins(
        weight_map: Dict[int, int],
        pinned_map: Dict[int, int],
        spill_order: List[int],
        active_indices: List[int],
        vram_by_index: Dict[int, int],
        target_total: int = 100,
    ) -> Optional[Dict[int, int]]:
        if not pinned_map:
            total = sum(weight_map.values())
            if total == target_total:
                return dict(weight_map)
            if total <= 0:
                return None
            return AutoBalancePlanner.scale_weight_map(weight_map, target_total)
        unpinned = [i for i in active_indices if i not in pinned_map]
        merged = dict(weight_map)
        merged.update(pinned_map)
        return AutoBalancePlanner.distribute_unpinned(
            pinned_map, unpinned, vram_by_index, spill_order, target_total
        )

    @staticmethod
    def max_weight_for_gpu(
        weight_map: Dict[int, int],
        spill_order: List[int],
        target_idx: int,
        pinned_map: Optional[Dict[int, int]] = None,
        target_total: int = 100,
    ) -> int:
        """Upper bound for target weight while keeping spill-priority donors."""
        pinned_map = pinned_map or {}
        if target_idx in pinned_map:
            return pinned_map[target_idx]

        active = AutoBalancePlanner.active_subset(weight_map)
        if target_idx not in active:
            return 0

        pinned_total = sum(pinned_map.get(i, 0) for i in active if i in pinned_map)
        locked_before = sum(
            weight_map.get(i, 0)
            for i in active
            if i != target_idx
            and i not in pinned_map
            and spill_order.index(i) < spill_order.index(target_idx)
        )
        return max(
            weight_map.get(target_idx, 0),
            target_total - pinned_total - locked_before,
        )

    @staticmethod
    def set_target_weight(
        weight_map: Dict[int, int],
        spill_order: List[int],
        target_idx: int,
        new_weight: int,
        pinned_map: Optional[Dict[int, int]] = None,
        target_total: int = 100,
    ) -> Optional[Dict[int, int]]:
        """
        Set target GPU weight; take slack from later spill GPUs first (down to 0).
        GPUs earlier in spill order than target are not reduced.
        Pinned GPUs are never modified.
        """
        pinned_map = pinned_map or {}
        if target_idx in pinned_map:
            return None

        active = AutoBalancePlanner.active_subset(weight_map)
        if target_idx not in active:
            return None

        new_weight = max(MIN_GPU_WEIGHT, min(100, new_weight))
        trial = {i: weight_map.get(i, 0) for i in active}
        delta = new_weight - trial[target_idx]
        trial[target_idx] = new_weight

        if delta == 0:
            return trial if sum(trial.values()) == target_total else None

        if delta > 0:
            donors = sorted(
                [i for i in active if i != target_idx],
                key=lambda i: spill_order.index(i),
                reverse=True,
            )
            need = delta
            for donor in donors:
                if donor in pinned_map:
                    continue
                if spill_order.index(donor) <= spill_order.index(target_idx):
                    continue
                take = min(trial[donor], need)
                trial[donor] -= take
                need -= take
                if need == 0:
                    break
            if need > 0:
                return None
        else:
            receivers = sorted(
                [i for i in active if i != target_idx],
                key=lambda i: spill_order.index(i),
                reverse=True,
            )
            give = -delta
            for recv in receivers:
                if recv in pinned_map:
                    continue
                if spill_order.index(recv) <= spill_order.index(target_idx):
                    continue
                trial[recv] += give
                give = 0
                break
            if give > 0:
                return None

        if sum(trial.values()) != target_total:
            return None
        return trial

    @staticmethod
    def format_weights(weight_map: Dict[int, int], spill_order: List[int]) -> str:
        parts = [
            f"GPU{idx}={weight_map.get(idx, 0)}%"
            for idx in spill_order
            if weight_map.get(idx, 0) > 0
        ]
        return ", ".join(parts)

    @staticmethod
    def validate_device_budget(
        gpu_map: Dict[int, int],
        cpu_weight: int,
        *,
        tolerance: int = DEVICE_BUDGET_TOLERANCE,
    ) -> Tuple[bool, str]:
        """Return whether GPU + CPU weights sum to ``DEVICE_BUDGET_TOTAL``."""
        gpu_sum = sum(gpu_map.values())
        total = gpu_sum + int(cpu_weight)
        if abs(total - DEVICE_BUDGET_TOTAL) <= tolerance:
            return True, ""
        return (
            False,
            f"Pesos somam {total}% (GPU {gpu_sum}% + CPU {cpu_weight}%; "
            f"esperado {DEVICE_BUDGET_TOTAL}%).",
        )

    @staticmethod
    def enforce_device_budget(
        gpu_map: Dict[int, int],
        cpu_config: Dict[str, Any],
    ) -> Tuple[Dict[int, int], int]:
        """Normalize GPU/CPU weights so the active budget sums to exactly 100%."""
        if not cpu_config.get("enabled"):
            active = AutoBalancePlanner.active_subset(gpu_map)
            if not active:
                return dict(gpu_map), 0
            scaled = AutoBalancePlanner.scale_weight_map(
                {idx: gpu_map[idx] for idx in active},
                DEVICE_BUDGET_TOTAL,
            )
            merged = dict(gpu_map)
            merged.update(scaled)
            for idx in merged:
                if idx not in active:
                    merged[idx] = 0
            return merged, 0

        if cpu_config.get("pinned"):
            pinned_w = int(cpu_config["weight"])
            gpu_target = max(0, DEVICE_BUDGET_TOTAL - pinned_w)
            scaled = AutoBalancePlanner.scale_weight_map(gpu_map, gpu_target)
            return scaled, pinned_w

        gpu_sum = sum(gpu_map.values())
        cpu_weight = max(0, DEVICE_BUDGET_TOTAL - gpu_sum)
        if gpu_sum + cpu_weight > DEVICE_BUDGET_TOTAL:
            gpu_target = DEVICE_BUDGET_TOTAL - cpu_weight
            gpu_map = AutoBalancePlanner.scale_weight_map(gpu_map, gpu_target)
        return dict(gpu_map), cpu_weight

    @staticmethod
    def validate_device_budget_from_weights(
        weights: List[GPUWeight],
        *,
        tolerance: float = DEVICE_BUDGET_TOLERANCE,
    ) -> Tuple[bool, str]:
        """Validate an exported ``GPUWeight`` list sums to 100% on active devices."""
        active = [w for w in weights if w.active]
        total = sum(w.weight for w in active)
        if abs(total - DEVICE_BUDGET_TOTAL) <= tolerance:
            return True, ""
        return (
            False,
            f"Pesos ativos somam {total:.1f}% (esperado {DEVICE_BUDGET_TOTAL}%).",
        )

    @staticmethod
    def format_weights_with_cpu(
        gpu_weight_map: Dict[int, int],
        spill_order: List[int],
        cpu_weight: int,
    ) -> str:
        """Format GPU weights string with optional CPU offload display."""
        gpu_parts = [
            f"GPU{idx}={gpu_weight_map.get(idx, 0)}%"
            for idx in spill_order
            if gpu_weight_map.get(idx, 0) > 0
        ]
        if cpu_weight > 0:
            gpu_parts.append(f"CPU={cpu_weight}%")
        return ", ".join(gpu_parts) if gpu_parts else "CPU=100%"

    @staticmethod
    def compute_cpu_offload_weights(
        gpu_weight_map: Dict[int, int],
        total_vram_mb: int,
        estimated_model_vram_mb: int,
    ) -> Tuple[Dict[int, int], int]:
        """Compute GPU and CPU weights for model offloading.

        Strategy:
        1. If VRAM >= model size, distribute 100% across GPUs (no CPU).
        2. If VRAM < model size, scale GPU weights down proportionally,
           assign remainder to CPU (no cap — CPU uses what's needed).

        Args:
            gpu_weight_map: Original GPU weight distribution (sum=100).
            total_vram_mb: Total VRAM across all active GPUs.
            estimated_model_vram_mb: Estimated VRAM needed for the model.

        Returns:
            Tuple of (gpu_weight_map, cpu_weight). GPU weights are scaled
            down proportionally if CPU offload is needed.
        """
        if estimated_model_vram_mb <= 0:
            return dict(gpu_weight_map), 0

        if total_vram_mb >= estimated_model_vram_mb:
            return dict(gpu_weight_map), 0

        gpu_fraction = total_vram_mb / estimated_model_vram_mb
        # No cap on gpu_fraction — let it be whatever the ratio is

        cpu_weight = 100 - int(round(gpu_fraction * 100))
        # No cap on cpu_weight — CPU uses what's needed
        gpu_total = 100 - cpu_weight

        original_total = sum(gpu_weight_map.values()) or 1
        scaled_map: Dict[int, int] = {}
        remaining = gpu_total
        gpu_indices = sorted(gpu_weight_map.keys())
        for pos, idx in enumerate(gpu_indices):
            if pos == len(gpu_indices) - 1:
                scaled_map[idx] = remaining
            else:
                share = int(round(gpu_weight_map[idx] * gpu_total / original_total))
                share = max(0, min(share, remaining - (len(gpu_indices) - pos - 1)))
                scaled_map[idx] = share
                remaining -= share

        if sum(scaled_map.values()) != gpu_total:
            diff = gpu_total - sum(scaled_map.values())
            if gpu_indices:
                scaled_map[gpu_indices[0]] += diff

        return scaled_map, cpu_weight

    @staticmethod
    def estimate_model_vram_mb(
        model_path: str, context_size: int, parallel_slots: int
    ) -> int:
        """Estimate model VRAM requirements in MB.

        Uses a heuristic based on model filename pattern and context size.
        Returns estimated MB needed to load the model in GPU memory.

        Heuristics:
        - Filename contains parameter count (e.g., "7b", "13b", "70b")
        - Quantization affects size (Q4 ~4bpw, Q5 ~5bpw, Q8 ~8bpw)
        - Context size adds KV cache overhead
        """
        model_name = os.path.basename(model_path).lower()
        estimated_model_size_gb = 0.0

        # Extract parameter count from filename
        import re as _re
        param_match = _re.search(r'(\d+\.?\d*)\s*[bB]', model_name)
        if param_match:
            params_b = float(param_match.group(1)) * 1e9
            # Base model size in bytes (FP16 = 2 bytes per param)
            base_size_bytes = params_b * 2
            estimated_model_size_gb = base_size_bytes / (1024 ** 3)
        else:
            # Fallback: estimate based on file size patterns
            # GGUF files typically 1-100GB; assume ~7B Q4 as baseline (~4GB)
            estimated_model_size_gb = 4.0  # conservative default

        # Apply quantization factor (estimate Q4 as default for GGUF)
        # Q4_K_M ≈ 4 bits per weight → ~0.5x FP16 size
        quant_factor = 0.5
        model_size_gb = estimated_model_size_gb * quant_factor

        # Add context/KV cache overhead
        # KV cache ≈ 2 * layers * head_dim * hidden_size * ctx_slots * 2 bytes
        # Simplified: ~0.1 MB per context token per slot for typical models
        ctx_overhead_mb = context_size * parallel_slots * 0.1

        total_mb = (model_size_gb * 1024) + ctx_overhead_mb
        return int(total_mb)


class AutoBalanceProber:
    """Finds a feasible split, then maximizes VRAM use per GPU (main first)."""

    @staticmethod
    def build_hardware_capacity_failure(
        request,
        all_gpus: List[dict],
        active_indices: List[int],
        vram_by_index: Dict[int, int],
        reason: str = "no_feasible_split",
    ) -> Tuple[str, Dict[str, Any]]:
        model_name = os.path.basename(request.path)
        total_vram_mb = sum(vram_by_index.get(i, 0) for i in active_indices)
        total_gb = total_vram_mb / 1024.0
        gpu_rows = []
        for idx in active_indices:
            name = next(
                (g["name"] for g in all_gpus if g["index"] == idx),
                f"GPU {idx}",
            )
            vram_mb = vram_by_index.get(idx, 0)
            gpu_rows.append(
                {"index": idx, "name": name, "vram_mb": vram_mb}
            )

        message = (
            f'Não foi possível carregar "{model_name}" em nenhuma divisão testada '
            f"entre as GPUs selecionadas. O modelo excede a capacidade do hardware "
            f"atual (~{total_gb:.1f} GB de VRAM em {len(active_indices)} GPU(s)) "
            f"com contexto {request.context_size} e {request.parallel_slots} slot(s)."
        )

        failure: Dict[str, Any] = {
            "code": FAILURE_HARDWARE_CAPACITY,
            "reason": reason,
            "model": model_name,
            "model_path": request.path,
            "context_size": request.context_size,
            "parallel_slots": request.parallel_slots,
            "total_vram_mb": total_vram_mb,
            "total_vram_gb": round(total_gb, 2),
            "gpus": gpu_rows,
            "suggestions": [
                "Use quantização menor (ex.: Q4_K_M, Q3_K_S)",
                "Reduza o contexto por slot ou o número de slots paralelos",
                "Desmarque Fixar em GPUs ou reduza percentuais fixados",
                "Escolha um modelo menor compatível com este hardware",
            ],
        }
        return message, failure

    def __init__(self, process_manager, config_manager, gpu_manager, log_manager):
        self.process_manager = process_manager
        self.config = config_manager
        self.gpu_manager = gpu_manager
        self.log_manager = log_manager
        self.planner = AutoBalancePlanner()

    def _raise_if_cancelled(self) -> None:
        if self.process_manager.auto_balance_cancel_requested:
            raise AutoBalanceCancelled()

    @staticmethod
    def _cpu_config_from_request(request) -> Dict[str, Any]:
        """CPU checkbox = valve on/off for auto-balance spill-over. No weight pinning."""
        cpu_w = next(
            (w for w in request.gpu_weights if w.device == "cpu"),
            None,
        )
        if not cpu_w or not cpu_w.active:
            return {"enabled": False, "pinned": False, "weight": 0}
        # CPU weight is calculated dynamically by LoadDistributor — no fixed weight
        return {"enabled": True, "pinned": False, "weight": 0}

    @staticmethod
    def _should_add_cpu(
        active_indices: List[int],
        weight_map: Dict[int, int],
        cpu_enabled: bool,
    ) -> bool:
        """CPU offload only after every *selected* GPU already carries weight."""
        if not cpu_enabled:
            return False
        for idx in active_indices:
            if weight_map.get(idx, 0) <= 0:
                return False
        return True

    @staticmethod
    def _initial_cpu_weight(cpu_config: Dict[str, Any]) -> int:
        if cpu_config.get("pinned"):
            return int(cpu_config["weight"])
        return 0

    def _algorithmic_gpu_map(
        self,
        spill_order: List[int],
        active_indices: List[int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
        active_count: int,
        cpu_weight: int,
    ) -> Optional[Dict[int, int]]:
        """VRAM-proportional GPU split; ignores manual UI percentages."""
        gpu_target = self._gpu_budget(cpu_weight)
        template_map = self.planner.weights_for_active_count(
            spill_order, vram_by_index, active_count
        )
        return self.planner.apply_pins(
            template_map,
            pinned_map,
            spill_order,
            active_indices,
            vram_by_index,
            gpu_target,
        )

    def _escalate_cpu_offload(
        self,
        weight_map: Dict[int, int],
        cpu_weight: int,
        spill_order: List[int],
        active_indices: List[int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
    ) -> Tuple[Optional[Dict[int, int]], int]:
        """Shift load from GPU to CPU in fixed steps. No upper cap — CPU uses what's needed."""
        # Reserve minimum 10% for GPUs
        max_cpu = 90
        new_cpu = min(cpu_weight + CPU_OFFLOAD_STEP, max_cpu)
        if new_cpu <= cpu_weight:
            return None, cpu_weight
        gpu_target = self._gpu_budget(new_cpu)
        template_map = self.planner.scale_weight_map(weight_map, 100)
        new_map = self.planner.apply_pins(
            template_map,
            pinned_map,
            spill_order,
            active_indices,
            vram_by_index,
            gpu_target,
        )
        return new_map, new_cpu

    @staticmethod
    def _gpu_budget(cpu_weight: int) -> int:
        return max(0, 100 - cpu_weight)

    def _effective_gpu_map(
        self, template_map: Dict[int, int], cpu_weight: int
    ) -> Dict[int, int]:
        return self.planner.scale_weight_map(
            template_map, self._gpu_budget(cpu_weight)
        )

    def _finalize_cpu_split(
        self,
        gpu_map: Dict[int, int],
        cpu_config: Dict[str, Any],
    ) -> Tuple[Dict[int, int], int]:
        """Finalize CPU split. No 70% cap — CPU uses what's needed as spill-over."""
        return self.planner.enforce_device_budget(gpu_map, cpu_config)

    def _resolve_probe_cpu_weight(
        self,
        weight_map: Dict[int, int],
        cpu_config: Dict[str, Any],
    ) -> int:
        """CPU weight for a probe trial — minimum spill-over from GPU sum."""
        _, cpu_weight = self._finalize_cpu_split(weight_map, cpu_config)
        return cpu_weight

    def _adjust_target_weight_for_maximize(
        self,
        weight_map: Dict[int, int],
        spill_order: List[int],
        target_idx: int,
        new_weight: int,
        pinned_map: Dict[int, int],
        cpu_config: Dict[str, Any],
    ) -> Optional[Dict[int, int]]:
        """Set one GPU weight for phase 2: spill from later GPUs, then reclaim CPU."""
        pinned_map = pinned_map or {}
        current_gpu_sum = sum(weight_map.values())
        current_weight = weight_map.get(target_idx, 0)
        new_weight = max(MIN_GPU_WEIGHT, min(100, int(new_weight)))
        cpu_enabled = bool(cpu_config.get("enabled")) and not cpu_config.get("pinned")

        if new_weight == current_weight:
            return dict(weight_map)

        if not cpu_enabled:
            return self.planner.set_target_weight(
                weight_map,
                spill_order,
                target_idx,
                new_weight,
                pinned_map,
                target_total=100,
            )

        if new_weight < current_weight:
            return self.planner.set_target_weight(
                weight_map,
                spill_order,
                target_idx,
                new_weight,
                pinned_map,
                target_total=current_gpu_sum,
            )

        current_cpu = self._resolve_probe_cpu_weight(weight_map, cpu_config)
        max_gpu_total = min(100, current_gpu_sum + current_cpu)
        for target_total in range(current_gpu_sum, max_gpu_total + 1):
            trial = self.planner.set_target_weight(
                weight_map,
                spill_order,
                target_idx,
                new_weight,
                pinned_map,
                target_total=target_total,
            )
            if trial is not None:
                return trial

        delta = new_weight - current_weight
        if delta > 0 and delta <= current_cpu:
            trial = dict(weight_map)
            trial[target_idx] = new_weight
            ok, _ = self.planner.validate_device_budget(
                trial, max(0, current_cpu - delta)
            )
            if ok:
                return trial
        return None

    def _max_gpu_weight_for_maximize(
        self,
        weight_map: Dict[int, int],
        spill_order: List[int],
        target_idx: int,
        pinned_map: Dict[int, int],
        cpu_config: Dict[str, Any],
    ) -> int:
        """Upper bound for binary search in phase 2 (includes reclaimable CPU budget)."""
        if bool(cpu_config.get("enabled")) and not cpu_config.get("pinned"):
            current_cpu = self._resolve_probe_cpu_weight(weight_map, cpu_config)
            gpu_budget = min(100, sum(weight_map.values()) + current_cpu)
        else:
            gpu_budget = 100
        return self.planner.max_weight_for_gpu(
            weight_map,
            spill_order,
            target_idx,
            pinned_map,
            target_total=gpu_budget,
        )

    def discover(
        self, request
    ) -> Tuple[bool, List[GPUWeight], str, Optional[Dict[str, Any]]]:
        all_gpus = self.gpu_manager.detect_gpus()
        if not all_gpus:
            return False, request.gpu_weights, "Nenhuma GPU detectada.", None

        vram_by_index = {g["index"]: g["vram"] for g in all_gpus}
        active_indices = [
            w.index
            for w in request.gpu_weights
            if w.active and w.device == "gpu"
        ]
        if not active_indices:
            return False, request.gpu_weights, "Selecione pelo menos uma GPU.", None

        main_weight = next((w for w in request.gpu_weights if w.is_main), None)
        main_index = (
            main_weight.index
            if main_weight
            else max(active_indices, key=lambda i: vram_by_index.get(i, 0))
        )
        if main_index not in active_indices:
            main_index = active_indices[0]

        spill_order = self.planner.spill_order(main_index, active_indices)
        pinned_map = self.planner.pinned_map_from_request(request.gpu_weights)
        cpu_config = self._cpu_config_from_request(request)
        initial_cpu_weight = self._initial_cpu_weight(cpu_config)
        attempt = 0
        initial_map = self._algorithmic_gpu_map(
            spill_order,
            active_indices,
            vram_by_index,
            pinned_map,
            active_count=1,
            cpu_weight=initial_cpu_weight,
        )
        if initial_map is None:
            msg, failure = self.build_hardware_capacity_failure(
                request,
                all_gpus,
                active_indices,
                vram_by_index,
                reason="pinned_weights_exceed_100",
            )
            return False, request.gpu_weights, msg, failure

        self._set_progress(
            0,
            "Iniciando auto-balance...",
            weight_map=initial_map,
            all_gpus=all_gpus,
            main_index=main_index,
            pinned_map=pinned_map,
            cpu_weight=initial_cpu_weight,
            cpu_config=cpu_config,
        )
        self._raise_if_cancelled()

        feasible, active_count, attempt, cpu_weight = self._find_feasible_split(
            request,
            all_gpus,
            main_index,
            spill_order,
            vram_by_index,
            pinned_map,
            active_indices,
            attempt,
            cpu_config,
        )
        self._raise_if_cancelled()
        if feasible is None:
            self.process_manager.stop()
            msg, failure = self.build_hardware_capacity_failure(
                request,
                all_gpus,
                active_indices,
                vram_by_index,
                reason="no_feasible_split",
            )
            return False, request.gpu_weights, msg, failure

        attempt += 1
        self._set_progress(
            attempt,
            "Fase 1 concluida — maximizando VRAM por GPU",
            weight_map=feasible,
            all_gpus=all_gpus,
            main_index=main_index,
            pinned_map=pinned_map,
            cpu_weight=cpu_weight,
            cpu_config=cpu_config,
        )

        optimized, attempt, cpu_weight = self._maximize_vram_per_gpu(
            request,
            all_gpus,
            main_index,
            spill_order,
            feasible,
            vram_by_index,
            pinned_map,
            attempt,
            cpu_config,
            cpu_weight,
        )
        self._raise_if_cancelled()

        optimized, cpu_weight = self._finalize_cpu_split(optimized, cpu_config)
        ok, budget_err = self.planner.validate_device_budget(optimized, cpu_weight)
        if not ok:
            logger.error(f"Auto-balance budget invalid after finalize: {budget_err}")
            msg, failure = self.build_hardware_capacity_failure(
                request,
                all_gpus,
                active_indices,
                vram_by_index,
                reason="invalid_device_budget",
            )
            return False, request.gpu_weights, msg, failure

        pinned_indices: Set[int] = set(pinned_map.keys())
        cpu_valve = bool(cpu_config.get("enabled"))
        gpu_weights = self.planner.to_gpu_weights(
            all_gpus,
            optimized,
            main_index,
            pinned_indices,
            cpu_weight=cpu_weight,
            cpu_pinned=bool(cpu_config.get("pinned") and cpu_weight > 0),
            cpu_valve_enabled=cpu_valve,
        )
        ok, budget_err = self.planner.validate_device_budget_from_weights(
            gpu_weights
        )
        if not ok:
            logger.error(f"Auto-balance saved weights invalid: {budget_err}")
            msg, failure = self.build_hardware_capacity_failure(
                request,
                all_gpus,
                active_indices,
                vram_by_index,
                reason="invalid_device_budget",
            )
            return False, request.gpu_weights, msg, failure
        vram_summary = self._vram_summary(optimized, spill_order)
        weight_label = self.planner.format_weights_with_cpu(
            optimized, spill_order, cpu_weight
        )
        msg = f"Balance otimizado ({weight_label}). {vram_summary}"
        logger.info(f"Auto-balance success: {msg}")
        return True, gpu_weights, msg, None

    def _find_feasible_split(
        self,
        request,
        all_gpus: List[dict],
        main_index: int,
        spill_order: List[int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
        active_indices: List[int],
        attempt: int,
        cpu_config: Dict[str, Any],
    ) -> Tuple[Optional[Dict[int, int]], int, int, int]:
        max_active = len(spill_order)
        active_count = 1
        cpu_enabled = bool(cpu_config.get("enabled"))
        cpu_pinned = bool(cpu_config.get("pinned"))
        cpu_weight = self._initial_cpu_weight(cpu_config)
        weight_map = self._algorithmic_gpu_map(
            spill_order,
            active_indices,
            vram_by_index,
            pinned_map,
            active_count,
            cpu_weight,
        )
        if weight_map is None:
            return None, active_count, attempt, cpu_weight
        gpu_target = self._gpu_budget(cpu_weight)
        max_attempts = max(max_active * 35, 90 // CPU_OFFLOAD_STEP * 5)  # max CPU = 90% (reserve 10% for GPUs)

        while attempt < max_attempts:
            self._raise_if_cancelled()
            attempt += 1
            label = self.planner.format_weights_with_cpu(
                weight_map, spill_order, cpu_weight
            )
            self._set_progress(
                attempt,
                f"Fase 1 — encaixar modelo: {label}",
                weight_map=weight_map,
                all_gpus=all_gpus,
                main_index=main_index,
                pinned_map=pinned_map,
                cpu_weight=cpu_weight,
                cpu_config=cpu_config,
            )

            outcome = self._probe_start(
                request,
                weight_map,
                main_index,
                all_gpus,
                attempt,
                cpu_weight=cpu_weight,
                cpu_config=cpu_config,
            )
            if outcome == "cancelled":
                raise AutoBalanceCancelled()
            if outcome == "ready":
                weight_map, cpu_weight = self._finalize_cpu_split(
                    weight_map, cpu_config
                )
                return dict(weight_map), active_count, attempt, cpu_weight

            if outcome == "oom" or outcome in ("timeout", "crashed"):
                if active_count < max_active:
                    active_count += 1
                    cpu_weight = self._initial_cpu_weight(cpu_config)
                    weight_map = self._algorithmic_gpu_map(
                        spill_order,
                        active_indices,
                        vram_by_index,
                        pinned_map,
                        active_count,
                        cpu_weight,
                    )
                    if weight_map is None:
                        return None, active_count, attempt, cpu_weight
                    gpu_target = self._gpu_budget(cpu_weight)
                    continue

                if spill_order[0] not in pinned_map:
                    reduced = self.planner.reduce_main_weight(
                        weight_map, spill_order, vram_by_index
                    )
                    if reduced is not None:
                        weight_map = self.planner.apply_pins(
                            reduced,
                            pinned_map,
                            spill_order,
                            active_indices,
                            vram_by_index,
                            gpu_target,
                        )
                        if weight_map is None:
                            return None, active_count, attempt, cpu_weight
                        continue

                if (
                    cpu_enabled
                    and not cpu_pinned
                    and self._should_add_cpu(
                        active_indices, weight_map, cpu_enabled
                    )
                ):
                    new_map, new_cpu = self._escalate_cpu_offload(
                        weight_map,
                        cpu_weight,
                        spill_order,
                        active_indices,
                        vram_by_index,
                        pinned_map,
                    )
                    if new_map is not None:
                        weight_map = new_map
                        cpu_weight = new_cpu
                        gpu_target = self._gpu_budget(cpu_weight)
                        continue

                return None, active_count, attempt, cpu_weight

        return None, active_count, attempt, cpu_weight

    def _maximize_vram_per_gpu(
        self,
        request,
        all_gpus: List[dict],
        main_index: int,
        spill_order: List[int],
        weight_map: Dict[int, int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
        attempt: int,
        cpu_config: Dict[str, Any],
        cpu_weight: int,
    ) -> Tuple[Dict[int, int], int, int]:
        """For each GPU in spill order, binary-search max weight until VRAM ~full."""
        optimized = dict(weight_map)
        active_ordered = [
            idx for idx in spill_order if optimized.get(idx, 0) > 0
        ]

        for target_idx in active_ordered:
            self._raise_if_cancelled()
            if target_idx in pinned_map:
                logger.info(
                    f"Auto-balance GPU {target_idx}: fixada em "
                    f"{pinned_map[target_idx]}% (ignorada na fase 2)"
                )
                continue

            lo = optimized.get(target_idx, 0)
            hi = self._max_gpu_weight_for_maximize(
                optimized, spill_order, target_idx, pinned_map, cpu_config
            )
            best = dict(optimized)
            best_vram = 0.0

            while lo <= hi:
                self._raise_if_cancelled()
                mid = (lo + hi + 1) // 2
                trial = self._adjust_target_weight_for_maximize(
                    optimized,
                    spill_order,
                    target_idx,
                    mid,
                    pinned_map,
                    cpu_config,
                )
                if trial is None:
                    hi = mid - 1
                    continue

                attempt += 1
                gpu_name = next(
                    (g["name"] for g in all_gpus if g["index"] == target_idx),
                    f"GPU{target_idx}",
                )
                probe_cpu = self._resolve_probe_cpu_weight(trial, cpu_config)
                self._set_progress(
                    attempt,
                    f"Fase 2 — maximizar {gpu_name}: testando {mid}%",
                    weight_map=trial,
                    all_gpus=all_gpus,
                    main_index=main_index,
                    pinned_map=pinned_map,
                    cpu_weight=probe_cpu,
                    cpu_config=cpu_config,
                )

                outcome = self._probe_start(
                    request,
                    trial,
                    main_index,
                    all_gpus,
                    attempt,
                    cpu_weight=probe_cpu,
                    cpu_config=cpu_config,
                )
                if outcome == "cancelled":
                    raise AutoBalanceCancelled()
                if outcome != "ready":
                    hi = mid - 1
                    continue

                time.sleep(VRAM_SETTLE_SEC)
                vram_pct = self._get_vram_pct(target_idx)
                if vram_pct >= best_vram:
                    best = trial
                    best_vram = vram_pct

                if vram_pct >= TARGET_VRAM_PCT:
                    break

                lo = mid + 1

            optimized = best
            cpu_weight = self._resolve_probe_cpu_weight(optimized, cpu_config)
            gpu_name = next(
                (g["name"] for g in all_gpus if g["index"] == target_idx),
                f"GPU{target_idx}",
            )
            attempt += 1
            self._set_progress(
                attempt,
                f"Fase 2 — {gpu_name} ajustada: {optimized.get(target_idx, 0)}% "
                f"(VRAM {best_vram:.0f}%)",
                weight_map=optimized,
                all_gpus=all_gpus,
                main_index=main_index,
                pinned_map=pinned_map,
                cpu_weight=cpu_weight,
                cpu_config=cpu_config,
            )
            logger.info(
                f"Auto-balance GPU {target_idx}: weight={optimized.get(target_idx)}% "
                f"vram={best_vram:.1f}%"
            )

        if cpu_config.get("enabled") and not cpu_config.get("pinned"):
            optimized, cpu_weight = self._finalize_cpu_split(optimized, cpu_config)

        return optimized, attempt, cpu_weight

    def _probe_start(
        self,
        request,
        weight_map: Dict[int, int],
        main_index: int,
        all_gpus: List[dict],
        attempt: int,
        *,
        cpu_weight: int = 0,
        cpu_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.process_manager.auto_balance_cancel_requested:
            return "cancelled"
        cpu_config = cpu_config or {}
        synced_map, probe_cpu = self._finalize_cpu_split(weight_map, cpu_config)
        gpu_weights = self.planner.to_gpu_weights(
            all_gpus,
            synced_map,
            main_index,
            cpu_weight=probe_cpu,
            cpu_pinned=bool(cpu_config.get("pinned") and probe_cpu > 0),
            cpu_valve_enabled=bool(cpu_config.get("enabled")),
        )
        self.process_manager.stop()
        try:
            self.process_manager.start(
                model_path=request.path,
                gpu_weights=gpu_weights,
                context_size=request.context_size,
                mmproj_path=request.mmproj_path,
                split_mode=request.split_mode,
                parallel_slots=request.parallel_slots,
                batch_size=request.batch_size,
                thinking_enabled=request.thinking_enabled,
                mtp_enabled=request.mtp_enabled,
                mtp_draft_tokens=request.mtp_draft_tokens,
                total_layers=request.total_layers,
            )
        except Exception as exc:
            logger.error(f"Auto-balance start failed: {exc}")
            return "crashed"
        return self._wait_for_outcome()

    def _get_vram_pct(self, gpu_index: int) -> float:
        metrics = self.gpu_manager.get_metrics()
        for gpu in metrics.get("gpus", []):
            if gpu.get("index") == gpu_index:
                return float(gpu.get("vram_pct", 0))
        return 0.0

    def _vram_summary(
        self, weight_map: Dict[int, int], spill_order: List[int]
    ) -> str:
        parts = []
        for idx in spill_order:
            if weight_map.get(idx, 0) <= 0:
                continue
            pct = self._get_vram_pct(idx)
            parts.append(f"GPU{idx} VRAM {pct:.0f}%")
        return "; ".join(parts)

    def _set_progress(
        self,
        attempt: int,
        message: str,
        *,
        weight_map: Optional[Dict[int, int]] = None,
        all_gpus: Optional[List[dict]] = None,
        main_index: Optional[int] = None,
        pinned_map: Optional[Dict[int, int]] = None,
        cpu_weight: int = 0,
        cpu_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = {
            "active": True,
            "failed": False,
            "message": message,
            "auto_balance": True,
            "attempt": attempt,
        }
        if weight_map is not None and all_gpus is not None and main_index is not None:
            pinned_indices = set((pinned_map or {}).keys())
            cpu_config = cpu_config or {}
            progress_map, progress_cpu = self._finalize_cpu_split(
                weight_map, cpu_config
            )
            weights = self.planner.to_gpu_weights(
                all_gpus,
                progress_map,
                main_index,
                pinned_indices,
                cpu_weight=progress_cpu,
                cpu_pinned=bool(cpu_config.get("pinned") and progress_cpu > 0),
                cpu_valve_enabled=bool(cpu_config.get("enabled")),
            )
            state["gpu_weights"] = [w.model_dump() for w in weights]
        self.process_manager.recovery_state = state

    def _wait_for_outcome(self) -> str:
        path = self.log_manager.get_server_log_path()
        deadline = time.time() + PROBE_TIMEOUT_SEC
        last_pos = 0

        while time.time() < deadline:
            if self.process_manager.auto_balance_cancel_requested:
                self.process_manager.stop()
                return "cancelled"
            if not self._process_alive():
                return "crashed"

            chunk, last_pos = self._read_log_since(path, last_pos)
            if chunk:
                if OOM_PATTERNS.search(chunk):
                    return "oom"
                if READY_PATTERNS.search(chunk):
                    return "ready"

            time.sleep(POLL_INTERVAL_SEC)

        return "timeout"

    def _process_alive(self) -> bool:
        with self.process_manager._lock:
            proc = self.process_manager._current_process
        if proc is None:
            return False
        return proc.poll() is None

    @staticmethod
    def _read_log_since(path: str, offset: int) -> Tuple[str, int]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                chunk = f.read()
                return chunk, f.tell()
        except OSError:
            return "", offset


def _to_gpu_weights(
    all_gpus: List[dict],
    weight_map: Dict[int, int],
    main_index: int,
    pinned_indices: Optional[Set[int]] = None,
    *,
    cpu_weight: int = 0,
    cpu_pinned: bool = False,
    cpu_valve_enabled: bool = False,
) -> List[GPUWeight]:
    pinned_indices = pinned_indices or set()
    result: List[GPUWeight] = []
    for gpu in all_gpus:
        idx = gpu["index"]
        weight = int(weight_map.get(idx, 0))
        active = weight > 0
        result.append(
            GPUWeight(
                index=idx,
                weight=float(weight),
                name=gpu.get("name", f"GPU {idx}"),
                active=active,
                is_main=(idx == main_index),
                pinned=(idx in pinned_indices),
                device="gpu",
            )
        )
    if cpu_valve_enabled or cpu_weight > 0:
        result.append(
            GPUWeight(
                index=-1,
                weight=float(cpu_weight),
                name="CPU",
                active=cpu_valve_enabled or cpu_weight > 0,
                is_main=False,
                pinned=cpu_pinned,
                device="cpu",
            )
        )
    return result


AutoBalancePlanner.to_gpu_weights = staticmethod(_to_gpu_weights)
