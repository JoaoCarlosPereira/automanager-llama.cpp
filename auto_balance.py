"""Progressive GPU weight discovery and VRAM maximization for auto-balance.

Priority-fill order (cláusula pétrea): main GPU -> spill GPUs in order -> CPU.
Phase 1 activates GPUs one at a time with the main at maximum share; Phase 2
raises each GPU's weight (in that order) until VRAM reaches ~95%% before
spilling to the next device. CPU offload is only attempted after all selected
GPUs are active. Model VRAM need is estimated from the on-disk GGUF size (plus
KV-cache/context overhead) to plan GPU count and CPU spill targets.
"""

from __future__ import annotations

import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil

from schemas import GPUWeight
from load_distributor import LoadDistributor
from paths import INSTALL_ROOT

logger = logging.getLogger("automanager")
SERVER_PORT = 8085

# -------------------------------------------------------------------------
# Dedicated Auto-Balance log file (logs/auto_balance.log)
# -------------------------------------------------------------------------
AUTOBALANCE_LOG_DIR = os.path.join(INSTALL_ROOT, "logs")
AUTOBALANCE_LOG_PATH = os.path.join(AUTOBALANCE_LOG_DIR, "auto_balance.log")
_ab_logger = logging.getLogger("automanager.auto_balance")
_ab_logger.setLevel(logging.INFO)
_ab_handler: Optional[RotatingFileHandler] = None


def _ensure_auto_balance_log() -> None:
    """Create the rotating file handler for auto_balance.log (idempotent)."""
    global _ab_handler
    if _ab_handler is not None:
        return
    try:
        os.makedirs(AUTOBALANCE_LOG_DIR, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        _ab_handler = RotatingFileHandler(
            AUTOBALANCE_LOG_PATH,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
        )
        _ab_handler.setFormatter(formatter)
        _ab_logger.addHandler(_ab_handler)
    except OSError:
        pass


def _auto_balance_log(msg: str, *args, level: str = "info", **kwargs) -> None:
    """Write formatted *msg* to both the main manager log AND the dedicated file."""
    _ensure_auto_balance_log()
    # Map deprecated "warn" to "warning"
    log_level = "warning" if level == "warn" else level
    fn = getattr(_ab_logger, log_level, _ab_logger.info)
    fn(msg, *args, **kwargs)
    fn = getattr(logger, log_level, logger.info)
    fn(msg, *args, **kwargs)

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
MIN_SPILL_GPU_WEIGHT = 10  # minimum slice when activating the next GPU in cascade
PROBE_TIMEOUT_SEC = 180
POLL_INTERVAL_SEC = 1.0
VRAM_SETTLE_SEC = 4.0
TARGET_VRAM_PCT = 95.0
TARGET_VRAM_PCT_MIN = 95.0
TARGET_VRAM_PCT_MAX = 99.0
FINE_TUNE_STEP = 1  # Fine-tuning Phase 3: 1% increments on main GPU
DEVICE_BUDGET_TOTAL = 100
DEVICE_BUDGET_TOLERANCE = 1
FAILURE_HARDWARE_CAPACITY = "hardware_capacity_exceeded"
FAILURE_SERVER_CRASH = "server_crashed"

# A probe that dies faster than this is almost certainly a startup crash
# (incompatible model/binary or bad launch flag), never an OOM during the
# weight allocation of a multi-GB model.
CRASH_FAST_FAIL_SEC = 20.0
# A fast crash may also be transient (e.g. the previous probe's server still
# releasing the HTTP port). Retry a few times before declaring it fatal.
CRASH_RETRY_MAX = 2
CRASH_RETRY_BACKOFF_SEC = 1.0

# CPU offload constants
CPU_OFFLOAD_STEP = 10  # CPU escalates in 10% increments
DEFAULT_N_GPU_LAYERS = 99  # Default llama-server --ngl value
# No MAX_CPU_WEIGHT_PCT — CPU uses whatever is needed as spill-over
MODEL_RUNTIME_OVERHEAD_MB = 256  # mmap/metadata headroom on top of GGUF weights
MODEL_RUNTIME_OVERHEAD_RATIO = 0.05  # 5% of file size, whichever is larger
SMART_CONTEXT_FALLBACKS = (
    524288,
    458752,
    393216,
    327680,
    278528,
    262144,
    131072,
    65536,
    32768,
    16384,
    8192,
    4096,
)


class AutoBalanceCancelled(Exception):
    """Raised when the user cancels an in-progress auto-balance run."""


class AutoBalanceServerCrashed(Exception):
    """Raised when a probe makes llama-server die WITHOUT an OOM signal.

    This is a fatal condition for auto-balance: a non-OOM crash means the model
    cannot be loaded by this binary at all (incompatible architecture, bad
    launch flag, missing dependency, ...), so escalating GPUs/CPU is pointless.
    The run aborts and the user is told it is a crash — not lack of VRAM.

    Carries enough context to build a user-facing message:
      - ``weight_map`` / ``cpu_weight``: the split that crashed.
      - ``elapsed``: seconds until the process died.
      - ``log_tail``: last lines of the llama-server log (best-effort).
    """

    def __init__(
        self,
        weight_map: Dict[int, int],
        cpu_weight: int,
        elapsed: float,
        log_tail: Optional[List[str]] = None,
    ) -> None:
        self.weight_map = dict(weight_map)
        self.cpu_weight = int(cpu_weight)
        self.elapsed = float(elapsed)
        self.log_tail = list(log_tail or [])
        super().__init__(
            f"llama-server crashed after {self.elapsed:.1f}s "
            f"(split={self.weight_map}, cpu={self.cpu_weight}%)"
        )


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
    def weights_for_cascade_spill(
        spill_order: List[int],
        active_count: int,
        vram_by_index: Optional[Dict[int, int]] = None,
        target_total: int = 100,
    ) -> Optional[Dict[int, int]]:
        """
        Priority-fill template: main (spill_order[0]) keeps the maximum share;
        each secondary GPU gets the flat minimum slice (``MIN_SPILL_GPU_WEIGHT``)
        in priority order — never a proportional split. The real per-GPU VRAM
        filling happens empirically in Fase 2 (:meth:`_maximize_vram_per_gpu`),
        which raises each GPU to its cap in spill order. Order:
        main -> GPUs by index -> CPU (handled later).

        ``vram_by_index`` is accepted for call-site compatibility but unused:
        the template no longer skews secondaries by VRAM (cláusula pétrea —
        nunca distribuir proporcionalmente entre GPUs).
        """
        if active_count <= 0 or active_count > len(spill_order):
            return None
        subset = spill_order[:active_count]
        if active_count == 1:
            return {subset[0]: target_total}

        secondaries = subset[1:]
        secondary_pool = MIN_SPILL_GPU_WEIGHT * len(secondaries)
        if secondary_pool >= target_total - MIN_MAIN_WEIGHT:
            return None
        main_share = target_total - secondary_pool
        if main_share < MIN_MAIN_WEIGHT:
            return None

        weights: Dict[int, int] = {subset[0]: main_share}
        for idx in secondaries:
            weights[idx] = MIN_SPILL_GPU_WEIGHT
        if sum(weights.values()) != target_total:
            weights[subset[-1]] += target_total - sum(weights.values())
        return weights

    @staticmethod
    def cascade_fill_weights(
        subset: List[int],
        vram_by_index: Dict[int, int],
        model_mb: int,
        target_total: int = 100,
        vram_limit_pct: float = TARGET_VRAM_PCT_MAX,
    ) -> Optional[Dict[int, int]]:
        """VRAM-cap priority fill across *subset* (spill order), as percentages.

        Mirrors :meth:`LoadDistributor.distribute`: each GPU is filled up to
        ``vram * vram_limit_pct`` MB in order; the LAST GPU of the subset
        absorbs whatever remains — even above its cap — so the probe actually
        tests that split (and OOMs, prompting Fase 1 to add the next GPU) when
        the subset cannot hold the model. The MB allocation is converted to
        integer percentages summing to *target_total* (largest-remainder).

        Returns None when *subset* is empty or *model_mb* <= 0 (caller falls
        back to the flat template).
        """
        if not subset or model_mb <= 0 or target_total <= 0:
            return None

        mb_by_idx: Dict[int, int] = {}
        remaining = model_mb
        last = len(subset) - 1
        for pos, idx in enumerate(subset):
            if pos == last:
                mb_by_idx[idx] = max(0, remaining)
                remaining = 0
            else:
                cap = int(max(0, vram_by_index.get(idx, 0)) * vram_limit_pct / 100.0)
                alloc = min(cap, remaining)
                mb_by_idx[idx] = alloc
                remaining -= alloc

        total_mb = sum(mb_by_idx.values())
        if total_mb <= 0:
            return None

        # MB -> integer % summing to target_total (largest-remainder method).
        floors: Dict[int, int] = {}
        remainders: List[Tuple[float, int]] = []
        for idx, mb in mb_by_idx.items():
            exact = mb * target_total / total_mb
            floor = int(exact)
            floors[idx] = floor
            remainders.append((exact - floor, idx))
        leftover = target_total - sum(floors.values())
        remainders.sort(reverse=True)
        for _, idx in remainders[: max(0, leftover)]:
            floors[idx] += 1
        return floors

    @staticmethod
    def gpu_weight_floors(
        spill_order: List[int],
        selected_indices: Optional[List[int]] = None,
    ) -> Dict[int, int]:
        """Minimum weight each selected GPU must keep when shifting budget to CPU."""
        if not spill_order or not selected_indices:
            return {}
        main_idx = spill_order[0]
        floors: Dict[int, int] = {}
        for idx in selected_indices:
            if idx == main_idx:
                floors[idx] = MIN_MAIN_WEIGHT
            else:
                floors[idx] = MIN_SPILL_GPU_WEIGHT
        return floors

    @staticmethod
    def gpu_weight_floors_for_maximize(
        weight_map: Dict[int, int],
        spill_order: List[int],
    ) -> Dict[int, int]:
        """Phase-2 floors: only GPUs carrying weight; secondaries may drain to 0."""
        if not spill_order:
            return {}
        main_idx = spill_order[0]
        floors: Dict[int, int] = {}
        for idx in AutoBalancePlanner.active_subset(weight_map):
            if idx == main_idx:
                floors[idx] = MIN_MAIN_WEIGHT
            else:
                floors[idx] = MIN_GPU_WEIGHT
        return floors

    @staticmethod
    def ensure_selected_gpu_floors(
        weight_map: Dict[int, int],
        spill_order: List[int],
        selected_indices: List[int],
        pinned_map: Dict[int, int],
        target_total: int = DEVICE_BUDGET_TOTAL,
    ) -> Optional[Dict[int, int]]:
        """Raise selected GPUs to their floors and rebalance to *target_total*."""
        if not spill_order or not selected_indices:
            return dict(weight_map)
        pinned_map = pinned_map or {}
        floors = AutoBalancePlanner.gpu_weight_floors(spill_order, selected_indices)
        trial = {
            idx: max(weight_map.get(idx, 0), floors.get(idx, 0))
            for idx in selected_indices
        }
        total = sum(trial.values())
        if total > target_total:
            excess = total - target_total
            for idx in reversed(spill_order):
                if idx not in trial or idx in pinned_map:
                    continue
                floor = floors.get(idx, 0)
                can_take = max(0, trial[idx] - floor)
                take = min(can_take, excess)
                trial[idx] -= take
                excess -= take
                if excess == 0:
                    break
            if excess > 0:
                return None
        elif total < target_total:
            main_idx = spill_order[0]
            deficit = target_total - total
            if main_idx in trial and main_idx not in pinned_map:
                trial[main_idx] = trial.get(main_idx, 0) + deficit
            else:
                for idx in spill_order:
                    if idx in selected_indices and idx not in pinned_map:
                        trial[idx] = trial.get(idx, 0) + deficit
                        break
        result = dict(weight_map)
        for idx in result:
            result[idx] = 0
        for idx in selected_indices:
            result[idx] = trial[idx]
        if sum(result[idx] for idx in selected_indices) != target_total:
            return None
        return result

    @staticmethod
    def shift_gpu_budget_for_cpu(
        weight_map: Dict[int, int],
        spill_order: List[int],
        pinned_map: Dict[int, int],
        gpu_target: int,
        selected_indices: Optional[List[int]] = None,
    ) -> Optional[Dict[int, int]]:
        """
        Reduce GPU budget to *gpu_target* by taking weight from later spill GPUs
        first; the main GPU (spill_order[0]) is reduced only as a last resort.
        Selected GPUs never drop below their floor (main MIN_MAIN_WEIGHT,
        secondaries MIN_SPILL_GPU_WEIGHT).
        """
        pinned_map = pinned_map or {}
        active = AutoBalancePlanner.active_subset(weight_map)
        if not active:
            return None

        floors = AutoBalancePlanner.gpu_weight_floors(spill_order, selected_indices)
        trial = {idx: weight_map.get(idx, 0) for idx in active}
        current = sum(trial.values())
        if current < gpu_target:
            return AutoBalancePlanner.scale_weight_map(trial, gpu_target)
        if current == gpu_target:
            return trial

        to_remove = current - gpu_target
        for idx in reversed(spill_order):
            if idx not in trial or trial[idx] <= 0:
                continue
            if idx in pinned_map:
                continue
            floor = floors.get(idx, 0)
            can_take = max(0, trial[idx] - floor)
            take = min(can_take, to_remove)
            trial[idx] -= take
            to_remove -= take
            if to_remove == 0:
                break

        if to_remove > 0:
            main_idx = spill_order[0]
            if main_idx in trial and main_idx not in pinned_map:
                floor = floors.get(main_idx, 0)
                can_take = max(0, trial[main_idx] - floor)
                take = min(can_take, to_remove)
                trial[main_idx] -= take
                to_remove -= take

        if to_remove > 0:
            return None
        if sum(trial.values()) != gpu_target:
            return None
        return trial

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
            if total > target_total:
                return AutoBalancePlanner.shift_gpu_budget_for_cpu(
                    weight_map, spill_order, {}, target_total, active_indices
                )
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
        weight_floors: Optional[Dict[int, int]] = None,
    ) -> Optional[Dict[int, int]]:
        """
        Set target GPU weight; take slack from later spill GPUs first (down to 0).
        GPUs earlier in spill order than target are not reduced.
        Pinned GPUs are never modified.
        """
        pinned_map = pinned_map or {}
        weight_floors = weight_floors or {}
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
                floor = weight_floors.get(donor, 0)
                take = min(max(0, trial[donor] - floor), need)
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
    def validate_cpu_not_dominant(
        gpu_map: Dict[int, int],
        cpu_weight: int,
    ) -> Tuple[bool, str]:
        """CPU weight must not exceed total GPU weight.

        Returns True when gpu_sum >= cpu_weight (valid).
        Returns False when cpu_weight > gpu_sum (CPU dominant -> rejected).
        """
        gpu_sum = sum(gpu_map.values())
        if gpu_sum == 0 and cpu_weight == 0:
            return True, ""
        if gpu_sum == 0 and cpu_weight > 0:
            return (
                False,
                f"CPU={cpu_weight}% com nenhuma GPU ativa — rejeitado.",
            )
        if cpu_weight > gpu_sum:
            return (
                False,
                f"CPU={cpu_weight}% > GPU total={gpu_sum}% — "
                "CPU não pode ultrapassar soma das GPUs.",
            )
        return True, ""

    @staticmethod
    def enforce_device_budget(
        gpu_map: Dict[int, int],
        cpu_config: Dict[str, Any],
        *,
        spill_order: Optional[List[int]] = None,
        selected_indices: Optional[List[int]] = None,
        pinned_map: Optional[Dict[int, int]] = None,
        skip_floors: bool = False,
    ) -> Tuple[Dict[int, int], int]:
        """Normalize GPU/CPU weights so the active budget sums to exactly 100%.

        When *skip_floors* is True, minimum weight floors are NOT applied.
        This is used during empirical probes so the trial weight_map is passed
        through unchanged (no floor inflation on idle GPUs).
        """
        pinned_map = pinned_map or {}
        spill_allowed = bool(cpu_config.get("cpu_spill_allowed"))

        if cpu_config.get("pinned"):
            pinned_w = int(cpu_config["weight"])
            gpu_target = max(0, DEVICE_BUDGET_TOTAL - pinned_w)
            scaled = AutoBalancePlanner.scale_weight_map(gpu_map, gpu_target)
            return scaled, pinned_w

        def _scale_gpus_to_full_budget(
            source_map: Dict[int, int],
        ) -> Tuple[Dict[int, int], int]:
            working = dict(source_map)
            if selected_indices and spill_order and not skip_floors:
                floored = AutoBalancePlanner.ensure_selected_gpu_floors(
                    working,
                    spill_order,
                    selected_indices,
                    pinned_map,
                    DEVICE_BUDGET_TOTAL,
                )
                if floored is not None:
                    working = floored
            if selected_indices and spill_order:
                active = [idx for idx in spill_order if idx in selected_indices]
            else:
                active = AutoBalancePlanner.active_subset(working)
            if not active:
                return dict(working), 0
            scaled = AutoBalancePlanner.scale_weight_map(
                {idx: working[idx] for idx in active},
                DEVICE_BUDGET_TOTAL,
            )
            merged = dict(working)
            merged.update(scaled)
            for idx in merged:
                if idx not in active:
                    merged[idx] = 0
            return merged, 0

        if not cpu_config.get("enabled") or not spill_allowed:
            return _scale_gpus_to_full_budget(gpu_map)

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
    def model_weights_mb_from_disk(model_path: str) -> Optional[int]:
        """Return GGUF file size in MB, or None if the path is missing/unreadable."""
        if not model_path:
            return None
        try:
            if not os.path.isfile(model_path):
                return None
            size_bytes = os.path.getsize(model_path)
            mb = int(size_bytes / (1024 * 1024))
            logger.debug(
                "model_weights_mb_from_disk: path=%s size_bytes=%d size_mb=%d",
                model_path,
                size_bytes,
                mb,
            )
            return mb
        except OSError as exc:
            logger.debug(
                "model_weights_mb_from_disk: FAILED path=%s error=%s",
                model_path,
                exc,
            )
            return None

    @staticmethod
    def _estimate_from_filename(model_path: str) -> float:
        """Heuristic model weight size in GB from filename (before KV cache)."""
        model_name = os.path.basename(model_path).lower()
        param_match = re.search(r"(\d+\.?\d*)\s*[bB]", model_name)
        if param_match:
            params_b = float(param_match.group(1)) * 1e9
            base_size_bytes = params_b * 2
            estimated_model_size_gb = base_size_bytes / (1024 ** 3)
        else:
            estimated_model_size_gb = 4.0
        quant_factor = 0.5
        return estimated_model_size_gb * quant_factor

    @staticmethod
    def _runtime_overhead_mb(weights_mb: int) -> int:
        return max(MODEL_RUNTIME_OVERHEAD_MB, int(weights_mb * MODEL_RUNTIME_OVERHEAD_RATIO))

    @staticmethod
    def plan_min_gpu_count(
        spill_order: List[int],
        vram_by_index: Dict[int, int],
        estimated_model_vram_mb: int,
    ) -> int:
        """
        Minimum GPUs (in spill order) whose combined VRAM at 95% fill can hold
        the estimated model size. Always at least 1 (main GPU is tried first).
        """
        if estimated_model_vram_mb <= 0 or not spill_order:
            return 1
        cumulative = 0
        for pos, idx in enumerate(spill_order):
            cumulative += max(0, vram_by_index.get(idx, 0))
            if cumulative * (TARGET_VRAM_PCT_MIN / 100.0) >= estimated_model_vram_mb:
                return pos + 1
        return len(spill_order)

    @staticmethod
    def estimate_cpu_spill_weight(
        total_gpu_vram_mb: int,
        estimated_model_vram_mb: int,
    ) -> int:
        """
        Estimated CPU weight (%%) when all GPUs are filled to TARGET_VRAM_PCT_MAX.
        Returns 0 when the model should fit entirely on GPU VRAM.
        """
        if estimated_model_vram_mb <= 0 or total_gpu_vram_mb <= 0:
            return 0
        usable_gpu_mb = total_gpu_vram_mb * (TARGET_VRAM_PCT_MAX / 100.0)
        if usable_gpu_mb >= estimated_model_vram_mb:
            return 0
        gpu_fraction = usable_gpu_mb / estimated_model_vram_mb
        cpu_weight = DEVICE_BUDGET_TOTAL - int(round(gpu_fraction * DEVICE_BUDGET_TOTAL))
        return max(0, min(90, cpu_weight))

    @staticmethod
    def align_cpu_weight_step(cpu_weight: int) -> int:
        """Round CPU weight up to the next offload step (10%%, max 90%%)."""
        if cpu_weight <= 0:
            return 0
        stepped = (
            (cpu_weight + CPU_OFFLOAD_STEP - 1) // CPU_OFFLOAD_STEP
        ) * CPU_OFFLOAD_STEP
        return min(90, stepped)

    @staticmethod
    def estimate_model_vram_mb(
        model_path: str,
        context_size: int,
        parallel_slots: int,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
    ) -> Dict[str, int]:
        """Estimate VRAM components separately: weights + KV-cache + runtime.

        Returns a dict with keys:
          - ``weights_mb``: GGUF disk size + runtime overhead (what
            ``--tensor-split`` and ``--ngl`` actually divide).
          - ``kv_cache_mb``: KV-cache overhead from context size.
          - ``total_mb``: weights + kv_cache + runtime (for total VRAM budget).

        The cascade MUST distribute only ``weights_mb``, because the llama-server
        splits weights (not KV-cache) via --tensor-split/--ngl.
        """
        # Multipliers relative to f16 (0.1 MB/token for GQA models)
        multipliers = {
            "f32": 2.0,
            "f16": 1.0,
            "bf16": 1.0,
            "q8_0": 0.5,
            "q5_1": 0.35,
            "q5_0": 0.32,
            "q4_1": 0.28,
            "q4_0": 0.25,
        }
        m_k = multipliers.get(cache_type_k, 1.0)
        m_v = multipliers.get(cache_type_v, 1.0)
        avg_mult = (m_k + m_v) / 2.0

        ctx_overhead_mb = context_size * parallel_slots * 0.1 * avg_mult

        disk_mb = AutoBalancePlanner.model_weights_mb_from_disk(model_path)
        if disk_mb is not None and disk_mb > 0:
            runtime_overhead = AutoBalancePlanner._runtime_overhead_mb(disk_mb)
            weights_mb = disk_mb + runtime_overhead
            kv_cache_mb = int(ctx_overhead_mb)
            total = weights_mb + kv_cache_mb
            _auto_balance_log(
                "estimate_model_vram_mb [DISK] model=%s disk_mb=%d "
                "cache_k=%s cache_v=%s weights_mb=%d "
                "kv_cache_mb=%d total_mb=%d",
                model_path,
                disk_mb,
                cache_type_k,
                cache_type_v,
                weights_mb,
                kv_cache_mb,
                total,
            )
            return {
                "weights_mb": weights_mb,
                "kv_cache_mb": kv_cache_mb,
                "total_mb": total,
            }

        model_size_gb = AutoBalancePlanner._estimate_from_filename(model_path)
        weights_mb = int(model_size_gb * 1024)
        runtime_overhead = AutoBalancePlanner._runtime_overhead_mb(weights_mb)
        kv_cache_mb = int(ctx_overhead_mb)
        total = weights_mb + kv_cache_mb + runtime_overhead
        _auto_balance_log(
            "estimate_model_vram_mb [FILENAME] model=%s "
            "estimated_gb=%.2f cache_k=%s cache_v=%s weights_mb=%d "
            "kv_cache_mb=%d total_mb=%d",
            model_path,
            model_size_gb,
            cache_type_k,
            cache_type_v,
            weights_mb,
            kv_cache_mb,
            total,
        )
        return {
            "weights_mb": weights_mb,
            "kv_cache_mb": kv_cache_mb,
            "total_mb": total,
        }


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
        _auto_balance_log(
            "=== HARDWARE CAPACITY FAILURE ===\n"
            "  reason=%s\n"
            "  model=%s path=%s\n"
            "  estimated_model_mb=?\n"
            "  total_vram_mb=%d total_vram_gb=%.2f\n"
            "  active_gpus=%s\n"
            "  gpu_details=%s\n"
            "  context_size=%d parallel_slots=%d\n"
            "  message=%s",
            reason,
            model_name,
            request.path,
            total_vram_mb,
            total_gb,
            active_indices,
            gpu_rows,
            request.context_size,
            request.parallel_slots,
            message,
        )
        return message, failure

    def build_server_crash_failure(
        self,
        request,
        crash: "AutoBalanceServerCrashed",
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the user-facing failure for a non-OOM server crash.

        Distinct from :meth:`build_hardware_capacity_failure`: this is NOT a
        VRAM-capacity problem. The server died without an OOM signal, so the
        model/binary/flags are incompatible and no GPU/CPU split would help.
        """
        model_name = os.path.basename(request.path) if request.path else "?"
        try:
            server_log_path = self.log_manager.get_server_log_path(self.port)
        except Exception:  # pragma: no cover - defensive
            server_log_path = "?"
        log_tail = crash.log_tail or []

        message = (
            f'O servidor encerrou inesperadamente (~{crash.elapsed:.1f}s) ao '
            f'carregar "{model_name}", sem indício de falta de memória (OOM). '
            "Isso indica incompatibilidade do modelo/binário llama.cpp ou um "
            "parâmetro de inicialização inválido — não falta de VRAM. "
            "O auto-balance foi interrompido."
        )

        failure: Dict[str, Any] = {
            "code": FAILURE_SERVER_CRASH,
            "reason": "server_crashed",
            "model": model_name,
            "model_path": request.path,
            "context_size": request.context_size,
            "parallel_slots": request.parallel_slots,
            "crashed_split": crash.weight_map,
            "crashed_cpu_weight": crash.cpu_weight,
            "elapsed_sec": round(crash.elapsed, 1),
            "server_log_path": server_log_path,
            "server_log_tail": log_tail,
            "suggestions": [
                "Verifique o log do servidor para o erro exato: "
                f"{server_log_path}",
                "Confirme que esta build do llama.cpp suporta a arquitetura do "
                "modelo (ex.: rode 'llama-server --model-info <modelo>')",
                "Atualize o llama.cpp para uma versão que suporte o modelo",
                "Revise flags de inicialização (contexto, flash-attn, "
                "reasoning, MTP, chat-template)",
            ],
        }
        _auto_balance_log(
            "=== SERVER CRASH FAILURE (não-OOM) ===\n"
            "  model=%s path=%s\n"
            "  elapsed=%.1fs crashed_split=%s cpu=%d%%\n"
            "  server_log=%s\n"
            "  log_tail=%s\n"
            "  message=%s",
            model_name,
            request.path,
            crash.elapsed,
            crash.weight_map,
            crash.cpu_weight,
            server_log_path,
            log_tail,
            message,
            level="error",
        )
        return message, failure

    def __init__(self, process_manager, config_manager, gpu_manager, log_manager):
        self.process_manager = process_manager
        self.config = config_manager
        self.gpu_manager = gpu_manager
        self.log_manager = log_manager
        self.planner = AutoBalancePlanner()
        self.port = SERVER_PORT
        # Diagnostics from the last probe that died, used to build a clear
        # crash message (vs. an OOM) for the user.
        self._last_server_log_tail: List[str] = []
        self._last_crash_elapsed: float = 0.0
        # Estimated model weight size (MB) for the current run; drives the
        # VRAM-cap cascade template in _algorithmic_gpu_map.
        self._weights_mb: int = 0

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

    def _cache_model_vram_estimate(self, request, est: Dict[str, int]) -> None:
        """Share VRAM estimate (weights_mb) with GPUManager / LoadDistributor during probes."""
        try:
            self.gpu_manager._cached_model_vram_mb = est["weights_mb"]
        except AttributeError:
            pass

    def _apply_cpu_budget(
        self,
        weight_map: Dict[int, int],
        cpu_weight: int,
        target_cpu: int,
        spill_order: List[int],
        active_indices: List[int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
    ) -> Tuple[Optional[Dict[int, int]], int]:
        """Set CPU offload to *target_cpu*, taking GPU budget from spill order."""
        target_cpu = self.planner.align_cpu_weight_step(
            max(cpu_weight, min(90, int(target_cpu)))
        )
        if target_cpu <= cpu_weight:
            return dict(weight_map), cpu_weight
        gpu_target = self._gpu_budget(target_cpu)
        shifted = self.planner.shift_gpu_budget_for_cpu(
            weight_map, spill_order, pinned_map, gpu_target, active_indices
        )
        if shifted is None:
            return None, cpu_weight
        new_map = self.planner.apply_pins(
            shifted,
            pinned_map,
            spill_order,
            active_indices,
            vram_by_index,
            gpu_target,
        )
        if new_map is None:
            return None, cpu_weight
        return new_map, target_cpu

    def _model_plan_summary(
        self,
        request,
        spill_order: List[int],
        vram_by_index: Dict[int, int],
    ) -> Tuple[int, int, Optional[int], int]:
        """Return (estimated_mb, planned_gpu_count, disk_mb, estimated_cpu).

        *estimated_mb* é o **weights_mb** (sem KV-cache), que é o que a
        cascata usa para distribuir pesos via --tensor-split.
        """
        est = self.planner.estimate_model_vram_mb(
            request.path,
            request.context_size,
            request.parallel_slots,
            cache_type_k=request.cache_type_k,
            cache_type_v=request.cache_type_v,
        )
        weights_mb = est["weights_mb"]
        total_mb = est["total_mb"]
        disk_mb = self.planner.model_weights_mb_from_disk(request.path)
        planned_gpus = self.planner.plan_min_gpu_count(
            spill_order, vram_by_index, weights_mb
        )
        total_vram = sum(vram_by_index.get(i, 0) for i in spill_order)
        est_cpu = self.planner.estimate_cpu_spill_weight(total_vram, weights_mb)
        return total_mb, planned_gpus, disk_mb, est_cpu

    def _algorithmic_gpu_map(
        self,
        spill_order: List[int],
        active_indices: List[int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
        active_count: int,
        cpu_weight: int,
    ) -> Optional[Dict[int, int]]:
        """Cascade priority-fill split; ignores manual UI percentages.

        When the model size is known, fills each GPU up to its VRAM cap in
        spill order (same policy as :class:`LoadDistributor`), so the probe
        tries the balanced split that actually fits (e.g. ~48/33/19 on a
        3090+2×P100) instead of overloading the main GPU. Falls back to the
        flat template (main max, secondaries minimum slice) when size unknown.
        """
        gpu_target = self._gpu_budget(cpu_weight)
        subset = spill_order[:active_count]
        template_map = None
        weights_mb = getattr(self, "_weights_mb", 0)
        if weights_mb and weights_mb > 0:
            template_map = self.planner.cascade_fill_weights(
                subset, vram_by_index, weights_mb, gpu_target
            )
        if template_map is None:
            template_map = self.planner.weights_for_cascade_spill(
                spill_order, active_count, vram_by_index, gpu_target
            )
        if template_map is None:
            return None
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
        """Shift load from later spill GPUs to CPU; main GPU is reduced last."""
        max_cpu = 90
        new_cpu = min(cpu_weight + CPU_OFFLOAD_STEP, max_cpu)
        if new_cpu <= cpu_weight:
            return None, cpu_weight
        gpu_target = self._gpu_budget(new_cpu)
        shifted = self.planner.shift_gpu_budget_for_cpu(
            weight_map, spill_order, pinned_map, gpu_target, active_indices
        )
        if shifted is None:
            return None, cpu_weight
        new_map = self.planner.apply_pins(
            shifted,
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

    @staticmethod
    def _phase1_gpu_attempt_label(
        active_count: int,
        spill_order: List[int],
        weight_map: Dict[int, int],
    ) -> str:
        """Human-readable label matching the Fase 1 probe table."""
        main_idx = spill_order[0]
        if active_count <= 1:
            return f"Tentativa {active_count}: Main GPU{main_idx} 100%"
        parts = [f"GPU{idx}={weight_map.get(idx, 0)}%" for idx in spill_order[:active_count]]
        return (
            f"Tentativa {active_count}: Main + {active_count - 1}ª GPU "
            f"(cascata: {', '.join(parts)})"
        )

    def _budget_selected_indices(
        self,
        weight_map: Dict[int, int],
        spill_order: Optional[List[int]],
        selected_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """GPUs that actually participate in the current weight trial.

        Unlike the UI ``active_indices`` (all checked GPUs), this list only
        includes devices with weight > 0 in *weight_map*, preserving spill
        order.  Using the full UI selection in ``enforce_device_budget`` would
        incorrectly apply MIN_SPILL floors to idle GPUs (e.g. turning a
        GPU2=100% probe into 80/10/10).
        """
        if spill_order:
            active = [idx for idx in spill_order if weight_map.get(idx, 0) > 0]
            if active:
                return active
            main_idx = spill_order[0]
            if main_idx in weight_map:
                return [main_idx]
        active = self.planner.active_subset(weight_map)
        if active:
            return active
        return list(selected_indices or [])

    def _finalize_cpu_split(
        self,
        gpu_map: Dict[int, int],
        cpu_config: Dict[str, Any],
        *,
        cpu_spill_allowed: Optional[bool] = None,
        spill_order: Optional[List[int]] = None,
        selected_indices: Optional[List[int]] = None,
        pinned_map: Optional[Dict[int, int]] = None,
        skip_floors: bool = False,
    ) -> Tuple[Dict[int, int], int]:
        """Finalize CPU split. CPU spill only when explicitly confirmed (OOM path).

        When *skip_floors* is True, minimum weight floors are NOT applied,
        used during empirical probes so the trial weight_map is passed through.
        """
        cfg = dict(cpu_config)
        if cpu_spill_allowed is not None:
            cfg["cpu_spill_allowed"] = cpu_spill_allowed
        elif "cpu_spill_allowed" not in cfg:
            cfg["cpu_spill_allowed"] = bool(cfg.get("pinned"))
        return self.planner.enforce_device_budget(
            gpu_map,
            cfg,
            spill_order=spill_order,
            selected_indices=selected_indices,
            pinned_map=pinned_map,
            skip_floors=skip_floors,
        )

    def _resolve_probe_cpu_weight(
        self,
        weight_map: Dict[int, int],
        cpu_config: Dict[str, Any],
        *,
        cpu_spill_allowed: Optional[bool] = None,
        spill_order: Optional[List[int]] = None,
        selected_indices: Optional[List[int]] = None,
        pinned_map: Optional[Dict[int, int]] = None,
    ) -> int:
        """CPU weight for a probe trial — only non-zero when spill was confirmed."""
        _, cpu_weight = self._finalize_cpu_split(
            weight_map,
            cpu_config,
            cpu_spill_allowed=cpu_spill_allowed,
            spill_order=spill_order,
            selected_indices=selected_indices,
            pinned_map=pinned_map,
        )
        return cpu_weight

    def _adjust_target_weight_for_maximize(
        self,
        weight_map: Dict[int, int],
        spill_order: List[int],
        target_idx: int,
        new_weight: int,
        pinned_map: Dict[int, int],
        cpu_config: Dict[str, Any],
        *,
        selected_indices: Optional[List[int]] = None,
    ) -> Optional[Dict[int, int]]:
        """Set one GPU weight for phase 2: spill from later GPUs, then reclaim CPU."""
        pinned_map = pinned_map or {}
        current_gpu_sum = sum(weight_map.values())
        current_weight = weight_map.get(target_idx, 0)
        new_weight = max(MIN_GPU_WEIGHT, min(100, int(new_weight)))
        floors = self.planner.gpu_weight_floors_for_maximize(
            weight_map, spill_order
        )
        cpu_spill_allowed = bool(cpu_config.get("cpu_spill_allowed"))
        cpu_enabled = (
            bool(cpu_config.get("enabled"))
            and not cpu_config.get("pinned")
            and cpu_spill_allowed
        )

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
                weight_floors=floors,
            )

        if new_weight < current_weight:
            return self.planner.set_target_weight(
                weight_map,
                spill_order,
                target_idx,
                new_weight,
                pinned_map,
                target_total=current_gpu_sum,
                weight_floors=floors,
            )

        current_cpu = self._resolve_probe_cpu_weight(
            weight_map,
            cpu_config,
            cpu_spill_allowed=True,
            spill_order=spill_order,
            selected_indices=selected_indices,
            pinned_map=pinned_map,
        )
        max_gpu_total = min(100, current_gpu_sum + current_cpu)
        for target_total in range(current_gpu_sum, max_gpu_total + 1):
            trial = self.planner.set_target_weight(
                weight_map,
                spill_order,
                target_idx,
                new_weight,
                pinned_map,
                target_total=target_total,
                weight_floors=floors,
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
        *,
        selected_indices: Optional[List[int]] = None,
    ) -> int:
        """Upper bound for binary search in phase 2 (includes reclaimable CPU budget)."""
        cpu_spill_allowed = bool(cpu_config.get("cpu_spill_allowed"))
        if (
            bool(cpu_config.get("enabled"))
            and not cpu_config.get("pinned")
            and cpu_spill_allowed
        ):
            current_cpu = self._resolve_probe_cpu_weight(
                weight_map,
                cpu_config,
                cpu_spill_allowed=True,
                spill_order=spill_order,
                selected_indices=selected_indices,
                pinned_map=pinned_map,
            )
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
        """Auto-balance empírico com sondagem de OOM (ADR-002 atualizado)."""
        self.port = getattr(request, "port", None) or SERVER_PORT
        self._model_path = request.path
        self._smart_calibration = bool(getattr(request, "smart_calibration", False))
        try:
            calibrated_request = request
            success, weights, message, failure = self._discover_empirical(
                calibrated_request
            )

            # Smart calibration may change every field that is not pinned. If
            # the exact request does not fit, retry safer cache/context choices
            # before declaring the hardware incapable. The first successful
            # candidate is the largest context actually proven by a real probe.
            if (
                not success
                and getattr(request, "smart_calibration", False)
                and failure
                and failure.get("code") == FAILURE_HARDWARE_CAPACITY
            ):
                for candidate in self._smart_fallback_requests(request):
                    self._raise_if_cancelled()
                    _auto_balance_log(
                        "SMART FALLBACK: tentando contexto=%d cache=%s/%s",
                        candidate.context_size,
                        candidate.cache_type_k,
                        candidate.cache_type_v,
                        level="warn",
                    )
                    success, weights, message, failure = self._discover_empirical(
                        candidate
                    )
                    if success:
                        calibrated_request = candidate
                        message = (
                            f"{message} Configuração Smart viável em contexto "
                            f"{candidate.context_size} com cache "
                            f"{candidate.cache_type_k}/{candidate.cache_type_v}."
                        )
                        break

            result_data = failure
            if success and getattr(request, "smart_calibration", False):
                result_data = failure or {}
                proposal = self._generate_smart_proposal(
                    calibrated_request, weights
                )
                # Context/cache changes must be values that the empirical
                # calibration actually loaded, not an untested heuristic bump.
                proposal["context_size"] = calibrated_request.context_size
                proposal["cache_type_k"] = calibrated_request.cache_type_k
                proposal["cache_type_v"] = calibrated_request.cache_type_v
                result_data["proposal"] = proposal

            return success, weights, message, result_data
        except AutoBalanceCancelled:
            return False, request.gpu_weights, "Auto-balance cancelado pelo usuário.", None
        except AutoBalanceServerCrashed as crash:
            self.process_manager.stop(self.port)
            msg, failure = self.build_server_crash_failure(request, crash)
            return False, request.gpu_weights, msg, failure

    @staticmethod
    def _smart_fallback_requests(request) -> List[Any]:
        """Safer unpinned configurations, ordered by expected performance."""
        pinned = request.pinned_fields or {}
        candidates: List[Any] = []
        seen: Set[Tuple[int, str, str]] = set()

        def add(context_size: int, cache_k: str, cache_v: str) -> None:
            key = (context_size, cache_k, cache_v)
            original = (
                request.context_size,
                request.cache_type_k,
                request.cache_type_v,
            )
            if key == original or key in seen:
                return
            seen.add(key)
            candidate = request.model_copy(deep=True)
            candidate.context_size = context_size
            candidate.cache_type_k = cache_k
            candidate.cache_type_v = cache_v
            candidates.append(candidate)

        cache_mutable = not pinned.get("cache_type")
        context_mutable = not pinned.get("context_size")
        safe_k = "q4_0" if cache_mutable else request.cache_type_k
        safe_v = "q4_0" if cache_mutable else request.cache_type_v

        # Quantizing KV at the requested context preserves the user's desired
        # context and is therefore preferable to reducing the context window.
        if cache_mutable:
            add(request.context_size, safe_k, safe_v)

        if context_mutable:
            for context_size in SMART_CONTEXT_FALLBACKS:
                if context_size < request.context_size:
                    add(context_size, safe_k, safe_v)

        return candidates

    def _generate_smart_proposal(self, request, final_weights: List[GPUWeight]) -> dict:
        """Heurística para sugerir a melhor performance baseada na VRAM sobrando."""
        all_gpus = self.gpu_manager.detect_gpus()
        vram_map = {g["index"]: g["vram"] for g in all_gpus}
        # VRAM disponível no início da calibração (desconta outras instâncias);
        # não relemos aqui porque o modelo sondado ainda está carregado.
        available_map = getattr(self, "_initial_available_vram", None) or {}

        # 1. Calcular VRAM total disponível nas GPUs ativas
        active_indices = [w.index for w in final_weights if w.active and w.device == "gpu"]
        total_vram_mb = sum(
            available_map.get(idx, vram_map.get(idx, 0)) for idx in active_indices
        )
        
        # 2. Estimar uso atual
        est = self.planner.estimate_model_vram_mb(
            request.path,
            request.context_size,
            request.parallel_slots,
            cache_type_k=request.cache_type_k,
            cache_type_v=request.cache_type_v
        )
        weights_mb = est["weights_mb"]
        kv_mb = est["kv_cache_mb"]
        total_used = weights_mb + kv_mb
        
        free_mb = total_vram_mb - total_used
        pinned = request.pinned_fields or {}
        
        proposal = {
            "context_size": request.context_size,
            "parallel_slots": request.parallel_slots,
            "batch_size": request.batch_size,
            "ubatch_size": request.ubatch_size,
            "cache_type_k": request.cache_type_k,
            "cache_type_v": request.cache_type_v,
            "threads": request.threads,
            "threads_batch": request.threads_batch,
            "thinking_enabled": request.thinking_enabled,
            "mtp_enabled": request.mtp_enabled,
            "mtp_draft_tokens": request.mtp_draft_tokens,
            "mtp_model_path": request.mtp_model_path,
            "numa_enabled": request.numa_enabled,
            "flash_attn_enabled": request.flash_attn_enabled,
            "split_mode": request.split_mode,
        }

        # Heurística de Cache
        if not pinned.get("cache_type"):
            if free_mb < 500 and request.cache_type_k == "f16":
                proposal["cache_type_k"] = "q4_0"
                proposal["cache_type_v"] = "q4_0"
            elif free_mb > 4000 and request.cache_type_k != "f16":
                proposal["cache_type_k"] = "f16"
                proposal["cache_type_v"] = "f16"

        # Heurística de Batch
        if not pinned.get("batch_size"):
            if free_mb > 2048:
                proposal["batch_size"] = 4096
            elif free_mb < 512:
                proposal["batch_size"] = 1024

        # Heurística de U-Batch
        if not pinned.get("ubatch_size"):
            # Usually 512 is safe and fast for modern GPUs
            proposal["ubatch_size"] = 512

        # Heurística de Threads (usar núcleos físicos detectados)
        if not pinned.get("threads"):
            cpu_info = self.gpu_manager.detect_cpu_info()
            if cpu_info.physical_cores > 0:
                proposal["threads"] = cpu_info.physical_cores
                proposal["threads_batch"] = cpu_info.physical_cores

        # Heurística de Contexto (Aumentar se sobrar MUITA VRAM)
        if not pinned.get("context_size") and free_mb > 8000:
            proposal["context_size"] = request.context_size * 2

        return proposal

    def _available_vram_by_index(
        self, vram_total_by_index: Dict[int, int]
    ) -> Dict[int, int]:
        """VRAM disponível por GPU (total - uso atual do driver).

        Em multi-instância outras instâncias llama-server permanecem
        carregadas durante a calibração; os caps da cascata devem considerar
        apenas a VRAM livre — como ocorria quando a sondagem partia de GPUs
        vazias. Sem métricas, retorna o total (comportamento antigo).
        """
        available = dict(vram_total_by_index)
        try:
            metrics = self.gpu_manager.get_metrics()
            gpu_entries = list(metrics.get("gpus", []) or [])
        except Exception:
            return available
        for gpu in gpu_entries:
            try:
                idx = int(gpu.get("index"))
                used = int(float(gpu.get("mem_used", 0) or 0))
            except (TypeError, ValueError, AttributeError):
                continue
            if idx in available and used > 0:
                available[idx] = max(0, available[idx] - used)
        return available

    def _discover_empirical(
        self, request
    ) -> Tuple[bool, List[GPUWeight], str, Optional[Dict[str, Any]]]:
        """Empírico: sondagem iterativa por OOM com mesma priorização do fonte."""
        all_gpus = self.gpu_manager.detect_gpus()
        if not all_gpus:
            return False, request.gpu_weights, "Nenhuma GPU detectada.", None

        vram_total_by_index = {g["index"]: g["vram"] for g in all_gpus}
        vram_by_index = self._available_vram_by_index(vram_total_by_index)
        # Guarda a disponibilidade no início da calibração: a proposta smart é
        # gerada com o modelo de sondagem ainda carregado e não pode reler.
        self._initial_available_vram = dict(vram_by_index)
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
        cpu_enabled = bool(cpu_config.get("enabled"))
        estimated_mb, planned_gpus, disk_mb, est_cpu = self._model_plan_summary(
            request, spill_order, vram_by_index
        )
        est = self.planner.estimate_model_vram_mb(
            request.path,
            request.context_size,
            request.parallel_slots,
            cache_type_k=request.cache_type_k,
            cache_type_v=request.cache_type_v,
        )
        self._cache_model_vram_estimate(request, est)
        weights_mb = est["weights_mb"]
        kv_cache_mb = est["kv_cache_mb"]
        total_mb = est["total_mb"]
        # Drives the VRAM-cap cascade template in _algorithmic_gpu_map.
        self._weights_mb = weights_mb
        gpu_names = {g["index"]: g.get("name", f"GPU{g['index']}") for g in all_gpus}
        model_name = os.path.basename(request.path) if request.path else "?"
        total_active_vram = sum(vram_by_index.get(i, 0) for i in active_indices)
        _auto_balance_log(
            "=== Auto-balance EMPIRICAL START ===\n"
            "  model=%s path=%s\n"
            "  context_size=%d parallel_slots=%d\n"
            "  weights_mb=%d (%.2f GB)  kv_cache_mb=%d  total_mb=%d (%.2f GB)\n"
            "  active_gpus=%d | vram_disponivel=%d MB (%.2f GB)\n"
            "  gpu_vram_disponivel={%s} (total={%s})\n"
            "  gpu_names={%s}\n"
            "  main_index=%d\n"
            "  spill_order=%s\n"
            "  cpu_enabled=%s\n"
            "  pins_count=%d",
            model_name,
            request.path,
            request.context_size,
            request.parallel_slots,
            weights_mb,
            weights_mb / 1024.0,
            kv_cache_mb,
            total_mb,
            total_mb / 1024.0,
            len(active_indices),
            total_active_vram,
            total_active_vram / 1024.0,
            ", ".join(f"{i}: {vram_by_index.get(i, 0)}" for i in active_indices),
            ", ".join(f"{i}: {vram_total_by_index.get(i, 0)}" for i in active_indices),
            ", ".join(f"{i}: {gpu_names.get(i, '?')}" for i in active_indices),
            main_index,
            spill_order,
            cpu_enabled,
            len([w for w in request.gpu_weights if w.pinned]),
        )
        initial_cpu_weight = self._initial_cpu_weight(cpu_config)
        # GPU-only config: CPU desabilitado explicitamente para Fases 1, 2 e 3
        gpu_only_cpu_config = {
            "enabled": False,
            "pinned": False,
            "weight": 0,
        }
        attempt = 0
        plan_hint = (
            f"~{estimated_mb} MB"
            + (f" (arquivo {disk_mb} MB)" if disk_mb else "")
            + f", previsto ≥{planned_gpus} GPU(s)"
            + (f", CPU ~{est_cpu}%" if est_cpu and cpu_enabled else "")
        )
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
            f"Iniciando auto-balance ({plan_hint})...",
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
            estimated_model_mb=estimated_mb,
            allow_cpu=False,
        )
        self._raise_if_cancelled()
        if feasible is None:
            _auto_balance_log(
                "Fase 1 — cascata GPU-only deu OOM; buscando split GPU-only "
                "maximizando o principal (resto balanceado por VRAM) antes do CPU",
                level="warn",
            )
            feasible, attempt, cpu_weight = self._try_full_gpu_maximize_before_cpu(
                request,
                all_gpus,
                main_index,
                spill_order,
                vram_by_index,
                pinned_map,
                active_indices,
                attempt,
            )
        self._raise_if_cancelled()
        if feasible is None and cpu_enabled and not cpu_config.get("pinned"):
            _auto_balance_log(
                "Fase 1 — maximização GPU-only falhou; iniciando CPU offload "
                "(último recurso, +%d%% por tentativa)",
                CPU_OFFLOAD_STEP,
                level="warn",
            )
            feasible, attempt, cpu_weight = self._escalate_cpu_until_feasible(
                request,
                all_gpus,
                main_index,
                spill_order,
                vram_by_index,
                pinned_map,
                active_indices,
                attempt,
                cpu_config,
                estimated_model_mb=estimated_mb,
            )
        self._raise_if_cancelled()
        if feasible is None:
            self.process_manager.stop(self.port)
            msg, failure = self.build_hardware_capacity_failure(
                request,
                all_gpus,
                active_indices,
                vram_by_index,
                reason="no_feasible_split",
            )
            return False, request.gpu_weights, msg, failure

        weight_label = self.planner.format_weights_with_cpu(
            feasible, spill_order, cpu_weight
        )
        _auto_balance_log(
            "=== Auto-balance: Fase 1 viavel ===\n"
            "  weight_label=%s\n"
            "  weight_map={%s}\n"
            "  cpu_weight=%d",
            weight_label,
            ", ".join(f"{i}: {w}%" for i, w in feasible.items()),
            cpu_weight,
        )

        attempt += 1
        self._set_progress(
            attempt,
            "Fase 1 concluida — removendo GPUs desnecessarias",
            weight_map=feasible,
            all_gpus=all_gpus,
            main_index=main_index,
            pinned_map=pinned_map,
            cpu_weight=cpu_weight,
            cpu_config=cpu_config,
        )

        trimmed, attempt, cpu_weight = self._trim_trailing_spill_gpus(
            request,
            all_gpus,
            main_index,
            spill_order,
            feasible,
            pinned_map,
            active_indices,
            attempt,
            cpu_config,
            cpu_weight,
        )
        self._raise_if_cancelled()

        attempt += 1
        self._set_progress(
            attempt,
            "Fase 1b concluida — maximizando VRAM por GPU",
            weight_map=trimmed,
            all_gpus=all_gpus,
            main_index=main_index,
            pinned_map=pinned_map,
            cpu_weight=cpu_weight,
            cpu_config=cpu_config,
        )

        # Se a Fase 1 precisou de CPU, a Fase 2 pode tentar recuperar esse
        # orçamento para as GPUs, mas deve preservar o último split READY caso
        # a tentativa GPU-only dê OOM.
        _auto_balance_log(
            "=== Fase 2 PREPARANDO ===\n"
            "  trimmed={%s} main_index=%d cpu_enabled=%s",
            ", ".join(f"{i}: {w}%" for i, w in trimmed.items()),
            main_index,
            cpu_enabled,
        )
        phase2_cpu_config = (
            {**cpu_config, "cpu_spill_allowed": True}
            if cpu_weight > 0
            else {
                "enabled": False,
                "pinned": False,
                "weight": 0,
                "cpu_spill_allowed": False,
            }
        )
        _auto_balance_log(
            "=== Fase 2 INICIO (maximizar VRAM%s) ===",
            ", preservando CPU comprovada" if cpu_weight > 0 else ", GPU-only",
        )
        optimized, attempt, cpu_weight = self._maximize_vram_per_gpu(
            request,
            all_gpus,
            main_index,
            spill_order,
            trimmed,
            vram_by_index,
            pinned_map,
            attempt,
            phase2_cpu_config,
            cpu_weight,
            active_indices=active_indices,
        )
        _auto_balance_log(
            "=== Fase 2 CONCLUIDA ===\n"
            "  weight_map={%s}\n"
            "  cpu_weight=%d",
            ", ".join(f"{i}: {w}%" for i, w in optimized.items()),
            cpu_weight,
        )
        self._raise_if_cancelled()

        # === Fase 3 — fine-tuning: sobe main GPU 1% por 1% até OOM ===
        attempt += 1
        self._set_progress(
            attempt,
            "Fase 3 iniciada — fine-tuning 1% na main GPU",
            weight_map=optimized,
            all_gpus=all_gpus,
            main_index=main_index,
            pinned_map=pinned_map,
            cpu_weight=cpu_weight,
            cpu_config=cpu_config,
        )

        if cpu_weight > 0:
            _auto_balance_log(
                "Fase 3: CPU offload %d%% faz parte do último split READY — "
                "pulando fine-tuning GPU-only",
                cpu_weight,
            )
        else:
            optimized, attempt = self._fine_tune_main_gpu_weight(
                request,
                all_gpus,
                main_index,
                spill_order,
                optimized,
                vram_by_index,
                pinned_map,
                active_indices,
                attempt,
                cpu_config,
                cpu_weight,
            )
        self._raise_if_cancelled()

        budget_selected = self._budget_selected_indices(
            optimized, spill_order, active_indices
        )
        optimized, cpu_weight = self._finalize_cpu_split(
            optimized,
            cpu_config,
            cpu_spill_allowed=(cpu_weight > 0),
            spill_order=spill_order,
            selected_indices=budget_selected,
            pinned_map=pinned_map,
            skip_floors=True,
        )
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

        ok, cpu_err = self.planner.validate_cpu_not_dominant(optimized, cpu_weight)
        if not ok:
            logger.error(f"Auto-balance CPU dominant rejected: {cpu_err}")
            msg, failure = self.build_hardware_capacity_failure(
                request,
                all_gpus,
                active_indices,
                vram_by_index,
                reason="cpu_dominant",
            )
            return False, request.gpu_weights, msg, failure

        pinned_indices: Set[int] = set(pinned_map.keys())
        # Final result: deactivate any device that ends at 0%. GPUs at 0% are
        # already inactive (to_gpu_weights sets active=weight>0); the CPU valve
        # must likewise be unchecked when it carries no load.
        cpu_active = bool(cpu_config.get("enabled")) and cpu_weight > 0
        gpu_weights = self.planner.to_gpu_weights(
            all_gpus,
            optimized,
            main_index,
            pinned_indices,
            cpu_weight=cpu_weight,
            cpu_pinned=bool(cpu_config.get("pinned") and cpu_weight > 0),
            cpu_valve_enabled=cpu_active,
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
        _auto_balance_log(
            "=== Auto-balance: peso final ===\n"
            "  weight_label=%s\n"
            "  weight_map={%s}\n"
            "  cpu_weight=%d",
            weight_label,
            ", ".join(f"{i}: {w}%" for i, w in optimized.items()),
            cpu_weight,
        )

        before = self._capture_hardware_snapshot("BEFORE_FINAL")
        _auto_balance_log(
            "=== HARDWARE SNAPSHOT BEFORE FINAL ===\n"
            "  gpus=[%s]",
            "; ".join(
                f"GPU{g['index']}: used={g['mem_used_mb']:.0f}MB "
                f"total={g['mem_total_mb']:.0f}MB util={g['util_pct']:.0f}%"
                for g in before["gpus"]
            ),
        )

        msg = f"Balance otimizado ({weight_label}). {vram_summary}"
        _auto_balance_log(
            "=== Auto-balance SUCCESS ===\n"
            "  message=%s\n"
            "  gpu_weights_exported=%d entries",
            msg,
            len(gpu_weights),
        )
        return True, gpu_weights, msg, None

    def _try_full_gpu_maximize_before_cpu(
        self,
        request,
        all_gpus: List[dict],
        main_index: int,
        spill_order: List[int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
        active_indices: List[int],
        attempt: int,
    ) -> Tuple[Optional[Dict[int, int]], int, int]:
        """Find the largest main-GPU weight that loads GPU-only (no CPU).

        The strict cap cascade fills the main GPU using only the **weights**
        size and ignores the KV-cache/compute buffers that also live on each
        GPU (proportionally to its layers), so it overloads the main and OOMs.
        Here we binary-search the main weight downward from its cap share; for
        each candidate the remainder is split across the other active GPUs by
        VRAM (balanced). The largest main weight that probes READY wins —
        honouring "main has priority" (it stays as high as physically fits)
        without trusting the (unreliable) VRAM estimate.

        Returns (feasible_map | None, attempt, 0).
        """
        others = [i for i in spill_order[1:] if i in active_indices]
        unpinned_others = [i for i in others if i not in pinned_map]
        if not others or main_index in pinned_map or not unpinned_others:
            return None, attempt, 0

        gpu_only_cpu = {"enabled": False, "pinned": False, "weight": 0}
        subset = [main_index] + others

        # hi = cap-based main share (what the cascade tried); lo = VRAM-
        # proportional main share (balanced) — the natural floor below which
        # the secondaries would themselves overload.
        cap_map = self.planner.cascade_fill_weights(
            subset, vram_by_index, self._weights_mb or 0, DEVICE_BUDGET_TOTAL
        )
        hi = (cap_map or {}).get(main_index, 0)
        total_vram = sum(max(1, vram_by_index.get(i, 1)) for i in subset)
        proportional_main = int(round(
            DEVICE_BUDGET_TOTAL
            * max(1, vram_by_index.get(main_index, 1)) / total_vram
        ))
        lo = max(MIN_MAIN_WEIGHT, min(proportional_main, hi or proportional_main))
        hi = max(lo, hi)

        _auto_balance_log(
            "=== Fase 1c — busca GPU-only (maximizar principal) ===\n"
            "  principal=GPU%d faixa=[%d%%, %d%%] resto por VRAM em %s",
            main_index,
            lo,
            hi,
            others,
        )

        best: Optional[Dict[int, int]] = None
        while lo <= hi:
            self._raise_if_cancelled()
            mid = (lo + hi + 1) // 2
            trial = self.planner.distribute_unpinned(
                {**pinned_map, main_index: mid},
                unpinned_others,
                vram_by_index,
                spill_order,
                DEVICE_BUDGET_TOTAL,
            )
            if trial is None:
                hi = mid - 1
                continue
            attempt += 1
            label = self.planner.format_weights(trial, spill_order)
            self._set_progress(
                attempt,
                f"Fase 1c — GPU-only: principal {mid}% ({label})",
                weight_map=trial,
                all_gpus=all_gpus,
                main_index=main_index,
                pinned_map=pinned_map,
                cpu_weight=0,
                cpu_config=gpu_only_cpu,
            )
            outcome = self._probe_start(
                request,
                trial,
                main_index,
                all_gpus,
                attempt,
                cpu_weight=0,
                cpu_config=gpu_only_cpu,
                spill_order=spill_order,
                selected_indices=active_indices,
                pinned_map=pinned_map,
            )
            if outcome == "ready":
                best = dict(trial)
                _auto_balance_log("Fase 1c — principal %d%% → READY (%s)", mid, label)
                lo = mid + 1  # main fits — try to give it even more
            else:
                _auto_balance_log(
                    "Fase 1c — principal %d%% → %s; reduzindo principal",
                    mid,
                    outcome,
                    level="warn",
                )
                hi = mid - 1

        if best is not None:
            logger.info(
                "Auto-balance: split GPU-only viável (principal maximizado): %s",
                self.planner.format_weights(best, spill_order),
            )
            return best, attempt, 0
        return None, attempt, 0

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
        *,
        estimated_model_mb: int = 0,
        allow_cpu: bool = True,
    ) -> Tuple[Optional[Dict[int, int]], int, int, int]:
        """Fase 1 — GPUs primeiro, CPU como último recurso (probe iterativo).

        Tentativa 1: main 100% → OOM → +1 GPU (cascata)
        Tentativa 2..N: main + spill GPUs (pesos cascata) → OOM → +1 GPU
        Após esgotar GPUs: se *allow_cpu*, CPU +10% por vez; senão retorna None
        (caller tenta maximizar GPUs e depois :meth:`_escalate_cpu_until_feasible`).
        """
        max_active = len(spill_order)
        active_count = 1
        cpu_enabled = bool(cpu_config.get("enabled"))
        cpu_pinned = bool(cpu_config.get("pinned"))
        cpu_weight = self._initial_cpu_weight(cpu_config)
        planned_gpus = self.planner.plan_min_gpu_count(
            spill_order, vram_by_index, estimated_model_mb
        )
        _auto_balance_log(
            "=== Fase 1 INICIO (GPUs primeiro%s) ===",
            ", CPU permitido neste passo" if allow_cpu and cpu_enabled else "",
        )
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
        max_attempts = max(max_active * 35, 90 // CPU_OFFLOAD_STEP * 5)

        while attempt < max_attempts:
            self._raise_if_cancelled()
            attempt += 1
            label = self.planner.format_weights_with_cpu(
                weight_map, spill_order, cpu_weight
            )
            attempt_label = self._phase1_gpu_attempt_label(
                active_count, spill_order, weight_map
            )
            _auto_balance_log("Fase 1 — %s", attempt_label)
            phase_hint = ""
            if estimated_model_mb > 0 and active_count < planned_gpus:
                phase_hint = f" (modelo ~{estimated_model_mb} MB, meta {planned_gpus} GPU(s))"
            self._set_progress(
                attempt,
                f"Fase 1 — {attempt_label}{phase_hint}",
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
                spill_order=spill_order,
                selected_indices=active_indices,
                pinned_map=pinned_map,
            )
            if outcome == "cancelled":
                raise AutoBalanceCancelled()
            if outcome == "ready":
                budget_selected = self._budget_selected_indices(
                    weight_map, spill_order, active_indices
                )
                weight_map, cpu_weight = self._finalize_cpu_split(
                    weight_map,
                    cpu_config,
                    cpu_spill_allowed=(cpu_weight > 0),
                    spill_order=spill_order,
                    selected_indices=budget_selected,
                    pinned_map=pinned_map,
                )
                ok, err = self.planner.validate_cpu_not_dominant(
                    weight_map, cpu_weight
                )
                if not ok:
                    _auto_balance_log(
                        "_find_feasible_split: REJECTED cpu_dominant — %s",
                        err,
                        level="warn",
                    )
                    # Treat as OOM to force new configuration
                    outcome = "oom"
                else:
                    _auto_balance_log(
                        "Fase 1 — %s → READY",
                        attempt_label,
                    )
                    return dict(weight_map), active_count, attempt, cpu_weight

            if outcome == "oom" or outcome in ("timeout", "crashed"):
                if active_count < max_active:
                    _auto_balance_log(
                        "Fase 1 — %s → %s; adicionar %dª GPU",
                        attempt_label,
                        outcome.upper(),
                        active_count + 1,
                        level="warn",
                    )
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
                    continue

                if allow_cpu and cpu_enabled and not cpu_pinned:
                    _auto_balance_log(
                        "Fase 1 — todas as GPUs esgotadas → CPU offload +%d%%",
                        CPU_OFFLOAD_STEP,
                        level="warn",
                    )
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
                        continue

                return None, active_count, attempt, cpu_weight

        return None, active_count, attempt, cpu_weight

    def _escalate_cpu_until_feasible(
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
        *,
        estimated_model_mb: int = 0,
    ) -> Tuple[Optional[Dict[int, int]], int, int]:
        """Fase 1 último recurso: todas GPUs ativas + CPU offload em +10%."""
        max_active = len(spill_order)
        cpu_weight = self._initial_cpu_weight(cpu_config)
        _auto_balance_log(
            "=== Fase 1 CPU (último recurso) — %d GPU(s) ativas, CPU +%d%% por tentativa ===",
            max_active,
            CPU_OFFLOAD_STEP,
        )
        weight_map = self._algorithmic_gpu_map(
            spill_order,
            active_indices,
            vram_by_index,
            pinned_map,
            max_active,
            cpu_weight,
        )
        if weight_map is None:
            return None, attempt, cpu_weight

        # Primeira tentativa CPU já com +10% (não repetir probe GPU-only).
        weight_map, cpu_weight = self._escalate_cpu_offload(
            weight_map,
            cpu_weight,
            spill_order,
            active_indices,
            vram_by_index,
            pinned_map,
        )
        if weight_map is None:
            return None, attempt, cpu_weight

        est = self.planner.estimate_model_vram_mb(
                request.path,
                request.context_size,
                request.parallel_slots,
                cache_type_k=request.cache_type_k,
                cache_type_v=request.cache_type_v,
            ) if estimated_model_mb <= 0 else {"total_mb": estimated_model_mb}
        total_mb = est.get("total_mb", 0)
        max_attempts = max(90 // CPU_OFFLOAD_STEP + 5, 20)
        while attempt < max_attempts and cpu_weight < 90:
            self._raise_if_cancelled()
            attempt += 1
            label = self.planner.format_weights_with_cpu(
                weight_map, spill_order, cpu_weight
            )
            cpu_hint = (
                f" (~{total_mb} MB no disco+ctx)"
                if total_mb > 0
                else ""
            )
            cpu_attempt = max(1, cpu_weight // CPU_OFFLOAD_STEP)
            _auto_balance_log(
                "Fase 1 — Tentativa CPU %d: CPU offload %d%% | %s",
                cpu_attempt,
                cpu_weight,
                label,
            )
            self._set_progress(
                attempt,
                f"Fase 1 — CPU offload{cpu_hint}: {label}",
                weight_map=weight_map,
                all_gpus=all_gpus,
                main_index=main_index,
                pinned_map=pinned_map,
                cpu_weight=cpu_weight,
                cpu_config=cpu_config,
            )
            cpu_probe_config = {
                **cpu_config,
                "cpu_spill_allowed": cpu_weight > 0,
            }
            outcome = self._probe_start(
                request,
                weight_map,
                main_index,
                all_gpus,
                attempt,
                cpu_weight=cpu_weight,
                cpu_config=cpu_probe_config,
                spill_order=spill_order,
                selected_indices=active_indices,
                pinned_map=pinned_map,
            )
            if outcome == "cancelled":
                raise AutoBalanceCancelled()
            if outcome == "ready":
                budget_selected = self._budget_selected_indices(
                    weight_map, spill_order, active_indices
                )
                weight_map, cpu_weight = self._finalize_cpu_split(
                    weight_map,
                    cpu_config,
                    cpu_spill_allowed=(cpu_weight > 0),
                    spill_order=spill_order,
                    selected_indices=budget_selected,
                    pinned_map=pinned_map,
                )
                ok, err = self.planner.validate_cpu_not_dominant(
                    weight_map, cpu_weight
                )
                if not ok:
                    _auto_balance_log(
                        "_escalate_cpu_until_feasible: REJECTED cpu_dominant — %s",
                        err,
                        level="warn",
                    )
                    # Treat as OOM to force CPU escalation
                    outcome = "oom"
                else:
                    _auto_balance_log(
                        "Fase 1 — Tentativa CPU %d%% → READY",
                        cpu_weight,
                    )
                    return dict(weight_map), attempt, cpu_weight

            if outcome in ("oom", "timeout", "crashed"):
                new_map, new_cpu = self._escalate_cpu_offload(
                    weight_map,
                    cpu_weight,
                    spill_order,
                    active_indices,
                    vram_by_index,
                    pinned_map,
                )
                if new_map is None:
                    _auto_balance_log(
                        "Fase 1 — CPU offload esgotado (max %d%%) → FALHA",
                        cpu_weight,
                        level="error",
                    )
                    return None, attempt, cpu_weight
                _auto_balance_log(
                    "Fase 1 — CPU %d%% → %s; escalando para CPU=%d%%",
                    cpu_weight,
                    outcome.upper(),
                    new_cpu,
                    level="warn",
                )
                weight_map = new_map
                cpu_weight = new_cpu
                continue

            return None, attempt, cpu_weight

        return None, attempt, cpu_weight

    def _trim_trailing_spill_gpus(
        self,
        request,
        all_gpus: List[dict],
        main_index: int,
        spill_order: List[int],
        weight_map: Dict[int, int],
        pinned_map: Dict[int, int],
        active_indices: List[int],
        attempt: int,
        cpu_config: Dict[str, Any],
        cpu_weight: int,
    ) -> Tuple[Dict[int, int], int, int]:
        """Remove trailing spill GPUs when the model still loads (probe-ready)."""
        optimized = dict(weight_map)
        main_idx = spill_order[0]

        for idx in reversed(spill_order[1:]):
            self._raise_if_cancelled()
            if idx in pinned_map:
                continue
            spill_w = optimized.get(idx, 0)
            if spill_w <= 0:
                continue

            trial = dict(optimized)
            trial[idx] = 0
            trial[main_idx] = trial.get(main_idx, 0) + spill_w

            attempt += 1
            label = self.planner.format_weights_with_cpu(
                trial, spill_order, cpu_weight
            )
            self._set_progress(
                attempt,
                f"Fase 1b — remover GPU{idx}: {label}",
                weight_map=trial,
                all_gpus=all_gpus,
                main_index=main_index,
                pinned_map=pinned_map,
                cpu_weight=cpu_weight,
                cpu_config=cpu_config,
            )
            outcome = self._probe_start(
                request,
                trial,
                main_index,
                all_gpus,
                attempt,
                cpu_weight=cpu_weight,
                cpu_config=cpu_config,
                spill_order=spill_order,
                selected_indices=active_indices,
                pinned_map=pinned_map,
            )
            if outcome == "cancelled":
                raise AutoBalanceCancelled()
            if outcome == "ready":
                _auto_balance_log(
                    "TRIM: GPU%d removida — %d%% redistribuido para GPU%d | %s",
                    idx,
                    spill_w,
                    main_idx,
                    label,
                )
                optimized = trial
            else:
                _auto_balance_log(
                    "TRIM: GPU%d mantida (%d%%) — probe=%s",
                    idx,
                    spill_w,
                    outcome,
                    level="warn",
                )

        return optimized, attempt, cpu_weight

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
        *,
        active_indices: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, int], int, int]:
        """Binary-search each GPU's max weight, in spill order, until VRAM ~full.

        Cláusula pétrea (cascata estrita): preenche cada GPU até seu cap de VRAM
        (~95–99%) **na ordem de prioridade** antes que a próxima receba a sobra.
        Para cada GPU (main → secundárias por ordem de spill) sobe-se o peso
        reclamando das GPUs posteriores na cascata (até 0); GPUs anteriores nunca
        são reduzidas, então a main mantém prioridade absoluta. A última GPU da
        ordem não tem doadora posterior — vira no-op e absorve o restante.
        """
        _auto_balance_log(
            "Fase 2: input weight_map={%s} main_index=%d",
            ", ".join(f"{i}: {w}%" for i, w in weight_map.items()),
            main_index,
        )
        optimized = dict(weight_map)
        selected = active_indices or self.planner.active_subset(optimized)
        active_ordered = [
            idx
            for idx in spill_order
            if idx in selected and optimized.get(idx, 0) > 0
        ]

        for target_idx in active_ordered:
            self._raise_if_cancelled()
            if target_idx in pinned_map:
                _auto_balance_log(
                    "Fase 2: GPU%d fixada em %d%% (ignorada)",
                    target_idx,
                    pinned_map[target_idx],
                )
                continue

            maximize_floors = self.planner.gpu_weight_floors_for_maximize(
                optimized, spill_order
            )
            lo = max(
                optimized.get(target_idx, 0),
                maximize_floors.get(target_idx, MIN_GPU_WEIGHT),
            )
            hi = self._max_gpu_weight_for_maximize(
                optimized,
                spill_order,
                target_idx,
                pinned_map,
                cpu_config,
                selected_indices=selected,
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
                    selected_indices=selected,
                )
                if trial is None:
                    hi = mid - 1
                    continue

                attempt += 1
                gpu_name = next(
                    (g["name"] for g in all_gpus if g["index"] == target_idx),
                    f"GPU{target_idx}",
                )
                probe_cpu = self._resolve_probe_cpu_weight(
                    trial,
                    cpu_config,
                    cpu_spill_allowed=bool(cpu_config.get("cpu_spill_allowed")),
                    spill_order=spill_order,
                    selected_indices=selected,
                    pinned_map=pinned_map,
                )
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
                    spill_order=spill_order,
                    selected_indices=selected,
                    pinned_map=pinned_map,
                )
                if outcome == "cancelled":
                    raise AutoBalanceCancelled()
                if outcome != "ready":
                    hi = mid - 1
                    continue

                time.sleep(VRAM_SETTLE_SEC)
                vram_pct = self._get_vram_pct(target_idx)
                if vram_pct > TARGET_VRAM_PCT_MAX:
                    hi = mid - 1
                    continue
                if vram_pct >= best_vram:
                    best = trial
                    best_vram = vram_pct

                # The principal is the primary optimization objective: keep
                # probing upward until OOM/the hard VRAM ceiling, even after it
                # first enters the target band. Secondary GPUs may stop once
                # they reach the band because they only absorb the remainder.
                if (
                    target_idx != main_index
                    and TARGET_VRAM_PCT_MIN <= vram_pct <= TARGET_VRAM_PCT_MAX
                ):
                    break

                lo = mid + 1

            optimized = best
            cpu_weight = self._resolve_probe_cpu_weight(
                optimized,
                cpu_config,
                cpu_spill_allowed=bool(cpu_config.get("cpu_spill_allowed")),
                spill_order=spill_order,
                selected_indices=selected,
                pinned_map=pinned_map,
            )
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
            optimized, cpu_weight = self._finalize_cpu_split(
                optimized,
                cpu_config,
                cpu_spill_allowed=bool(cpu_config.get("cpu_spill_allowed")),
                spill_order=spill_order,
                selected_indices=selected,
                pinned_map=pinned_map,
            )

        ok, err = self.planner.validate_cpu_not_dominant(
            optimized, cpu_weight
        )
        if not ok:
            _auto_balance_log(
                "_maximize_vram_per_gpu: REJECTED cpu_dominant — %s",
                err,
                level="warn",
            )

        return optimized, attempt, cpu_weight

    def _fine_tune_main_gpu_weight(
        self,
        request,
        all_gpus: List[dict],
        main_index: int,
        spill_order: List[int],
        optimized: Dict[int, int],
        vram_by_index: Dict[int, int],
        pinned_map: Dict[int, int],
        active_indices: List[int],
        attempt: int,
        cpu_config: Dict[str, Any],
        cpu_weight: int,
    ) -> Tuple[Dict[int, int], int]:
        """Fase 3 — fine-tuning: sobe main GPU 1% por 1% até OOM.

        GPU-ONLY: CPU completamente desabilitado. Apenas GPUs participam.
        O último % sem OOM na main GPU é o máximo potencial.
        """
        if not spill_order:
            return optimized, attempt

        current_main_weight = optimized.get(main_index, 0)
        if current_main_weight >= 100:
            _auto_balance_log(
                "Fase 3: GPU%d já está em 100%% — pulando fine-tuning",
                main_index,
            )
            return optimized, attempt

        # Encontra o peso máximo das GPUs secundárias ativas
        secondary_total = sum(
            optimized.get(idx, 0) for idx in spill_order[1:] if optimized.get(idx, 0) > 0
        )
        # Máximo teórico: main = 100
        max_main = min(100, current_main_weight + secondary_total)

        # CPU completamente desabilitado na Fase 3 (GPU-only)
        gpu_only_config = {"enabled": False, "pinned": False, "weight": 0}

        _auto_balance_log(
            "=== Fase 3 INICIO (fine-tuning 1%%, GPU-only) ===\n"
            "  main_gpu=%d current_weight=%d%% max_possible=%d%%\n"
            "  secondary_total=%d%% CPU=desabilitado",
            main_index,
            current_main_weight,
            max_main,
            secondary_total,
        )

        lo = current_main_weight + FINE_TUNE_STEP
        hi = max_main
        best_weight = current_main_weight
        best_map = dict(optimized)

        while lo <= hi:
            self._raise_if_cancelled()
            main_weight = lo
            # Calcula o trial: sobe main, reduz secundárias (GPU-only)
            trial = self._adjust_main_fine_tune(
                optimized,
                spill_order,
                main_index,
                main_weight,
                pinned_map,
            )

            if trial is None:
                break

            attempt += 1
            gpu_name = next(
                (g["name"] for g in all_gpus if g["index"] == main_index),
                f"GPU{main_index}",
            )
            self._set_progress(
                attempt,
                f"Fase 3 — fine-tuning {gpu_name}: testando {main_weight}%",
                weight_map=trial,
                all_gpus=all_gpus,
                main_index=main_index,
                pinned_map=pinned_map,
                cpu_weight=0,
                cpu_config=gpu_only_config,
            )

            outcome = self._probe_start(
                request,
                trial,
                main_index,
                all_gpus,
                attempt,
                cpu_weight=0,
                cpu_config=gpu_only_config,
                spill_order=spill_order,
                selected_indices=active_indices,
                pinned_map=pinned_map,
            )

            if outcome == "cancelled":
                raise AutoBalanceCancelled()

            if outcome == "ready":
                best_weight = main_weight
                best_map = dict(trial)
                _auto_balance_log(
                    "Fase 3 — GPU%d=%d%% → READY (novo melhor)",
                    main_index,
                    main_weight,
                )
                lo = main_weight + FINE_TUNE_STEP
            else:
                # OOM ou crashed — este peso é demais, tenta menos
                _auto_balance_log(
                    "Fase 3 — GPU%d=%d%% → %s (limit reached)",
                    main_index,
                    main_weight,
                    outcome.upper(),
                    level="warn",
                )
                # Não precisa baixar hi, porque já sabemos que lo é o limite
                break

        if best_weight > current_main_weight:
            _auto_balance_log(
                "=== Fase 3 CONCLUIDA ===\n"
                "  GPU%d weight: %d%% → %d%% (+%d%%)\n"
                "  weight_map={%s}",
                main_index,
                current_main_weight,
                best_weight,
                best_weight - current_main_weight,
                ", ".join(f"{i}: {w}%" for i, w in best_map.items()),
            )
        else:
            _auto_balance_log(
                "Fase 3: sem incremento possivel — main GPU%d ja em %d%%",
                main_index,
                current_main_weight,
                level="warn",
            )

        return best_map, attempt

    def _adjust_main_fine_tune(
        self,
        optimized: Dict[int, int],
        spill_order: List[int],
        main_index: int,
        new_main_weight: int,
        pinned_map: Dict[int, int],
    ) -> Optional[Dict[int, int]]:
        """Adjust weights for fine-tuning: raise main GPU, drain secondaries.

        Takes slack from secondary GPUs in spill order (down to 0), respecting
        pinned GPUs. Returns a valid weight map that sums to 100.
        """
        pinned_map = pinned_map or {}
        new_main_weight = max(0, min(100, new_main_weight))

        active = [idx for idx in spill_order if optimized.get(idx, 0) > 0]
        if main_index not in active:
            return None

        trial = {i: optimized.get(i, 0) for i in active}
        delta = new_main_weight - trial[main_index]
        trial[main_index] = new_main_weight

        if delta == 0:
            return trial if sum(trial.values()) == 100 else None

        if delta > 0:
            # Need to drain secondary GPUs
            donors = [
                i for i in active
                if i != main_index and i not in pinned_map
            ]
            # Sort by spill order (later spill GPUs give first)
            donors.sort(key=lambda i: spill_order.index(i), reverse=True)

            need = delta
            for donor in donors:
                if spill_order.index(donor) <= spill_order.index(main_index):
                    continue
                take = min(trial[donor], need)
                trial[donor] -= take
                need -= take
                if need == 0:
                    break

            if need > 0:
                # Not enough secondary capacity — check if we can stay within 100
                if sum(trial.values()) - need <= 100:
                    # Just reduce and keep what we can
                    for donor in donors:
                        if need <= 0:
                            break
                        take = min(trial[donor], need)
                        trial[donor] -= take
                        need -= take
                    return trial if sum(trial.values()) <= 100 else None
                return None
        else:
            # Main weight decreased — redistribute to secondaries
            receivers = [
                i for i in active
                if i != main_index and i not in pinned_map
            ]
            receivers.sort(key=lambda i: spill_order.index(i), reverse=True)

            give = -delta
            for recv in receivers:
                if spill_order.index(recv) <= spill_order.index(main_index):
                    continue
                trial[recv] += give
                give = 0
                break
            if give > 0:
                return None

        if sum(trial.values()) != 100:
            return None
        return trial

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
        spill_order: Optional[List[int]] = None,
        selected_indices: Optional[List[int]] = None,
        pinned_map: Optional[Dict[int, int]] = None,
    ) -> str:
        if self.process_manager.auto_balance_cancel_requested:
            return "cancelled"
        cpu_config = cpu_config or {}
        cpu_spill_allowed = cpu_weight > 0 or bool(cpu_config.get("pinned"))
        budget_selected = self._budget_selected_indices(
            weight_map, spill_order, selected_indices
        )
        synced_map, probe_cpu = self._finalize_cpu_split(
            weight_map,
            cpu_config,
            cpu_spill_allowed=cpu_spill_allowed,
            spill_order=spill_order,
            selected_indices=budget_selected,
            pinned_map=pinned_map,
            skip_floors=True,
        )
        gpu_weights = self.planner.to_gpu_weights(
            all_gpus,
            synced_map,
            main_index,
            cpu_weight=probe_cpu,
            cpu_pinned=bool(cpu_config.get("pinned") and probe_cpu > 0),
            cpu_valve_enabled=bool(cpu_config.get("enabled")),
        )

        gpu_weight_details = ", ".join(
            f"{w.device}[{w.index}]={w.weight}%" for w in gpu_weights if w.active
        )
        _auto_balance_log(
            "PROBE #%d START (gpus=%d): weight_map={%s} synced={%s} cpu=%d->%d [%s]",
            attempt,
            len(budget_selected),
            ", ".join(f"{i}: {w}%" for i, w in weight_map.items()),
            ", ".join(
                f"{i}: {synced_map.get(i, 0)}%"
                for i in sorted(set(list(weight_map.keys()) + list(synced_map.keys())))
            ),
            cpu_weight,
            probe_cpu,
            gpu_weight_details,
        )

        crash_retries = 0
        while True:
            self.process_manager.stop(self.port)  # also waits for the port to be released
            try:
                self.process_manager.start(
                    model_path=request.path,
                    gpu_weights=gpu_weights,
                    context_size=request.context_size,
                    mmproj_path=request.mmproj_path,
                    mmproj_disabled=request.mmproj_disabled
                    or request.vision_enabled is False,
                    vision_enabled=request.vision_enabled,
                    split_mode=request.split_mode,
                    parallel_slots=request.parallel_slots,
                    batch_size=request.batch_size,
                    # Sonda deve refletir o consumo real de VRAM da carga final:
                    # cache/ubatch/threads/numa/binário afetam a memória usada.
                    ubatch_size=request.ubatch_size,
                    cache_type_k=request.cache_type_k,
                    cache_type_v=request.cache_type_v,
                    threads=request.threads,
                    threads_batch=request.threads_batch,
                    numa_enabled=request.numa_enabled,
                    thinking_enabled=request.thinking_enabled,
                    mtp_enabled=request.mtp_enabled,
                    mtp_draft_tokens=request.mtp_draft_tokens,
                    mtp_model_path=request.mtp_model_path,
                    flash_attn_enabled=request.flash_attn_enabled,
                    total_layers=request.total_layers,
                    port=self.port,
                    llama_server_bin=request.llama_server_bin,
                )
            except Exception as exc:
                logger.error(
                    "Auto-balance probe #%d: START EXCEPTION model=%s error=%s",
                    attempt,
                    request.path,
                    exc,
                )
                # Could not even launch the server — treat as a (retryable) crash.
                self._last_crash_elapsed = 0.0
                self._last_server_log_tail = [f"START EXCEPTION: {exc}"]
                outcome = "crashed"
            else:
                outcome = self._wait_for_outcome()

            _auto_balance_log(
                "PROBE #%d RESULT: %s | synced={%s} cpu=%d",
                attempt,
                outcome,
                ", ".join(f"{i}: {synced_map.get(i, 0)}%" for i in synced_map),
                probe_cpu,
                level="warn" if outcome in ("oom", "timeout", "crashed") else "info",
            )

            is_fast_crash = (
                outcome == "crashed"
                and self._last_crash_elapsed < CRASH_FAST_FAIL_SEC
            )
            # A fast non-OOM death is often transient (port not yet released).
            # Retry a few times before deciding it is fatal.
            if (
                is_fast_crash
                and crash_retries < CRASH_RETRY_MAX
                and not self.process_manager.auto_balance_cancel_requested
            ):
                crash_retries += 1
                _auto_balance_log(
                    "PROBE #%d → CRASH (não-OOM, %.1fs) — retry %d/%d "
                    "(provável porta/transitório)",
                    attempt,
                    self._last_crash_elapsed,
                    crash_retries,
                    CRASH_RETRY_MAX,
                    level="warn",
                )
                time.sleep(CRASH_RETRY_BACKOFF_SEC)
                continue

            # Persisted after retries: a genuine crash (incompatible
            # model/binary or bad launch flag). Abort and inform the user —
            # escalating GPUs/CPU would only produce a misleading
            # "hardware capacity exceeded" verdict. A slow non-OOM death is left
            # on the normal path (some real OOMs print messages our regex misses).
            if is_fast_crash:
                _auto_balance_log(
                    "PROBE #%d → CRASH persistente (não-OOM, %.1fs após %d retries) "
                    "— abortando auto-balance",
                    attempt,
                    self._last_crash_elapsed,
                    crash_retries,
                    level="error",
                )
                raise AutoBalanceServerCrashed(
                    weight_map,
                    probe_cpu,
                    self._last_crash_elapsed,
                    self._last_server_log_tail,
                )
            return outcome

    def _get_vram_pct(self, gpu_index: int) -> float:
        metrics = self.gpu_manager.get_metrics()
        for gpu in metrics.get("gpus", []):
            if gpu.get("index") == gpu_index:
                pct = float(gpu.get("vram_pct", 0))
                logger.debug(
                    "_get_vram_pct: gpu=%d vram_pct=%.1f",
                    gpu_index,
                    pct,
                )
                return pct
        logger.debug("_get_vram_pct: gpu=%d not found in metrics", gpu_index)
        return 0.0

    def _capture_hardware_snapshot(self, label: str) -> Dict[str, Any]:
        """Captura snapshot completo de hardware (GPUs + CPU/RAM)."""
        metrics = self.gpu_manager.get_metrics()
        gpu_details = []
        for gpu in metrics.get("gpus", []):
            gpu_details.append({
                "index": gpu["index"],
                "mem_used_mb": float(gpu.get("mem_used", 0)),
                "mem_total_mb": float(gpu.get("mem_total", 0)),
                "vram_pct": gpu.get("vram_pct", 0),
                "util_pct": float(gpu.get("util", 0)),
                "temp_c": float(gpu.get("temp", 0)) if gpu.get("temp") else 0,
                "power_w": float(gpu.get("power", 0)) if gpu.get("power") else 0,
            })

        vm = psutil.virtual_memory() if "psutil" in globals() or hasattr(psutil, "virtual_memory") else None
        ram_total = vm.total // (1024 * 1024) if vm else 0
        ram_used = vm.used // (1024 * 1024) if vm else 0
        ram_pct = vm.percent if vm else 0
        cpu_percent = metrics.get("cpu", 0)

        return {
            "label": label,
            "gpus": gpu_details,
            "ram_used_mb": ram_used,
            "ram_total_mb": ram_total,
            "ram_pct": ram_pct,
            "cpu_pct": cpu_percent,
            "cpu_name": metrics.get("cpu_name", "Unknown"),
            "cpu_temp": metrics.get("cpu_temp"),
            "cpu_power": metrics.get("cpu_power"),
        }

    def _vram_summary(
        self, weight_map: Dict[int, int], spill_order: List[int]
    ) -> str:
        parts = []
        for idx in spill_order:
            if weight_map.get(idx, 0) <= 0:
                continue
            pct = self._get_vram_pct(idx)
            parts.append(f"GPU{idx} VRAM {pct:.0f}%")
        summary = "; ".join(parts) if parts else "N/A"
        logger.info(
            "_vram_summary: weight_map={%s} -> summary=%s",
            ", ".join(f"{i}: {w}%" for i, w in weight_map.items()),
            summary,
        )
        return summary

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
        spill_order: Optional[List[int]] = None,
        selected_indices: Optional[List[int]] = None,
    ) -> None:
        prev = self.process_manager.recovery_state or {}
        state = {
            "active": True,
            "failed": False,
            "message": message,
            "auto_balance": True,
            "attempt": attempt,
            "model": getattr(self, "_model_path", None) or prev.get("model"),
            "smart_calibration": getattr(
                self, "_smart_calibration", prev.get("smart_calibration", False)
            ),
            "run_id": prev.get("run_id"),
        }
        if weight_map is not None and all_gpus is not None and main_index is not None:
            pinned_indices = set((pinned_map or {}).keys())
            cpu_config = cpu_config or {}
            progress_map, progress_cpu = self._finalize_cpu_split(
                weight_map,
                cpu_config,
                cpu_spill_allowed=(
                    cpu_weight > 0 or bool(cpu_config.get("pinned"))
                ),
                spill_order=spill_order,
                selected_indices=selected_indices,
                pinned_map=pinned_map,
            )
            # Progress display: only mark CPU when it actually carries load in
            # THIS probe (or is pinned). During GPU-only phases (1, 1b, 2, 3)
            # progress_cpu is 0, so the CPU valve must show unmarked even when
            # the user enabled it — it only re-appears once spill is in use.
            cpu_in_use = progress_cpu > 0 or bool(cpu_config.get("pinned"))
            weights = self.planner.to_gpu_weights(
                all_gpus,
                progress_map,
                main_index,
                pinned_indices,
                cpu_weight=progress_cpu,
                cpu_pinned=bool(cpu_config.get("pinned") and progress_cpu > 0),
                cpu_valve_enabled=cpu_in_use,
            )
            state["gpu_weights"] = [w.model_dump() for w in weights]
        self.process_manager.recovery_state = state

    def _wait_for_outcome(self) -> str:
        # self.port: em multi-instância a sonda não roda na porta default e o
        # log fica em server_{port}.log — ler server.log observaria outra instância.
        path = self.log_manager.get_server_log_path(self.port)
        deadline = time.time() + PROBE_TIMEOUT_SEC
        last_pos = 0
        start_time = time.time()

        while time.time() < deadline:
            if self.process_manager.auto_balance_cancel_requested:
                self.process_manager.stop(self.port)
                elapsed = time.time() - start_time
                logger.info(
                    "_wait_for_outcome: CANCELLED after %.1fs (log=%s)",
                    elapsed,
                    path,
                )
                return "cancelled"
            if not self._process_alive():
                # Process died. Drain whatever the server wrote before exiting
                # and check for an OOM signal FIRST — an OOM that kills the
                # process between polls must not be misread as a generic crash.
                chunk, last_pos = self._read_log_since(path, last_pos)
                elapsed = time.time() - start_time
                if chunk and OOM_PATTERNS.search(chunk):
                    self._last_server_log_tail = chunk.strip().split("\n")[-5:]
                    logger.warning(
                        "_wait_for_outcome: OOM-on-exit after %.1fs (log=%s) | preview=%s",
                        elapsed,
                        path,
                        self._last_server_log_tail,
                    )
                    return "oom"
                # No OOM marker -> genuine crash (incompatible model/binary or
                # bad launch flag). Capture the tail for the user-facing message.
                self._last_server_log_tail = (
                    chunk.strip().split("\n")[-5:] if chunk else []
                )
                self._last_crash_elapsed = elapsed
                logger.warning(
                    "_wait_for_outcome: CRASHED after %.1fs (log=%s) | preview=%s",
                    elapsed,
                    path,
                    self._last_server_log_tail,
                )
                return "crashed"

            chunk, last_pos = self._read_log_since(path, last_pos)
            if chunk:
                if OOM_PATTERNS.search(chunk):
                    elapsed = time.time() - start_time
                    # Log as 3 primeiras linhas do log do servidor para diagnóstico
                    preview_lines = chunk.strip().split("\n")[:3]
                    logger.warning(
                        "_wait_for_outcome: OOM detected after %.1fs (log=%s) | preview=%s",
                        elapsed,
                        path,
                        preview_lines,
                    )
                    return "oom"
                if READY_PATTERNS.search(chunk):
                    elapsed = time.time() - start_time
                    ready_lines = chunk.strip().split("\n")[:3]
                    logger.info(
                        "_wait_for_outcome: READY after %.1fs (log=%s) | preview=%s",
                        elapsed,
                        path,
                        ready_lines,
                    )
                    return "ready"

            time.sleep(POLL_INTERVAL_SEC)

        elapsed = time.time() - start_time
        logger.warning(
            "_wait_for_outcome: TIMEOUT after %.1fs (log=%s, last_pos=%d)",
            elapsed,
            path,
            last_pos,
        )
        return "timeout"

    def _process_alive(self) -> bool:
        with self.process_manager._lock:
            proc = self.process_manager.processes.get(self.port)
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
