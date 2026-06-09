"""Unified GPU/CPU load distribution engine — strict priority-fill cascade.

This module provides a stateless engine that calculates the distribution of a
model across GPUs and CPU. It is the single source of truth for all load
distribution logic in the AutoManager system.

Policy (ADR-001 / ADR-003):
  - Strict priority-fill: fill each GPU up to ``vram_limit_pct`` (default 98%)
    of its VRAM, in priority order, before using the next GPU. Never split
    proportionally.
  - Priority order: main GPU first, then the remaining GPUs by ascending index.
  - CPU/RAM is the last resort: it only receives load after every enabled GPU
    is filled to the limit.
  - CPU valve: when ``cpu_enabled=False`` and the model does not fit in the
    GPUs, the result is marked infeasible (caller blocks with an alert).

Output contract: weights are expressed as percentages of the model
(``mb_device / model_mb * 100``), preserving ``DistributionResult`` and the
downstream mapping to ``--tensor-split`` / ``--ngl`` / ``--main-gpu``.

Usage examples:

    >>> # Model fits entirely in the main GPU — nothing spills
    >>> result = LoadDistributor.distribute(
    ...     gpu_vram={0: 24000, 1: 16000},
    ...     priority_order=[0, 1],
    ...     estimated_model_vram_mb=20000,
    ...     cpu_enabled=True,
    ... )
    >>> result.gpu_weights[1]
    0
    >>> result.cpu_weight
    0

    >>> # Model exceeds all GPUs, CPU enabled — CPU absorbs the remainder
    >>> result = LoadDistributor.distribute(
    ...     gpu_vram={0: 4000},
    ...     priority_order=[0],
    ...     estimated_model_vram_mb=8000,
    ...     cpu_enabled=True,
    ... )
    >>> result.cpu_weight > 0
    True

    >>> # Model exceeds all GPUs, CPU disabled — infeasible
    >>> result = LoadDistributor.distribute(
    ...     gpu_vram={0: 4000},
    ...     priority_order=[0],
    ...     estimated_model_vram_mb=8000,
    ...     cpu_enabled=False,
    ... )
    >>> result.is_feasible
    False
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# VRAM occupation limit per GPU (ADR-001). Fixed, not exposed in the UI.
DEFAULT_VRAM_LIMIT_PCT = 98.0

# Sentinel key for the CPU device in internal MB→% conversion.
_CPU_KEY = -1


@dataclass(frozen=True)
class DistributionResult:
    """Result of a GPU/CPU load distribution calculation.

    Attributes:
        gpu_weights: Mapping of GPU index to weight percentage.
        cpu_weight: CPU weight percentage (0 when GPU-only or valve off).
        total_gpu_pct: Sum of all GPU weight percentages.
        is_feasible: False when model exceeds all available hardware.
    """

    gpu_weights: Dict[int, int]
    cpu_weight: int
    total_gpu_pct: int
    is_feasible: bool


class LoadDistributor:
    """
    Stateless engine that calculates the optimal distribution of model layers
    across GPUs and CPU following the GPU-first, CPU-minimum policy.

    The CPU checkbox acts as an on/off valve:
      - OFF (cpu_enabled=False)  -> CPU_weight = 0, model must fit in GPUs
      - ON  (cpu_enabled=True)   -> CPU absorbs spill-over only

    No hard caps on CPU weight — CPU uses whatever is needed as spill-over.
    """

    @staticmethod
    def distribute(
        gpu_vram: Dict[int, int],
        gpu_weights: Optional[Dict[int, int]] = None,
        total_layers: int = 0,
        estimated_model_vram_mb: int = 0,
        cpu_enabled: bool = True,
        priority_order: Optional[List[int]] = None,
        vram_limit_pct: float = DEFAULT_VRAM_LIMIT_PCT,
    ) -> DistributionResult:
        """
        Calculate the GPU/CPU distribution as a strict priority-fill cascade.

        For each GPU in priority order, allocate
        ``min(vram_total * vram_limit_pct / 100, remaining_model_mb)`` in MB.
        The next GPU only receives load once the previous one reaches the limit.
        The CPU receives the remainder only after every GPU is filled. The MB
        allocations are converted to percentages of the model (the contract
        ``DistributionResult`` expects), reconciled to sum 100%.

        Args:
            gpu_vram: Mapping of GPU index to total VRAM in MB.
            gpu_weights: Deprecated. Kept for signature compatibility; only its
                key order is used as a fallback when ``priority_order`` is None.
            total_layers: Unused by the cascade (layer mapping happens
                downstream); kept for signature compatibility.
            estimated_model_vram_mb: Estimated VRAM needed for the model in MB.
            cpu_enabled: CPU valve — True allows spill-over, False blocks it.
            priority_order: GPU indices in priority order (main first, then by
                ascending index). When None, derived from ``gpu_weights`` keys
                or sorted ``gpu_vram`` keys.
            vram_limit_pct: VRAM occupation limit per GPU (default 98%).

        Returns:
            DistributionResult with gpu_weights (%), cpu_weight (%),
            total_gpu_pct and is_feasible.
        """
        if not gpu_vram:
            return DistributionResult(
                gpu_weights={},
                cpu_weight=0,
                total_gpu_pct=0,
                is_feasible=False,
            )

        order = LoadDistributor._resolve_order(gpu_vram, gpu_weights, priority_order)

        # Unknown model size: cannot fill by MB — keep everything on GPUs.
        if estimated_model_vram_mb <= 0:
            passthrough = dict(gpu_weights) if gpu_weights else {}
            total_gpu_pct = sum(passthrough.values()) or 100
            return DistributionResult(
                gpu_weights=passthrough,
                cpu_weight=0,
                total_gpu_pct=total_gpu_pct,
                is_feasible=True,
            )

        # Strict priority-fill cascade (MB).
        mb_by_gpu: Dict[int, int] = {idx: 0 for idx in gpu_vram}
        remaining = estimated_model_vram_mb
        for idx in order:
            if remaining <= 0:
                break
            cap = int(max(0, gpu_vram.get(idx, 0)) * vram_limit_pct / 100.0)
            alloc = min(cap, remaining)
            mb_by_gpu[idx] = alloc
            remaining -= alloc

        # Remainder did not fit in the GPUs.
        if remaining > 0 and not cpu_enabled:
            # Infeasible: preserve GPU shares (as % of GPU caps) for reporting.
            gpu_pct = LoadDistributor._mb_to_pct(mb_by_gpu, _CPU_KEY, 0)
            cpu_weight = gpu_pct.pop(_CPU_KEY, 0)
            return DistributionResult(
                gpu_weights=gpu_pct,
                cpu_weight=cpu_weight,
                total_gpu_pct=sum(gpu_pct.values()),
                is_feasible=False,
            )

        cpu_mb = max(0, remaining)
        pct = LoadDistributor._mb_to_pct(mb_by_gpu, _CPU_KEY, cpu_mb)
        cpu_weight = pct.pop(_CPU_KEY, 0)
        return DistributionResult(
            gpu_weights=pct,
            cpu_weight=cpu_weight,
            total_gpu_pct=sum(pct.values()),
            is_feasible=True,
        )

    @staticmethod
    def _resolve_order(
        gpu_vram: Dict[int, int],
        gpu_weights: Optional[Dict[int, int]],
        priority_order: Optional[List[int]],
    ) -> List[int]:
        """Resolve the GPU priority order, keeping only indices present in VRAM."""
        if priority_order:
            ordered = [i for i in priority_order if i in gpu_vram]
            # Append any GPU missing from priority_order (deterministic tail).
            ordered += [i for i in sorted(gpu_vram) if i not in ordered]
            return ordered
        if gpu_weights:
            ordered = [i for i in gpu_weights if i in gpu_vram]
            ordered += [i for i in sorted(gpu_vram) if i not in ordered]
            return ordered
        return sorted(gpu_vram)

    @staticmethod
    def _mb_to_pct(
        mb_by_gpu: Dict[int, int], cpu_key: int, cpu_mb: int
    ) -> Dict[int, int]:
        """Convert per-device MB into integer percentages summing to 100.

        Uses the largest-remainder method so rounding never drops or adds a
        point; every GPU index in ``mb_by_gpu`` is present in the result (0 when
        it received nothing). The CPU share is keyed by ``cpu_key``.
        """
        amounts: Dict[int, int] = dict(mb_by_gpu)
        if cpu_mb > 0:
            amounts[cpu_key] = cpu_mb
        total = sum(amounts.values())
        if total <= 0:
            return {idx: 0 for idx in mb_by_gpu}

        floors: Dict[int, int] = {}
        remainders = []
        for idx, mb in amounts.items():
            exact = mb * 100.0 / total
            floor = int(exact)
            floors[idx] = floor
            remainders.append((exact - floor, idx))

        leftover = 100 - sum(floors.values())
        # Distribute the leftover points to the largest fractional remainders.
        remainders.sort(reverse=True)
        for _, idx in remainders[:max(0, leftover)]:
            floors[idx] += 1

        # Ensure every GPU key is present even when it got 0 MB.
        for idx in mb_by_gpu:
            floors.setdefault(idx, 0)
        return floors

    @staticmethod
    def is_feasible(
        gpu_vram: Dict[int, int],
        estimated_model_vram_mb: int,
        cpu_enabled: bool = True,
    ) -> bool:
        """
        Check if the model can fit in the given hardware.

        Args:
            gpu_vram: Mapping of GPU index to available VRAM in MB.
            estimated_model_vram_mb: Estimated VRAM needed for the model in MB.
            cpu_enabled: Whether CPU spill-over is allowed.

        Returns:
            True if the model can be loaded, False otherwise.
        """
        if not gpu_vram or estimated_model_vram_mb <= 0:
            return estimated_model_vram_mb <= 0

        total_gpu_vram = sum(gpu_vram.values())
        if total_gpu_vram >= estimated_model_vram_mb:
            return True
        return cpu_enabled

    @staticmethod
    def compute_n_gpu_layers(
        total_layers: int,
        gpu_weight_pct: float,
    ) -> int:
        """
        Convert GPU weight percentage to --ngl (number of GPU layers) value.

        Args:
            total_layers: Total number of layers in the model.
            gpu_weight_pct: Percentage of layers to put on GPU (0-100).

        Returns:
            Number of GPU layers, clamped to [0, total_layers].
        """
        return max(0, min(total_layers, int(round(gpu_weight_pct / 100.0 * total_layers))))
