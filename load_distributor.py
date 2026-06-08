"""Unified GPU/CPU load distribution engine — GPU-first, CPU-minimum policy.

This module provides a stateless engine that calculates the optimal distribution
of model layers across GPUs and CPU. It is the single source of truth for all
load distribution logic in the AutoManager system.

Policy:
  - GPU-first: maximize GPU usage before offloading to CPU.
  - CPU as spill-over: CPU absorbs only what GPUs cannot hold.
  - CPU valve: the checkbox in the GPU resources panel acts as on/off valve.

Usage examples:

    >>> # Model fits entirely in GPUs — no CPU needed
    >>> result = LoadDistributor.distribute(
    ...     gpu_vram={0: 8000, 1: 8000},
    ...     gpu_weights={0: 50, 1: 50},
    ...     total_layers=80,
    ...     estimated_model_vram_mb=12000,
    ...     cpu_enabled=True,
    ... )
    >>> result.cpu_weight
    0
    >>> result.is_feasible
    True

    >>> # Model exceeds GPU capacity, CPU enabled — spill-over
    >>> result = LoadDistributor.distribute(
    ...     gpu_vram={0: 4000},
    ...     gpu_weights={0: 100},
    ...     total_layers=32,
    ...     estimated_model_vram_mb=8000,
    ...     cpu_enabled=True,
    ... )
    >>> result.cpu_weight > 0
    True

    >>> # Model exceeds GPU capacity, CPU disabled — infeasible
    >>> result = LoadDistributor.distribute(
    ...     gpu_vram={0: 4000},
    ...     gpu_weights={0: 100},
    ...     total_layers=32,
    ...     estimated_model_vram_mb=8000,
    ...     cpu_enabled=False,
    ... )
    >>> result.is_feasible
    False
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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
        gpu_weights: Dict[int, int],
        total_layers: int,
        estimated_model_vram_mb: int,
        cpu_enabled: bool = True,
    ) -> DistributionResult:
        """
        Calculate the GPU/CPU distribution following GPU-first, CPU-minimum.

        Decision rules (evaluated in order):
        1. cpu_enabled=False: distribute across GPUs only, CPU = 0.
           If model fits in VRAM -> is_feasible=True.
           If model doesn't fit -> is_feasible=False (for error reporting).
        2. total_gpu_vram >= estimated_model_vram_mb: distribute across GPUs,
           CPU = 0, is_feasible=True.
        3. total_gpu_vram < estimated_model_vram_mb AND cpu_enabled=True:
           GPU receives proportional share of model_vram, CPU = remainder,
           is_feasible=True.
        4. total_gpu_vram < estimated_model_vram_mb AND cpu_enabled=False:
           is_feasible=False, GPU weights preserved for error reporting.

        Args:
            gpu_vram: Mapping of GPU index to available VRAM in MB.
            gpu_weights: User-defined weight percentages per GPU (manual mode).
                Preserved proportionally when spill-over is needed.
            total_layers: Total number of layers in the model.
            estimated_model_vram_mb: Estimated VRAM needed for the model in MB.
            cpu_enabled: CPU valve — True allows spill-over, False disables CPU.

        Returns:
            DistributionResult with gpu_weights, cpu_weight, total_gpu_pct,
            and is_feasible flags.
        """
        if not gpu_vram:
            return DistributionResult(
                gpu_weights={},
                cpu_weight=0,
                total_gpu_pct=0,
                is_feasible=False,
            )

        total_gpu_vram = sum(gpu_vram.values())

        # Rule 1: CPU valve OFF — no spill-over allowed
        if not cpu_enabled:
            gpu_weights_scaled = dict(gpu_weights)
            total_gpu_pct = sum(gpu_weights_scaled.values()) or 100
            if total_gpu_vram >= estimated_model_vram_mb and estimated_model_vram_mb > 0:
                is_feasible = True
            elif estimated_model_vram_mb <= 0:
                is_feasible = True
            else:
                is_feasible = False
            return DistributionResult(
                gpu_weights=gpu_weights_scaled,
                cpu_weight=0,
                total_gpu_pct=total_gpu_pct,
                is_feasible=is_feasible,
            )

        # Rule 2: Enough VRAM — no CPU needed
        if estimated_model_vram_mb <= 0 or total_gpu_vram >= estimated_model_vram_mb:
            total_gpu_pct = sum(gpu_weights.values()) or 100
            return DistributionResult(
                gpu_weights=dict(gpu_weights),
                cpu_weight=0,
                total_gpu_pct=total_gpu_pct,
                is_feasible=True,
            )

        # Rule 3 & 4: VRAM insufficient, CPU enabled — spill-over
        gpu_fraction = total_gpu_vram / estimated_model_vram_mb
        gpu_fraction = min(gpu_fraction, 1.0)

        # Scale user weights proportionally to the GPU fraction
        original_total = sum(gpu_weights.values()) or 1
        gpu_total = int(round(gpu_fraction * 100))
        gpu_total = max(1, min(gpu_total, 99))

        scaled_map: Dict[int, int] = {}
        remaining = gpu_total
        gpu_indices = sorted(gpu_weights.keys())
        for pos, idx in enumerate(gpu_indices):
            if pos == len(gpu_indices) - 1:
                scaled_map[idx] = remaining
            else:
                share = int(round(gpu_weights[idx] * gpu_total / original_total))
                share = max(0, min(share, remaining - (len(gpu_indices) - pos - 1)))
                scaled_map[idx] = share
                remaining -= share

        if sum(scaled_map.values()) != gpu_total and gpu_indices:
            scaled_map[gpu_indices[0]] += gpu_total - sum(scaled_map.values())

        cpu_weight = 100 - gpu_total

        return DistributionResult(
            gpu_weights=scaled_map,
            cpu_weight=cpu_weight,
            total_gpu_pct=gpu_total,
            is_feasible=True,
        )

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
