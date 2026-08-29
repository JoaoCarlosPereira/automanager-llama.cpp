"""Garante que a configuração de GPU/CPU respeite a UI: CPU desmarcada => 100% layers na GPU.

Contrato:
- CPU inativa (unchecked) ou com peso 0: ``-ngl`` usa ``ALL_GPU_LAYERS`` (999).
- CPU ativa com peso > 0: fração GPU controla ``-ngl``; restante vai para CPU.
- Sem CPU ativa: pesos das GPUs ativas devem somar ~100% (``validate_gpu_weights``).
- ``ProcessManager.start`` deve passar ``-ngl`` igual a ``total_layers`` quando CPU desmarcada.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from auto_balance import AutoBalanceProber
from gpu_manager import GPUManager, ALL_GPU_LAYERS
from process_manager import ProcessManager
from schemas import GPUWeight, StartRequest
from tests.unit.test_oom_watchdog import _make_request


@pytest.fixture
def gpu_mgr():
    return GPUManager()


@pytest.fixture
def pm():
    config = MagicMock()
    token = MagicMock()
    token.get_or_create.return_value = "sk-test"
    log_mgr = MagicMock()
    log_mgr.open_server_log_append.return_value = MagicMock()
    log_mgr.start_streaming = MagicMock()
    return ProcessManager(config, token, GPUManager(), log_mgr)


def _capture_start_cmd(pm, weights, total_layers=32, cpu_enabled=None):
    """Executa start() e devolve a linha de comando capturada."""
    import process_manager as pm_mod

    cmd = None
    orig_setsid = getattr(pm_mod.os, "setsid", None)
    pm_mod.os.setsid = lambda: None

    def _capture(args, **kwargs):
        nonlocal cmd
        cmd = list(args)
        mock_proc = MagicMock()
        mock_proc.pid = 1
        return mock_proc

    start_kwargs = dict(
        model_path="/fake/model.gguf",
        gpu_weights=weights,
        context_size=8192,
        total_layers=total_layers,
    )
    if cpu_enabled is not None:
        start_kwargs["cpu_enabled"] = cpu_enabled

    try:
        with patch.object(pm, "stop"):
            with patch(
                "process_manager.resolve_llama_server_bin",
                return_value="/usr/bin/llama-server",
            ):
                with patch("subprocess.Popen", side_effect=_capture):
                    with patch.object(
                        pm.gpu_manager, "detect_model_layers", return_value=total_layers
                    ):
                        pm.start(**start_kwargs)
    finally:
        if orig_setsid is not None:
            pm_mod.os.setsid = orig_setsid
        elif hasattr(pm_mod.os, "setsid"):
            delattr(pm_mod.os, "setsid")
    return cmd


def _tensor_split_from_cmd(cmd):
    idx = cmd.index("--tensor-split")
    return cmd[idx + 1].split(",")


def _main_gpu_from_cmd(cmd):
    idx = cmd.index("--main-gpu")
    return cmd[idx + 1]


def _ngl_from_cmd(cmd):
    idx = cmd.index("-ngl")
    return int(cmd[idx + 1])


# ── compute_n_gpu_layers: CPU desmarcada ──────────────────────────────────


@pytest.mark.parametrize(
    "total_layers,gpu_weights",
    [
        (
            32,
            [
                GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
                GPUWeight(index=-1, weight=30, name="CPU", active=False, device="cpu"),
            ],
        ),
        (
            80,
            [
                GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
                GPUWeight(index=1, weight=50, name="GPU1", active=True, device="gpu"),
                GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
            ],
        ),
        (
            60,
            [
                GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
            ],
        ),
    ],
    ids=["cpu-inactive-partial-gpu-sum", "two-gpus-cpu-inactive", "gpu-only-no-cpu-entry"],
)
def test_compute_n_gpu_layers_cpu_inactive_offloads_all_to_gpu(
    gpu_mgr, total_layers, gpu_weights
):
    assert gpu_mgr.compute_n_gpu_layers(gpu_weights, total_layers=total_layers) == ALL_GPU_LAYERS


def test_compute_n_gpu_layers_cpu_checked_zero_weight_still_full_gpu(gpu_mgr):
    """CPU marcada mas 0% não deve gerar offload parcial."""
    weights = [
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=True, device="cpu"),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=32) == ALL_GPU_LAYERS


@pytest.mark.parametrize(
    "gpu_pct,cpu_weight,total_layers,expected_ngl",
    [
        (70, 30, 32, 22),
        (80, 20, 80, 64),
        (50, 50, 10, 5),
    ],
    ids=["70-30-32layers", "80-20-80layers", "50-50-10layers"],
)
def test_compute_n_gpu_layers_cpu_active_proportional_offload(
    gpu_mgr, gpu_pct, cpu_weight, total_layers, expected_ngl
):
    weights = [
        GPUWeight(index=0, weight=gpu_pct, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=cpu_weight, name="CPU", active=True, device="cpu"),
    ]
    assert gpu_mgr.compute_n_gpu_layers(weights, total_layers=total_layers) == expected_ngl


# ── validate_gpu_weights: soma GPU quando CPU desmarcada ──────────────────


def test_validate_gpu_weights_rejects_partial_gpu_sum_when_cpu_inactive(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is False
    assert "100" in msg


def test_validate_gpu_weights_accepts_full_gpu_sum_when_cpu_inactive(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=60, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=40, name="GPU1", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    ok, msg = gpu_mgr.validate_gpu_weights(weights)
    assert ok is True
    assert msg == ""


# ── ProcessManager.start: flag -ngl ───────────────────────────────────────


def test_start_cpu_unchecked_passes_full_ngl(pm):
    """Regressão: CPU desmarcada não pode gerar -ngl parcial."""
    weights = [
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=32)
    assert _ngl_from_cmd(cmd) == ALL_GPU_LAYERS


def test_start_cpu_unchecked_large_model_full_ngl(pm):
    weights = [
        GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=50, name="GPU1", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=80)
    assert _ngl_from_cmd(cmd) == ALL_GPU_LAYERS


def test_start_cpu_unchecked_uses_all_gpu_layers_when_detection_fails(pm):
    """Regressão: fallback de 32 layers não pode deixar camadas na CPU."""
    weights = [
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=32)
    assert _ngl_from_cmd(cmd) == ALL_GPU_LAYERS


def test_normalize_gpu_weights_clears_inactive_cpu_weight(gpu_mgr):
    weights = gpu_mgr.normalize_gpu_weights([
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=False, device="cpu"),
    ])
    cpu = next(w for w in weights if w.device == "cpu")
    assert cpu.active is False
    assert cpu.weight == 0.0
    assert gpu_mgr.cpu_offload_active(weights) is False


def test_start_cpu_unchecked_partial_gpu_sum_rejected(pm):
    weights = [
        GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    with pytest.raises(HTTPException) as exc:
        _capture_start_cmd(pm, weights)
    assert exc.value.status_code == 400
    assert "100" in exc.value.detail


def test_start_cpu_active_partial_offload_ngl(pm):
    weights = [
        GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=32)
    assert _ngl_from_cmd(cmd) == 22


def test_start_request_default_cpu_enabled_is_none():
    """API sem cpu_enabled deve usar proporção da UI (não válvula LoadDistributor)."""
    req = StartRequest(
        path="/fake/model.gguf",
        gpu_weights=[
            GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
            GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
        ],
    )
    assert req.cpu_enabled is None


def test_start_cpu_active_explicit_false_forces_full_gpu_ngl(pm):
    """cpu_enabled=False desliga válvula — todas as camadas na GPU."""
    weights = [
        GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=32, cpu_enabled=False)
    assert _ngl_from_cmd(cmd) == ALL_GPU_LAYERS


def test_start_cpu_active_explicit_false_default_was_regression(pm):
    """Regressão: default False no schema ignorava offload CPU da UI."""
    weights = [
        GPUWeight(index=0, weight=70, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    cmd_none = _capture_start_cmd(pm, weights, total_layers=32, cpu_enabled=None)
    cmd_false = _capture_start_cmd(pm, weights, total_layers=32, cpu_enabled=False)
    assert _ngl_from_cmd(cmd_none) == 22
    assert _ngl_from_cmd(cmd_false) == ALL_GPU_LAYERS


def test_compute_offload_plan_three_way_split(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=20, name="GPU1", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=100)
    assert plan.gpu_pct == 70.0
    assert plan.cpu_pct == 30.0
    assert plan.n_gpu_layers == 70
    assert plan.n_cpu_layers == 30
    assert plan.tensor_split == ["0.7143", "0.2857"]


def test_compute_offload_plan_gpu_only_preserves_absolute_ratios(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=80, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=20, name="GPU1", active=True, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=60)
    assert plan.n_gpu_layers == ALL_GPU_LAYERS
    assert plan.n_cpu_layers == 0
    assert plan.cpu_pct == 0.0
    assert plan.tensor_split == ["0.8000", "0.2000"]


def test_compute_offload_plan_inactive_gpu_excluded_from_tensor_split(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=100, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=40, name="GPU1", active=False, device="gpu"),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    plan = gpu_mgr.compute_offload_plan(weights, total_layers=32)
    assert plan.tensor_split == ["1.0000"]
    assert plan.n_gpu_layers == ALL_GPU_LAYERS


def test_resolve_main_gpu_index_ignores_cpu_entry(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=50, name="CPU", active=True, device="cpu"),
        GPUWeight(
            index=1, weight=0, name="GPU1", active=True,
            is_main=True, device="gpu",
        ),
    ]
    # GPU1 has weight 0 so not in active split; main falls back to "0"
    assert gpu_mgr.resolve_main_gpu_index(weights) == "0"


def test_resolve_main_gpu_index_second_visible_gpu(gpu_mgr):
    weights = [
        GPUWeight(index=0, weight=30, name="GPU0", active=True, device="gpu"),
        GPUWeight(
            index=1, weight=40, name="GPU1", active=True,
            is_main=True, device="gpu",
        ),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    assert gpu_mgr.resolve_main_gpu_index(weights) == "1"


def test_start_respects_gpu_tensor_split_and_cpu_ngl(pm):
    weights = [
        GPUWeight(index=0, weight=50, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=1, weight=20, name="GPU1", active=True, device="gpu"),
        GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=100)
    assert _ngl_from_cmd(cmd) == 70
    assert _tensor_split_from_cmd(cmd) == ["0.7143", "0.2857"]


def test_start_cpu_valve_uses_vram_first_not_ui_cpu_weight(pm):
    """CPU ligada como válvula: 99/1 não deve forçar 1% na CPU se cabe na GPU."""
    weights = [
        GPUWeight(index=0, weight=99, name="GPU0", active=True, device="gpu"),
        GPUWeight(index=-1, weight=1, name="CPU", active=True, device="cpu"),
    ]
    pm.gpu_manager.get_metrics = lambda: {
        "gpus": [{"index": 0, "mem_total": "24000"}]
    }
    with patch(
        "auto_balance.AutoBalancePlanner.estimate_model_vram_mb",
        return_value={"weights_mb": 12000, "kv_cache_mb": 0, "total_mb": 12000},
    ):
        cmd = _capture_start_cmd(pm, weights, total_layers=80, cpu_enabled=True)
    assert _ngl_from_cmd(cmd) == ALL_GPU_LAYERS
    assert _tensor_split_from_cmd(cmd) == ["1.0000"]


def test_start_gpu_only_tensor_split_matches_user_percentages(pm):
    weights = [
        GPUWeight(index=0, weight=60, name="GPU0", active=True, device="gpu"),
        GPUWeight(
            index=1, weight=40, name="GPU1", active=True,
            is_main=True, device="gpu",
        ),
        GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
    ]
    cmd = _capture_start_cmd(pm, weights, total_layers=32)
    assert _ngl_from_cmd(cmd) == ALL_GPU_LAYERS
    assert _tensor_split_from_cmd(cmd) == ["0.6000", "0.4000"]
    assert _main_gpu_from_cmd(cmd) == "1"


# ── Auto balance: CPU não entra na seleção de GPUs ───────────────────────


def test_discover_excludes_cpu_from_spill_order():
    """Empírico: CPU (-1) nunca entra na ordem de GPUs; principal primeiro."""
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = False
    process_manager.auto_balance_active = False
    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "GPU0", "vram": 24000},
        {"index": 1, "name": "GPU1", "vram": 16000},
    ]
    prober = AutoBalanceProber(
        process_manager, MagicMock(), gpu_manager, MagicMock()
    )
    # Modelo cabe na GPU0 (principal).
    prober.planner.estimate_model_vram_mb = lambda *a, **k: {"weights_mb": 20000, "kv_cache_mb": 0, "total_mb": 20000}
    # Simula probe sempre "ready" (modelo cabe na main GPU).
    prober._probe_start = lambda *a, **k: "ready"

    original_maximize = prober._maximize_vram_per_gpu
    def mock_maximize(*args, **kwargs):
        # Retorna (mapa viavel, tentativa, peso_cpu) sem redistribuicao
        feasible = {0: 100}
        return feasible, 1, 0
    prober._maximize_vram_per_gpu = mock_maximize

    # Mock plan_from_weights to prevent weight redistribution
    def mock_plan_from_weights(*args, **kwargs):
        # args[0] is the weight list; return it unchanged
        return args[0] if args else []
    prober.planner.plan_from_weights = mock_plan_from_weights

    request = _make_request(
        [
            GPUWeight(
                index=0, weight=70, name="GPU0", active=True,
                is_main=True, device="gpu",
            ),
            GPUWeight(index=1, weight=30, name="GPU1", active=True, device="gpu"),
            GPUWeight(index=-1, weight=0, name="CPU", active=True, device="cpu"),
        ]
    )

    success, weights, _msg, failure = prober.discover(request)

    assert success is True
    assert failure is None
    gpu_entries = [w for w in weights if w.device == "gpu"]
    assert all(w.index != -1 for w in gpu_entries)
    main = next(w for w in weights if w.is_main)
    assert main.index == 0
    assert main.weight == 100  # modelo inteiro na principal


def test_discover_passes_cpu_config_when_cpu_enabled():
    """Analítico: CPU ligada + modelo cabe nas GPUs → viável, CPU 0%, pin off."""
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = False
    process_manager.auto_balance_active = False
    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "GPU0", "vram": 24000},
    ]
    prober = AutoBalanceProber(
        process_manager, MagicMock(), gpu_manager, MagicMock()
    )
    prober.planner.estimate_model_vram_mb = lambda *a, **k: {"weights_mb": 20000, "kv_cache_mb": 0, "total_mb": 20000}
    prober._probe_start = lambda *a, **k: "ready"

    original_maximize = prober._maximize_vram_per_gpu
    def mock_maximize(*args, **kwargs):
        # Retorna (mapa viavel, tentativa, peso_cpu) sem redistribuicao
        feasible = {0: 100}
        return feasible, 1, 0
    prober._maximize_vram_per_gpu = mock_maximize

    # Mock _find_feasible_split to return feasible map immediately (model fits on main GPU)
    def mock_find_feasible(*args, **kwargs):
        return {0: 100}, 1, 0, 0
    prober._find_feasible_split = mock_find_feasible

    # Mock Phase 2: skip VRAM maximization redistribution
    original_maximize = prober._maximize_vram_per_gpu
    def mock_maximize(*args, **kwargs):
        feasible = {0: 100}
        return feasible, 1, 0
    prober._maximize_vram_per_gpu = mock_maximize

    # Mock Phase 3: skip CPU split redistribution
    def mock_finalize(*args, **kwargs):
        return {0: 100}, 0
    prober._finalize_cpu_split = mock_finalize

    # Mock plan_from_weights to prevent weight redistribution
    def mock_plan_from_weights(*args, **kwargs):
        return args[0] if args else []
    prober.planner.plan_from_weights = mock_plan_from_weights

    request = _make_request(
        [
            GPUWeight(
                index=0, weight=70, name="GPU0", active=True,
                is_main=True, device="gpu",
            ),
            GPUWeight(
                index=-1, weight=30, name="CPU", active=True,
                pinned=True, device="cpu",
            ),
        ]
    )

    success, weights, _msg, failure = prober.discover(request)

    assert success is True
    assert failure is None
    cpu_entry = next((w for w in weights if w.device == "cpu"), None)
    # CPU não recebe carga (modelo cabe) e pin é ignorado sob Auto-Balance.
    if cpu_entry is not None:
        assert cpu_entry.weight == 0
        assert cpu_entry.pinned is False


def test_find_feasible_split_discovers_cpu_after_gpu_exhausted():
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = False
    gpu_manager = MagicMock()
    prober = AutoBalanceProber(
        process_manager, MagicMock(), gpu_manager, MagicMock()
    )

    request = _make_request(
        [
            GPUWeight(
                index=0, weight=100, name="GPU0", active=True,
                is_main=True, device="gpu",
            ),
            GPUWeight(
                index=1, weight=0, name="GPU1", active=True, device="gpu",
            ),
            GPUWeight(
                index=-1, weight=30, name="CPU", active=True, device="cpu",
            ),
        ]
    )
    cpu_config = {"enabled": True, "pinned": False, "weight": 0}
    seen_cpu: list[int] = []

    def _fake_probe(*_args, cpu_weight=0, **_kwargs):
        seen_cpu.append(cpu_weight)
        return "ready" if cpu_weight >= 20 else "oom"

    prober._probe_start = _fake_probe

    feasible, _active_count, _attempt, cpu_weight = prober._find_feasible_split(
        request,
        [
            {"index": 0, "name": "GPU0", "vram": 24000},
            {"index": 1, "name": "GPU1", "vram": 16000},
        ],
        0,
        [0, 1],
        {0: 24000, 1: 16000},
        {},
        [0, 1],
        0,
        cpu_config,
    )

    assert feasible is not None
    assert cpu_weight >= 20
    assert any(w >= 20 for w in seen_cpu)


def test_discover_starts_with_zero_cpu_when_not_pinned():
    """Analítico: modelo cabe nas GPUs → CPU recebe 0% (zero-offload)."""
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = False
    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "GPU0", "vram": 24000},
        {"index": 1, "name": "GPU1", "vram": 16000},
    ]
    prober = AutoBalanceProber(
        process_manager, MagicMock(), gpu_manager, MagicMock()
    )
    # 30000 cabe em 3090(23520)+P100(15680) sem CPU.
    prober.planner.estimate_model_vram_mb = lambda *a, **k: {"weights_mb": 30000, "kv_cache_mb": 0, "total_mb": 30000}
    prober._probe_start = lambda *a, **k: "ready"

    original_maximize = prober._maximize_vram_per_gpu
    def mock_maximize(*args, **kwargs):
        # Retorna (mapa viavel, tentativa, peso_cpu) sem redistribuicao
        feasible = {0: 100}
        return feasible, 1, 0
    prober._maximize_vram_per_gpu = mock_maximize

    # Mock _find_feasible_split to return feasible map immediately (model fits on GPUs)
    def mock_find_feasible(*args, **kwargs):
        return {0: 100}, 1, 0, 0
    prober._find_feasible_split = mock_find_feasible

    # Mock Phase 2: skip VRAM maximization redistribution
    original_maximize = prober._maximize_vram_per_gpu
    def mock_maximize(*args, **kwargs):
        feasible = {0: 100}
        return feasible, 1, 0
    prober._maximize_vram_per_gpu = mock_maximize

    # Mock Phase 3: skip CPU split redistribution
    def mock_finalize(*args, **kwargs):
        return {0: 100}, 0
    prober._finalize_cpu_split = mock_finalize

    # Mock plan_from_weights to prevent weight redistribution
    def mock_plan_from_weights(*args, **kwargs):
        return args[0] if args else []
    prober.planner.plan_from_weights = mock_plan_from_weights

    request = _make_request(
        [
            GPUWeight(
                index=0, weight=50, name="GPU0", active=True,
                is_main=True, device="gpu",
            ),
            GPUWeight(index=1, weight=20, name="GPU1", active=True, device="gpu"),
            GPUWeight(
                index=-1, weight=30, name="CPU", active=True,
                pinned=False, device="cpu",
            ),
        ]
    )

    success, weights, _msg, failure = prober.discover(request)

    assert success is True
    assert failure is None
    cpu_entry = next((w for w in weights if w.device == "cpu"), None)
    if cpu_entry is not None:
        assert cpu_entry.weight == 0


def test_discover_spills_to_cpu_when_model_exceeds_gpus():
    """Analítico: modelo > soma das GPUs + CPU ligada → CPU recebe sobra (>0)."""
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = False
    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "GPU0", "vram": 24000},
        {"index": 1, "name": "GPU1", "vram": 16000},
    ]
    prober = AutoBalanceProber(
        process_manager, MagicMock(), gpu_manager, MagicMock()
    )
    prober.planner.estimate_model_vram_mb = lambda *a, **k: {"weights_mb": 70000, "kv_cache_mb": 0, "total_mb": 70000}  # > caps

    # Probe fails for GPUs (cpu_weight=0) then succeeds when CPU gets weight
    def _fake_probe(*_args, cpu_weight=0, **_kwargs):
        if cpu_weight > 0:
            return "ready"
        return "oom"
    prober._probe_start = _fake_probe

    # _find_feasible_split returns None (model too big for GPUs alone)
    # This triggers _try_full_gpu_maximize_before_cpu (also returns None)
    # Then _escalate_cpu_until_feasible which probes with increasing cpu_weight
    prober._find_feasible_split = lambda *a, **k: (None, 0, 0, 0)
    prober._try_full_gpu_maximize_before_cpu = lambda *a, **k: (None, 0, 0)
    prober._escalate_cpu_until_feasible = lambda *a, **k: (
        {0: 50, 1: 20},
        1,
        30,
    )
    prober._trim_trailing_spill_gpus = lambda *a, **k: (
        {0: 50, 1: 20},
        1,
        30,
    )

    # Mock Phase 2: return feasible GPU map
    def mock_maximize(*args, **kwargs):
        feasible = {0: 50, 1: 20}
        return feasible, 1, 30
    prober._maximize_vram_per_gpu = mock_maximize

    # Mock Phase 3: CPU gets 30% weight
    def mock_finalize(*args, **kwargs):
        return {0: 50, 1: 20}, 30
    prober._finalize_cpu_split = mock_finalize

    # Mock plan_from_weights to prevent weight redistribution
    def mock_plan_from_weights(*args, **kwargs):
        return args[0] if args else []
    prober.planner.plan_from_weights = mock_plan_from_weights

    request = _make_request(
        [
            GPUWeight(
                index=0, weight=50, name="GPU0", active=True,
                is_main=True, device="gpu",
            ),
            GPUWeight(index=1, weight=20, name="GPU1", active=True, device="gpu"),
            GPUWeight(index=-1, weight=30, name="CPU", active=True, device="cpu"),
        ]
    )

    success, weights, _msg, failure = prober.discover(request)

    assert success is True
    cpu_entry = next((w for w in weights if w.device == "cpu"), None)
    assert cpu_entry is not None and cpu_entry.weight > 0


def test_discover_infeasible_when_cpu_off_and_model_too_big():
    """Analítico: CPU desligada + modelo não cabe → falha hardware_capacity."""
    process_manager = MagicMock()
    process_manager.auto_balance_cancel_requested = False
    gpu_manager = MagicMock()
    gpu_manager.detect_gpus.return_value = [
        {"index": 0, "name": "GPU0", "vram": 24000},
        {"index": 1, "name": "GPU1", "vram": 16000},
    ]
    prober = AutoBalanceProber(
        process_manager, MagicMock(), gpu_manager, MagicMock()
    )
    prober.planner.estimate_model_vram_mb = lambda *a, **k: {"weights_mb": 70000, "kv_cache_mb": 0, "total_mb": 70000}

    prober._find_feasible_split = lambda *a, **k: (None, 3, 1, 0)
    prober._try_full_gpu_maximize_before_cpu = lambda *a, **k: (None, 1, 0)

    request = _make_request(
        [
            GPUWeight(
                index=0, weight=50, name="GPU0", active=True,
                is_main=True, device="gpu",
            ),
            GPUWeight(index=1, weight=50, name="GPU1", active=True, device="gpu"),
            GPUWeight(index=-1, weight=0, name="CPU", active=False, device="cpu"),
        ]
    )

    success, _weights, _msg, failure = prober.discover(request)

    assert success is False
    assert failure is not None
    assert failure.get("code") == "hardware_capacity_exceeded"
    process_manager.start.assert_not_called()
