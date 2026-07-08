
import threading
from unittest.mock import MagicMock, patch
import pytest
from auto_balance import AutoBalanceProber, AutoBalancePlanner
from schemas import StartRequest, GPUWeight
from process_manager import ProcessManager

THREE_GPU_HARDWARE = [
    {"index": 0, "name": "Tesla P100-PCIE-16GB", "vram": 16384},
    {"index": 1, "name": "Tesla P100-PCIE-16GB", "vram": 16384},
    {"index": 2, "name": "NVIDIA GeForce RTX 3090", "vram": 24576},
]

def _make_request(weights, **kwargs):
    req = StartRequest(
        path="/models/test.gguf",
        gpu_weights=weights,
        context_size=8192,
        parallel_slots=1,
        **kwargs
    )
    return req

class TestSmartAutoBalance:
    def _prober(self):
        pm = MagicMock()
        pm._auto_balance_cancel = False
        config = MagicMock()
        gpu_mgr = MagicMock()
        log_mgr = MagicMock()
        prober = AutoBalanceProber(pm, config, gpu_mgr, log_mgr)
        return prober, pm

    @patch("auto_balance.AutoBalancePlanner.estimate_model_vram_mb")
    def test_generate_smart_proposal_respects_pins(self, mock_est):
        prober, _ = self._prober()
        prober.gpu_manager.detect_gpus.return_value = THREE_GPU_HARDWARE
        prober.gpu_manager.detect_cpu_info.return_value.physical_cores = 8
        
        # Scenario: Tight VRAM, Cache is NOT pinned. Algorithm should suggest q4_0.
        mock_est.return_value = {
            "weights_mb": 24000, 
            "kv_cache_mb": 1000, 
            "total_mb": 25000
        }
        
        weights = [
            GPUWeight(index=2, weight=100, name="3090", active=True, is_main=True)
        ]
        
        # 1. Cache NOT pinned -> should change to q4_0 (heuristic: free < 500)
        req = _make_request(weights, smart_calibration=True, pinned_fields={"cache_type": False})
        proposal = prober._generate_smart_proposal(req, weights)
        assert proposal["cache_type_k"] == "q4_0"
        
        # 2. Cache IS pinned -> should stay f16
        req_pinned = _make_request(weights, smart_calibration=True, pinned_fields={"cache_type": True})
        proposal_pinned = prober._generate_smart_proposal(req_pinned, weights)
        assert proposal_pinned["cache_type_k"] == "f16"

    @patch("auto_balance.AutoBalancePlanner.estimate_model_vram_mb")
    def test_smart_proposal_threads_heuristic(self, mock_est):
        prober, _ = self._prober()
        prober.gpu_manager.detect_gpus.return_value = THREE_GPU_HARDWARE
        prober.gpu_manager.detect_cpu_info.return_value.physical_cores = 16
        
        mock_est.return_value = {"weights_mb": 1000, "kv_cache_mb": 100, "total_mb": 1100}
        weights = [GPUWeight(index=2, weight=100, name="3090", active=True, is_main=True)]

        # Threads NOT pinned -> suggest physical cores
        req = _make_request(weights, smart_calibration=True, pinned_fields={"threads": False})
        proposal = prober._generate_smart_proposal(req, weights)
        assert proposal["threads"] == 16
        
        # Threads IS pinned -> keep original (0)
        req_pinned = _make_request(weights, smart_calibration=True, pinned_fields={"threads": True}, threads=0)
        proposal_pinned = prober._generate_smart_proposal(req_pinned, weights)
        assert proposal_pinned["threads"] == 0

    def test_available_vram_subtracts_other_instances(self):
        prober, _ = self._prober()
        prober.gpu_manager.get_metrics.return_value = {
            "gpus": [
                {"index": 0, "mem_used": "8000", "mem_total": "16384"},
                {"index": 2, "mem_used": "300", "mem_total": "24576"},
            ]
        }
        totals = {0: 16384, 1: 16384, 2: 24576}
        available = prober._available_vram_by_index(totals)
        assert available == {0: 8384, 1: 16384, 2: 24276}

    def test_available_vram_falls_back_to_total_on_metrics_error(self):
        prober, _ = self._prober()
        prober.gpu_manager.get_metrics.side_effect = RuntimeError("nvidia-smi indisponivel")
        totals = {0: 16384, 1: 16384}
        assert prober._available_vram_by_index(totals) == totals

    @patch("auto_balance.AutoBalancePlanner.estimate_model_vram_mb")
    def test_smart_proposal_uses_available_vram_from_calibration_start(self, mock_est):
        prober, _ = self._prober()
        prober.gpu_manager.detect_gpus.return_value = THREE_GPU_HARDWARE
        prober.gpu_manager.detect_cpu_info.return_value.physical_cores = 8
        mock_est.return_value = {
            "weights_mb": 20000, "kv_cache_mb": 1000, "total_mb": 21000
        }
        weights = [
            GPUWeight(index=2, weight=100, name="3090", active=True, is_main=True)
        ]
        req = _make_request(
            weights, smart_calibration=True, pinned_fields={"cache_type": False}
        )

        # Sem outra instância: 24576 total - 21000 usados = 3576 livres -> mantém f16
        proposal = prober._generate_smart_proposal(req, weights)
        assert proposal["cache_type_k"] == "f16"

        # Outra instância ocupava 4000 MB no início da calibração:
        # 20576 disponíveis - 21000 = folga negativa -> sugere q4_0
        prober._initial_available_vram = {2: 20576}
        proposal = prober._generate_smart_proposal(req, weights)
        assert proposal["cache_type_k"] == "q4_0"

    @patch("auto_balance.AutoBalanceProber._discover_empirical")
    @patch("auto_balance.AutoBalanceProber._generate_smart_proposal")
    def test_discover_returns_tuple_with_proposal(self, mock_gen, mock_disc):
        prober, _ = self._prober()
        weights = [GPUWeight(index=2, weight=100, name="3090", active=True, is_main=True)]
        mock_disc.return_value = (True, weights, "Success", None)
        mock_gen.return_value = {"batch_size": 4096}
        
        req = _make_request(weights, smart_calibration=True)
        ok, res_weights, msg, result_data = prober.discover(req)
        
        assert ok is True
        assert result_data["proposal"] == {"batch_size": 4096}
        assert res_weights == weights

def test_process_manager_handles_smart_calibration_flow():
    config = MagicMock()
    token_mgr = MagicMock()
    gpu_mgr = MagicMock()
    log_mgr = MagicMock()
    pm = ProcessManager(config, token_mgr, gpu_mgr, log_mgr)

    weights = [GPUWeight(index=0, weight=100, name="G", active=True, is_main=True)]
    req = _make_request(weights, smart_calibration=True)

    # Patch _get_auto_balance_types in process_manager (where the cached class
    # is returned) so the prober returns our fake success result instead of
    # spawning real llama-server processes.  Also mock stop() so that the
    # internal self.stop() call (which calls _wait_port_released) does not
    # block on an actual port.
    with patch("process_manager._get_auto_balance_types") as mock_get_types:
        mock_prober_class = MagicMock()
        prober_instance = mock_prober_class.return_value
        prober_instance.discover.return_value = (
            True, weights, "Done", {"proposal": {"threads": 8}}
        )
        mock_get_types.return_value = (mock_prober_class, MagicMock)

        with patch.object(pm, "stop"):
            pm._run_auto_balance(req)

        # Verify recovery_state
        state = pm.recovery_state
        assert state["active"] is False
        assert state.get("smart_calibration") is True
        assert state.get("smart_proposal") == {"threads": 8}

        # IMPORTANT: Verify that config.update_model_settings was NOT called
        # (user must apply manually)
        config.update_model_settings.assert_not_called()


def test_auto_balance_stops_same_model_instance_before_probe():
    """Instância antiga do MESMO modelo deve ser parada antes da sondagem,
    senão sua VRAM distorce a calibração (regressão da era multi-instância)."""
    config = MagicMock()
    token_mgr = MagicMock()
    gpu_mgr = MagicMock()
    log_mgr = MagicMock()
    pm = ProcessManager(config, token_mgr, gpu_mgr, log_mgr)

    weights = [GPUWeight(index=0, weight=100, name="G", active=True, is_main=True)]
    req = _make_request(weights, smart_calibration=True)

    old_instance_req = MagicMock()
    old_instance_req.path = req.path
    other_model_req = MagicMock()
    other_model_req.path = "/models/other.gguf"
    pm._requests = {8085: old_instance_req, 8086: other_model_req}

    with patch("process_manager._get_auto_balance_types") as mock_get_types:
        mock_prober_class = MagicMock()
        mock_prober_class.return_value.discover.return_value = (
            True, weights, "Done", {"proposal": {}}
        )
        mock_get_types.return_value = (mock_prober_class, MagicMock)

        with patch.object(pm, "stop") as mock_stop, \
                patch("process_manager.time.sleep"):
            pm._run_auto_balance(req)

    stopped_ports = [c.args[0] for c in mock_stop.call_args_list if c.args]
    assert 8085 in stopped_ports
    assert 8086 not in stopped_ports
