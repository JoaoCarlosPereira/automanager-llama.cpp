
import pytest
from schemas import StartRequest, GPUWeight, DEFAULT_CACHE_TYPE, CACHE_TYPE_PRESETS, DEFAULT_FLASH_ATTN_ENABLED
from auto_balance import AutoBalancePlanner
from process_manager import ProcessManager
from unittest.mock import MagicMock, patch

def test_start_request_cache_type_defaults():
    req = StartRequest(
        path="/models/test.gguf",
        gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}]
    )
    assert req.cache_type_k == DEFAULT_CACHE_TYPE
    assert req.cache_type_v == DEFAULT_CACHE_TYPE
    assert req.ubatch_size == 512
    assert req.numa_enabled is False
    assert req.flash_attn_enabled is DEFAULT_FLASH_ATTN_ENABLED
    assert req.threads == 0

def test_start_request_custom_performance_opts():
    req = StartRequest(
        path="/models/test.gguf",
        gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        ubatch_size=128,
        numa_enabled=True,
        threads=8,
        threads_batch=16
    )
    assert req.cache_type_k == "q8_0"
    assert req.cache_type_v == "q4_0"
    assert req.ubatch_size == 128
    assert req.numa_enabled is True
    assert req.threads == 8
    assert req.threads_batch == 16

def test_estimate_vram_with_different_cache_types():
    # context 1024, 1 slot. Formula: 1024 * 1 * 0.1 = 102.4 MB for f16
    # f16, f16 -> mult 1.0 -> 102 MB
    est_f16 = AutoBalancePlanner.estimate_model_vram_mb("any.gguf", 1024, 1, "f16", "f16")
    assert est_f16["kv_cache_mb"] == 102
    
    # q8_0, q8_0 -> mult 0.5 -> 51 MB
    est_q8 = AutoBalancePlanner.estimate_model_vram_mb("any.gguf", 1024, 1, "q8_0", "q8_0")
    assert est_q8["kv_cache_mb"] == 51
    
    # q4_0, q4_0 -> mult 0.25 -> 25 MB
    est_q4 = AutoBalancePlanner.estimate_model_vram_mb("any.gguf", 1024, 1, "q4_0", "q4_0")
    assert est_q4["kv_cache_mb"] == 25

@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_includes_performance_flags(mock_exists, mock_bin):
    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.return_value = [{"index": 0, "weight": 100, "active": True, "name": "GPU0"}]
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0"
    gpu_mgr.detect_model_layers.return_value = 32
    
    cpu_info = MagicMock()
    cpu_info.physical_cores = 8
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    
    plan = MagicMock()
    plan.tensor_split = ["1.0"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "0"
    
    log_mgr = MagicMock()
    
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)
    
    req = StartRequest(
        path="/models/test.gguf",
        gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        ubatch_size=256,
        numa_enabled=True,
        threads=4,
        threads_batch=8
    )
    
    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path=req.path,
            gpu_weights=req.gpu_weights,
            context_size=req.context_size,
            cache_type_k=req.cache_type_k,
            cache_type_v=req.cache_type_v,
            ubatch_size=req.ubatch_size,
            numa_enabled=req.numa_enabled,
            threads=req.threads,
            threads_batch=req.threads_batch
        )
        args = mock_popen.call_args[0][0]
        
        assert "--cache-type-k" in args
        assert "q8_0" in args
        assert args[args.index("--cache-type-k") + 1] == "q8_0"
        
        assert "--ubatch-size" in args
        assert "256" in args
        
        assert "--numa" in args
        assert args[args.index("--numa") + 1] == "isolate"
        
        assert "--threads" in args
        assert "4" in args
        
        assert "--threads-batch" in args
        assert "8" in args
        
        assert "--flash-attn" in args
        assert args[args.index("--flash-attn") + 1] == "on"

@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_flash_attn_disabled(mock_exists, mock_bin):
    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.return_value = [{"index": 0, "weight": 100, "active": True, "name": "GPU0"}]
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0"
    gpu_mgr.detect_model_layers.return_value = 32
    cpu_info = MagicMock()
    cpu_info.physical_cores = 8
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    plan = MagicMock()
    plan.tensor_split = ["1.0"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "0"
    log_mgr = MagicMock()
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)

    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path="/models/test.gguf",
            gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
            context_size=65536,
            flash_attn_enabled=False,
        )
        args = mock_popen.call_args[0][0]
        assert "--flash-attn" not in args

@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_omits_mmproj_flags_when_vision_disabled(mock_exists, mock_bin):
    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.return_value = [{"index": 0, "weight": 100, "active": True, "name": "GPU0"}]
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0"
    gpu_mgr.detect_model_layers.return_value = 32
    cpu_info = MagicMock()
    cpu_info.physical_cores = 8
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    plan = MagicMock()
    plan.tensor_split = ["1.0"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "0"
    log_mgr = MagicMock()
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)

    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path="/models/test.gguf",
            gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
            context_size=65536,
            mmproj_disabled=True,
        )
        args = mock_popen.call_args[0][0]
        assert "--no-mmproj" not in args
        assert "--mmproj" not in args
        assert "--mmproj-auto" not in args

@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_numa_distribute_for_multi_gpu(mock_exists, mock_bin):
    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.side_effect = lambda weights: weights
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0,1"
    gpu_mgr.detect_model_layers.return_value = 32
    cpu_info = MagicMock()
    cpu_info.physical_cores = 8
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    plan = MagicMock()
    plan.tensor_split = ["0.6000", "0.4000"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "1"
    log_mgr = MagicMock()
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)

    gpu_weights = [
        GPUWeight(index=0, weight=60, name="GPU0", active=True, is_main=False, device="gpu"),
        GPUWeight(index=1, weight=40, name="GPU1", active=True, is_main=True, device="gpu"),
    ]

    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path="/models/test.gguf",
            gpu_weights=gpu_weights,
            context_size=65536,
            numa_enabled=True,
        )
        args = mock_popen.call_args[0][0]
        assert args[args.index("--numa") + 1] == "distribute"

@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_omits_verbose_logging(mock_exists, mock_bin):
    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.return_value = [{"index": 0, "weight": 100, "active": True, "name": "GPU0"}]
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0"
    gpu_mgr.detect_model_layers.return_value = 32
    cpu_info = MagicMock()
    cpu_info.physical_cores = 8
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    plan = MagicMock()
    plan.tensor_split = ["1.0"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "0"
    log_mgr = MagicMock()
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)

    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path="/models/test.gguf",
            gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
            context_size=65536,
        )
        args = mock_popen.call_args[0][0]
        assert "--verbose" not in args
        assert "--log-verbosity" not in args
        log_mgr.start_streaming.assert_called_once()
        assert log_mgr.start_streaming.call_args.kwargs.get("cmd") == args

@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_auto_threads(mock_exists, mock_bin):
    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.return_value = [{"index": 0, "weight": 100, "active": True, "name": "GPU0"}]
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0"
    gpu_mgr.detect_model_layers.return_value = 32
    
    cpu_info = MagicMock()
    cpu_info.physical_cores = 12
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    
    plan = MagicMock()
    plan.tensor_split = ["1.0"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "0"
    
    log_mgr = MagicMock()
    
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)
    
    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path="/models/test.gguf",
            gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
            context_size=1024,
            threads=0 # Auto
        )
        args = mock_popen.call_args[0][0]
        
        assert "--threads" in args
        assert "12" in args


@patch("process_manager.supports_cli_flag")
@patch("process_manager.resolve_llama_server_bin", return_value="/usr/bin/llama-server")
@patch("process_manager.os.path.exists", return_value=True)
def test_process_manager_includes_kv_unified_when_supported(mock_exists, mock_bin, mock_supports):
    mock_supports.side_effect = lambda flag, _bin: flag == "--kv-unified"

    config_mgr = MagicMock()
    token_mgr = MagicMock()
    token_mgr.get_or_create.return_value = "test-token"
    gpu_mgr = MagicMock()
    gpu_mgr.normalize_gpu_weights.return_value = [{"index": 0, "weight": 100, "active": True, "name": "GPU0"}]
    gpu_mgr.validate_gpu_weights.return_value = (True, "")
    gpu_mgr.get_visible_devices.return_value = "0"
    gpu_mgr.detect_model_layers.return_value = 32
    cpu_info = MagicMock()
    cpu_info.physical_cores = 8
    gpu_mgr.detect_cpu_info.return_value = cpu_info
    plan = MagicMock()
    plan.tensor_split = ["1.0"]
    plan.n_gpu_layers = 32
    plan.gpu_pct = 100.0
    gpu_mgr.compute_offload_plan.return_value = plan
    gpu_mgr.resolve_main_gpu_index.return_value = "0"
    log_mgr = MagicMock()
    pm = ProcessManager(config_mgr, token_mgr, gpu_mgr, log_mgr)

    with patch("process_manager.subprocess.Popen") as mock_popen:
        pm.start(
            model_path="/models/test.gguf",
            gpu_weights=[{"index": 0, "weight": 100, "name": "GPU0", "active": True, "is_main": True}],
            context_size=65536,
        )
        args = mock_popen.call_args[0][0]
        assert "--kv-unified" in args
