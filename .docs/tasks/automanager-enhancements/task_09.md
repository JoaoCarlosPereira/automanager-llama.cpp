---
status: pending
title: GPU strict enforcement via CUDA_VISIBLE_DEVICES
type: backend
complexity: high
dependencies:
  - task_08
---

# GPU strict enforcement via CUDA_VISIBLE_DEVICES

## Overview

Implement GPU strict enforcement so that `llama-server` cannot see or use GPUs that are not explicitly marked as active. This implements PRD Feature 1 and TechSpec Section 5.1. Uses `CUDA_VISIBLE_DEVICES` environment variable to hide disabled GPUs at the OS driver level, and recomputes the tensor-split array proportionally.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 5.1.
- Focus on WHAT: disabled GPUs must be invisible to llama-server.
- Every change MUST have corresponding tests.
- This is the most critical feature — incorrect implementation breaks GPU allocation.

## Requirements

1. MUST add `compute_tensor_split()` method to `GPUManager` in `gpu_manager.py`.
2. MUST add `get_visible_devices()` method to `GPUManager` in `gpu_manager.py`.
3. MUST add `validate_gpu_weights()` method to `GPUManager` in `gpu_manager.py`.
4. `compute_tensor_split()` MUST filter inactive GPUs and normalize weights proportionally.
5. `get_visible_devices()` MUST return comma-separated GPU indices or `None` if no active GPUs.
6. `validate_gpu_weights()` MUST return `(True, "")` for valid weights or `(False, "error message")` for invalid.
7. MUST modify `ProcessManager.start()` to call `gpu_manager.get_visible_devices()` and set `CUDA_VISIBLE_DEVICES` env var.
8. MUST modify `ProcessManager.start()` to call `gpu_manager.compute_tensor_split()` instead of the current split calculation.
9. MUST reject start if no GPUs are active with a clear error message.
10. MUST NOT break existing GPU detection logic (GPUDetector.detect_gpus is unchanged).

## Subtasks

- [ ] Add GPUInfo dataclass to gpu_manager.py (if not already present from task_02)
- [ ] Implement GPUManager.compute_tensor_split() — filter active, normalize proportionally
- [ ] Implement GPUManager.get_visible_devices() — return CUDA_VISIBLE_DEVICES string
- [ ] Implement GPUManager.validate_gpu_weights() — return validity tuple
- [ ] Update ProcessManager.start() to use gpu_manager.compute_tensor_split()
- [ ] Update ProcessManager.start() to use gpu_manager.get_visible_devices() for CUDA_VISIBLE_DEVICES
- [ ] Update ProcessManager.start() to reject start when no GPUs active
- [ ] Update OOMWatchdog to work with new tensor split format
- [ ] Write unit tests for compute_tensor_split (5 test cases)
- [ ] Write unit tests for get_visible_devices (4 test cases)
- [ ] Write unit tests for validate_gpu_weights (3 test cases)
- [ ] Write integration test: POST /start with CUDA_VISIBLE_DEVICES verification

## Implementation Details

### File Paths to Create
- None (all changes to existing files)

### File Paths to Modify
- `gpu_manager.py` — add 3 new methods to GPUManager
- `process_manager.py` — update ProcessManager.start() to use GPUManager methods
- `tests/unit/test_gpu_manager_new.py` — new test file

### Integration Points
- `ProcessManager.start()` → uses GPUManager methods for split calculation
- `ProcessManager.start()` → sets env["CUDA_VISIBLE_DEVICES"]
- `OOMWatchdog._handle_oom()` → may need update for new split format

### Relevant Files
- `gpu_manager.py` — new methods
- `process_manager.py` — updated start() method
- `llama_manager.py` POST /start route (line 1104) — no change needed
- `design/js/scripts.js` — not relevant

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-003](adrs/adr-003.md) — GPU Strict Enforcement via CUDA_VISIBLE_DEVICES

## Deliverables

- 3 new GPUManager methods
- Updated ProcessManager.start()
- 12+ new unit tests
- 1 integration test
- All existing tests still passing

## Tests

### Unit Tests
- [ ] `test_compute_tensor_split_single_active_gpu` — GPU with weight 100 → split [1.0]
- [ ] `test_compute_tensor_split_two_active_gpus` — GPUs with weights 80, 20 → split [0.8, 0.2]
- [ ] `test_compute_tensor_split_inactive_gpu_excluded` — GPU 0 inactive, GPU 1 weight 50 → split [1.0]
- [ ] `test_compute_tensor_split_all_inactive` — all GPUs inactive → split []
- [ ] `test_compute_tensor_split_weight_normalization` — GPUs 60, 40 → split [0.6, 0.4]

- [ ] `test_get_visible_devices_single_gpu` — GPU 0 active → "0"
- [ ] `test_get_visible_devices_multiple_gpus` — GPU 0 and 2 active → "0,2"
- [ ] `test_get_visible_devices_no_active` — all inactive → None
- [ ] `test_get_visible_devices_weight_zero_excluded` — GPU active but weight 0 → None

- [ ] `test_validate_gpu_weights_valid` — active GPU with weight > 0 → (True, "")
- [ ] `test_validate_gpu_weights_all_inactive` — no active GPUs → (False, error message)
- [ ] `test_validate_gpu_weights_zero_weight` — active but weight 0 → (False, error message)

### Integration Tests
- [ ] `test_start_model_sets_cuda_visible_devices` — mock subprocess.Popen and verify env["CUDA_VISIBLE_DEVICES"] is set correctly

## Success Criteria

- All new tests passing (12+)
- All existing tests still passing
- `CUDA_VISIBLE_DEVICES` set correctly when model starts
- Inactive GPUs excluded from tensor-split array
- Start rejected with clear error when no GPUs active
