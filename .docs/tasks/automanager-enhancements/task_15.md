---
status: pending
title: Full test suite + E2E validation
type: test
complexity: medium
dependencies:
  - task_09
  - task_10
  - task_12
  - task_13
  - task_14
---

# Full test suite + E2E validation

## Overview

Run the complete test suite and perform end-to-end validation across all implemented features. This is the final quality gate before the enhancements are considered complete.

## <critical>

- Read the PRD and TechSpec before starting.
- Focus on WHAT: verify ALL features work correctly together.
- Do NOT introduce new features. This is validation only.
- Fix any failing tests before considering this task complete.

## Requirements

1. MUST run full test suite: `pytest -v` — all tests must pass.
2. MUST verify GPU strict enforcement: `CUDA_VISIBLE_DEVICES` set correctly, inactive GPUs excluded.
3. MUST verify log rotation: `logs/` directory exists, files rotate at 10MB.
4. MUST verify OFFLINE initial state: dashboard loads with OFFLINE, transitions correctly.
5. MUST verify Pac-Man background: canvas present in HTML, gradients applied.
6. MUST verify Quick-Install: `installer/setup.sh` syntax valid, README sections complete.
7. MUST verify all existing API endpoints functional: `curl` all endpoints.
8. MUST verify test coverage >= 80%: `pytest --cov=.` reports coverage.

## Subtasks

- [ ] Run full test suite: `pytest -v`
- [ ] Fix any failing tests
- [ ] Verify GPU enforcement: mock test for CUDA_VISIBLE_DEVICES
- [ ] Verify log rotation: create test log > 10MB, verify rotation
- [ ] Verify OFFLINE state: render dashboard HTML, check initial class
- [ ] Verify Pac-Man canvas: check HTML contains canvas element
- [ ] Verify gradients: check HTML contains gradient CSS classes
- [ ] Verify setup.sh: `bash -n installer/setup.sh`
- [ ] Verify README: all 12 sections present
- [ ] Verify LICENSE: Apache 2.0 text present
- [ ] Run E2E: start app, test login, list models, start model, check logs, stop model
- [ ] Generate coverage report: `pytest --cov=.`

## Implementation Details

### File Paths to Verify
- All modules: `config_manager.py`, `gpu_manager.py`, `log_manager.py`, `process_manager.py`, `model_manager.py`, `ui_renderer.py`, `llama_manager.py`
- All tests: `tests/unit/test_config_token.py`, `tests/unit/test_gpu_scanner.py`, `tests/unit/test_oom_watchdog.py`, `tests/integration/test_api_endpoints.py`, `tests/unit/test_gpu_manager_new.py`, `tests/unit/test_log_manager.py`, `tests/unit/test_ui_renderer_new.py`
- Installer: `installer/setup.sh`
- Docs: `README.md`, `LICENSE`

### Integration Points
- Full app start → all modules loaded → all routes functional
- GPU detection → tensor split → CUDA_VISIBLE_DEVICES → llama-server start
- Log writing → rotation → SSE streaming
- Dashboard rendering → OFFLINE state → Pac-Man canvas → gradients

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy

## Deliverables

- All tests passing (existing + new)
- Test coverage >= 80%
- E2E validation complete
- No regressions

## Tests

- [ ] `pytest -v` — all tests pass
- [ ] `pytest --cov=.` — coverage >= 80%
- [ ] Manual E2E: full workflow (start → run → stop)
- [ ] Manual: verify dashboard UI in browser
- [ ] Manual: verify logs in `logs/` directory
- [ ] Manual: verify setup.sh syntax

## Success Criteria

- All tests passing (existing + new, 60+ total)
- Test coverage >= 80%
- No regressions from existing functionality
- Full E2E workflow verified
- All features working together correctly
- App starts, serves all endpoints, GPU enforcement works, logs rotate, UI shows correct state
