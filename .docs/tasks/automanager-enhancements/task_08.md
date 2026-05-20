---
status: pending
title: Fix tests after refactoring
type: test
complexity: medium
dependencies:
  - task_07
---

# Fix tests after refactoring

## Overview

Update all test files to import from the new extracted modules instead of the monolithic `llama_manager`. Ensure all existing tests pass after the modular refactoring. This is a cleanup task — no new functionality, just import fixes and fixture updates.

## <critical>

- Read the PRD and TechSpec before starting.
- Focus on WHAT: update imports to match new module structure.
- Do NOT change test logic or assertions.
- All existing tests must pass at the end of this task.

## Requirements

1. MUST update `tests/conftest.py` imports to reference extracted modules.
2. MUST update `tests/unit/test_config_token.py` imports to reference `config_manager`.
3. MUST update `tests/unit/test_gpu_scanner.py` imports to reference `gpu_manager` and `model_manager`.
4. MUST update `tests/unit/test_oom_watchdog.py` imports to reference `process_manager` and `config_manager`.
5. MUST update `tests/integration/test_api_endpoints.py` imports to reference extracted modules.
6. MUST NOT change any test logic, assertions, or fixture behavior.
7. MUST ensure all existing tests pass: `pytest` exits with 0.

## Subtasks

- [ ] Update `tests/conftest.py` — fix all imports to reference extracted modules
- [ ] Update `tests/unit/test_config_token.py` — import from config_manager
- [ ] Update `tests/unit/test_gpu_scanner.py` — import from gpu_manager, model_manager
- [ ] Update `tests/unit/test_oom_watchdog.py` — import from process_manager, config_manager
- [ ] Update `tests/integration/test_api_endpoints.py` — import from all extracted modules
- [ ] Run full test suite: `pytest -v`
- [ ] Fix any import errors or failing tests
- [ ] Verify all 50+ existing tests pass

## Implementation Details

### File Paths to Modify
- `tests/conftest.py` — update imports
- `tests/unit/test_config_token.py` — update imports
- `tests/unit/test_gpu_scanner.py` — update imports
- `tests/unit/test_oom_watchdog.py` — update imports
- `tests/integration/test_api_endpoints.py` — update imports

### Integration Points
- All test files mock `llama_manager.GPUDetector`, `llama_manager.psutil`, etc. → update to mock extracted module paths.

### Relevant Files
- `tests/conftest.py` — shared fixtures
- `tests/unit/test_config_token.py` — 16 tests
- `tests/unit/test_gpu_scanner.py` — 8 tests
- `tests/unit/test_oom_watchdog.py` — 9 tests
- `tests/integration/test_api_endpoints.py` — 17 tests

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy

## Deliverables

- All test files with correct imports
- All existing tests passing
- Test command: `pytest -v` exits with 0

## Tests

- Run `pytest -v` — all 50+ existing tests must pass
- Verify test output shows no import errors
- Verify test output shows no assertion failures

## Success Criteria

- All 50 existing tests passing
- No import errors in any test file
- No assertion failures
- `pytest -v` exits with code 0
