---
status: pending
title: Refactor llama_manager.py entry point
type: refactor
complexity: critical
dependencies:
  - task_01
  - task_02
  - task_03
  - task_04
  - task_05
  - task_06
---

# Refactor llama_manager.py entry point

## Overview

Slim `llama_manager.py` from 2408 lines to ~400 lines by updating all imports, service initialization, and route functions to use the newly extracted modules. This is the most critical task — all previous extractions depend on it for the final assembly.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.7.
- Focus on WHAT: wire up the extracted modules into a working application.
- Do NOT introduce new features. This is purely wiring and cleanup.
- After this task, the app MUST start and serve all existing endpoints identically.
- Tests must pass before considering this task complete.

## Requirements

1. MUST update all imports in `llama_manager.py` to import from extracted modules:
   - `from config_manager import ConfigManager, TokenManager, AuthManager`
   - `from gpu_manager import GPUDetector, GPUInfo`
   - `from log_manager import LogManager`
   - `from process_manager import ProcessManager, OOMWatchdog`
   - `from model_manager import ModelScanner, DownloadManager`
   - `from ui_renderer import UIRenderer`
2. MUST update service singleton initialization (lines 999–1009) to use the new module-level constants for log paths.
3. MUST update all 19 API route functions to reference the injected service instances.
4. MUST update `startup_event()` (lines 2321–2370) to use injected services.
5. MUST keep the FastAPI app creation (`app = FastAPI(...)`) and entry point (`uvicorn.run(...)`) in `llama_manager.py`.
6. MUST remove ALL dead code: unused imports (`glob`, `pathlib`), unused constants (`MODEL_SETTINGS_PATH`).
7. MUST NOT change any API endpoint behavior, request/response schemas, or HTML output.

## Subtasks

- [ ] Update all imports in `llama_manager.py` to import from extracted modules
- [ ] Update service singleton initialization block
- [ ] Update all 19 API route functions to use injected service instances
- [ ] Update `startup_event()` to use injected services
- [ ] Remove dead imports (glob, pathlib) and unused constants (MODEL_SETTINGS_PATH)
- [ ] Remove all class/function definitions already extracted to modules
- [ ] Verify `llama_manager.py` is ~400 lines
- [ ] Run `python3 -m py_compile llama_manager.py` to verify syntax
- [ ] Verify app starts: `python3 llama_manager.py &` and check `curl http://localhost:8000/status`
- [ ] Run full test suite: `pytest`

## Implementation Details

### File Paths to Modify
- `llama_manager.py` — massive refactor: ~2000 lines removed, ~400 lines remain

### Integration Points
- All 19 API routes (lines 1059–1235)
- `startup_event()` (lines 2321–2370)
- Entry point (lines 2406–2407)

### Relevant Files
- `llama_manager.py` — the file being refactored
- All extracted modules (task_01–06) — consumed by this task

### Dependent Files
- `tests/unit/test_config_token.py` — imports may need updates
- `tests/unit/test_gpu_scanner.py` — imports may need updates
- `tests/unit/test_oom_watchdog.py` — imports may need updates
- `tests/integration/test_api_endpoints.py` — imports may need updates

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `llama_manager.py` reduced to ~400 lines
- All API routes functional with injected services
- App starts and serves all endpoints identically
- Zero dead code (no unused imports or constants)

## Tests

- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify app starts without errors
- Verify `curl http://localhost:8000/status` returns valid JSON
- Verify `curl http://localhost:8000/` returns HTML
- Run full test suite: `pytest` (all existing tests must pass)
- Manual: test login, model listing, metrics endpoints

## Success Criteria

- `llama_manager.py` is ~400 lines (2408 → ~400)
- All existing tests passing
- App starts and all endpoints functional
- No behavioral changes to any endpoint
- No missing symbols, no import errors
