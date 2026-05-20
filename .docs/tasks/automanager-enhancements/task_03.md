---
status: pending
title: Extract log_manager.py
type: refactor
complexity: low
dependencies: []
---

# Extract log_manager.py

## Overview

Extract the logging configuration (lines 52–63) and `SSEStreamer` class (lines 968–994) from `llama_manager.py` into a new `log_manager.py` module. Create a `LogManager` class that encapsulates both the logging setup and SSE streaming. No behavioral changes.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.3.
- Focus on WHAT: extract exactly as-is. Do NOT modify behavior.
- The `SSEStreamer.stream()` static method must remain callable the same way.
- Tests must pass before considering this task complete.

## Requirements

1. MUST create `log_manager.py` in the project root.
2. MUST create `LogManager` class wrapping the logging setup and SSE streaming.
3. MUST extract logging.basicConfig configuration (lines 52–63) into `LogManager.__init__`.
4. MUST extract `SSEStreamer.stream()` static method (lines 968–994) into `LogManager.stream_logs()`.
5. MUST keep required constants: `SERVER_LOG_PATH`, `MANAGER_LOG_PATH`.
6. MUST NOT change any method signatures or logic.
7. MUST provide `LogManager.setup_logging()` method that returns the configured logger.
8. MUST provide `LogManager.stream_logs()` method that returns `StreamingResponse`.

## Subtasks

- [ ] Create `log_manager.py` with LogManager class
- [ ] Extract logging.basicConfig setup into LogManager.__init__
- [ ] Extract SSEStreamer.stream() into LogManager.stream_logs()
- [ ] Copy required constants (SERVER_LOG_PATH, MANAGER_LOG_PATH) to `log_manager.py`
- [ ] Remove logging setup and SSEStreamer from `llama_manager.py`
- [ ] Add import statements to `llama_manager.py`
- [ ] Update service singleton for log manager in `llama_manager.py`
- [ ] Run `python3 -m py_compile log_manager.py` to verify syntax
- [ ] Run existing tests: `pytest tests/unit/test_oom_watchdog.py`

## Implementation Details

### File Paths to Create
- `log_manager.py` — new file

### File Paths to Modify
- `llama_manager.py` — remove logging setup (lines 52–63) and SSEStreamer (lines 968–994), add imports

### Integration Points
- `logging.getLogger("automanager")` in other modules → unchanged (uses same logger name)
- `@app.get("/logs")` route at line 1205 → will call `log_mgr.stream_logs()` instead of `SSEStreamer.stream()`
- `OOMWatchdog._check_log()` → reads `SERVER_LOG_PATH` constant, may need update to use log_manager path

### Relevant Files
- `llama_manager.py` lines 52–63 — logging setup
- `llama_manager.py` lines 968–994 — SSEStreamer class
- `tests/unit/test_oom_watchdog.py` — existing tests (9 tests) for OOMWatchdog

### Dependent Files
- `llama_manager.py` — will import from log_manager
- `process_manager.py` (task_04) — may reference log paths
- `tests/unit/test_oom_watchdog.py` — imports may need updates

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `log_manager.py` with LogManager class
- `llama_manager.py` with removed logging setup and SSEStreamer, new imports
- All 9 existing tests passing (`test_oom_watchdog.py`)

## Tests

- Verify all 9 existing tests in `test_oom_watchdog.py` pass
- Verify `python3 -m py_compile log_manager.py` succeeds
- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify SSE stream still works (manual test: `curl http://localhost:8000/logs`)

## Success Criteria

- All 9 existing tests in `test_oom_watchdog.py` passing
- `log_manager.py` compiles cleanly
- `llama_manager.py` compiles cleanly
- App starts without errors
- SSE stream endpoint still functional
- Test coverage maintained (no regression from 9 tests)
