---
status: pending
title: Extract process_manager.py
type: refactor
complexity: medium
dependencies:
  - task_02
  - task_03
---

# Extract process_manager.py

## Overview

Extract the `ProcessManager` class (lines 374–577) and `OOMWatchdog` class (lines 585–740) from `llama_manager.py` into a new `process_manager.py` module. These classes have internal dependencies on each other and external dependencies on `gpu_manager` and `log_manager`. No behavioral changes.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.4.
- Focus on WHAT: extract exactly as-is. Do NOT modify behavior.
- Pay attention to cross-references between ProcessManager and OOMWatchdog.
- Tests must pass before considering this task complete.

## Requirements

1. MUST create `process_manager.py` in the project root.
2. MUST extract `ProcessManager` class (lines 374–577) exactly as-is.
3. MUST extract `OOMWatchdog(threading.Thread)` class (lines 585–740) exactly as-is.
4. MUST keep required constants: `MODELS_DIR`, `SERVER_LOG_PATH`, `MANAGER_LOG_PATH`, `LLAMA_SERVER_BIN`, `SERVER_PORT`, `DEFAULT_CONTEXT_SIZE`.
5. MUST keep required imports: `subprocess`, `os`, `signal`, `time`, `threading`, `re`, `logging`.
6. MUST NOT change any method signatures, class attributes, or logic.
7. MUST update `ProcessManager.start()` to use `GPUDetector` from gpu_manager (already imported via task_02).
8. MUST update `OOMWatchdog._check_log()` to use `SERVER_LOG_PATH` from log_manager.

## Subtasks

- [ ] Create `process_manager.py` with ProcessManager and OOMWatchdog classes
- [ ] Copy ProcessManager class (lines 374–577) to `process_manager.py`
- [ ] Copy OOMWatchdog class (lines 585–740) to `process_manager.py`
- [ ] Copy required constants to `process_manager.py`
- [ ] Remove class definitions from `llama_manager.py`
- [ ] Add import statements to `llama_manager.py`: `from process_manager import ProcessManager, OOMWatchdog`
- [ ] Update service singleton initialization in `llama_manager.py`
- [ ] Run `python3 -m py_compile process_manager.py` to verify syntax
- [ ] Run existing tests: `pytest tests/unit/test_oom_watchdog.py`

## Implementation Details

### File Paths to Create
- `process_manager.py` — new file

### File Paths to Modify
- `llama_manager.py` — remove lines 374–740, add imports

### Integration Points
- `process_manager = ProcessManager(config_manager, token_manager)` at line 1004 → will need updated constructor signature
- `oom_watchdog = OOMWatchdog(process_manager, config_manager)` at lines 1007–1009 → will need updated constructor
- `ProcessManager.start()` at line 1104 route → unchanged
- `OOMWatchdog.run()` called in startup_event at line 2323 → unchanged

### Relevant Files
- `llama_manager.py` lines 374–740 — source classes to extract
- `llama_manager.py` lines 1000–1009 — service singleton initialization
- `llama_manager.py` lines 2320–2370 — startup_event
- `tests/unit/test_oom_watchdog.py` — existing tests (9 tests)

### Dependent Files
- `llama_manager.py` — will import from process_manager
- `gpu_manager.py` (task_02) — ProcessManager depends on GPUDetector
- `log_manager.py` (task_03) — OOMWatchdog reads log files
- `tests/unit/test_oom_watchdog.py` — imports must update

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `process_manager.py` with ProcessManager and OOMWatchdog classes
- `llama_manager.py` with removed class definitions and new imports
- All 9 existing tests passing (`test_oom_watchdog.py`)

## Tests

- Verify all 9 existing tests in `test_oom_watchdog.py` pass
- Verify `python3 -m py_compile process_manager.py` succeeds
- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify app starts without errors

## Success Criteria

- All 9 existing tests in `test_oom_watchdog.py` passing
- `process_manager.py` compiles cleanly
- `llama_manager.py` compiles cleanly
- App starts without errors
- Test coverage maintained (no regression from 9 tests)
