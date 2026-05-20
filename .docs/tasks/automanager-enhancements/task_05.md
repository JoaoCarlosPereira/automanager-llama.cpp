---
status: pending
title: Extract model_manager.py
type: refactor
complexity: low
dependencies:
  - task_01
---

# Extract model_manager.py

## Overview

Extract the `ModelScanner` class (lines 747–873) and `DownloadManager` class (lines 880–961) from `llama_manager.py` into a new `model_manager.py` module. No behavioral changes; pure code movement.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.6.
- Focus on WHAT: extract exactly as-is. Do NOT modify behavior.
- Minimize code changes: only move lines and fix imports.
- Tests must pass before considering this task complete.

## Requirements

1. MUST create `model_manager.py` in the project root.
2. MUST extract `ModelScanner` class (lines 747–873) exactly as-is.
3. MUST extract `DownloadManager` class (lines 880–961) exactly as-is.
4. MUST keep required constants: `MODELS_DIR`.
5. MUST keep required imports: `os`, `glob`, `threading`, `time`, `subprocess`, `requests`.
6. MUST NOT change any method signatures, class attributes, or logic.
7. MUST update all references in `llama_manager.py` to import from `model_manager`.

## Subtasks

- [ ] Create `model_manager.py` with ModelScanner and DownloadManager classes
- [ ] Copy ModelScanner class (lines 747–873) to `model_manager.py`
- [ ] Copy DownloadManager class (lines 880–961) to `model_manager.py`
- [ ] Copy required constants (MODELS_DIR) to `model_manager.py`
- [ ] Remove class definitions from `llama_manager.py`
- [ ] Add import statements to `llama_manager.py`: `from model_manager import ModelScanner, DownloadManager`
- [ ] Update service singleton initialization in `llama_manager.py`
- [ ] Run `python3 -m py_compile model_manager.py` to verify syntax
- [ ] Run existing tests: `pytest tests/unit/test_gpu_scanner.py`

## Implementation Details

### File Paths to Create
- `model_manager.py` — new file

### File Paths to Modify
- `llama_manager.py` — remove lines 747–961, add imports

### Integration Points
- `model_scanner = ModelScanner()` at line 1005 → will need updated constructor (takes config_manager)
- `download_mgr = DownloadManager()` at line 1006 → unchanged
- All API routes referencing `model_scanner` and `download_mgr` → unchanged

### Relevant Files
- `llama_manager.py` lines 747–961 — source classes to extract
- `llama_manager.py` lines 1000–1009 — service singleton initialization
- `tests/unit/test_gpu_scanner.py` — existing tests (8 tests) including ModelScanner tests

### Dependent Files
- `llama_manager.py` — will import from model_manager
- `config_manager.py` (task_01) — ModelScanner depends on ConfigManager
- `tests/unit/test_gpu_scanner.py` — imports must update

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `model_manager.py` with ModelScanner and DownloadManager classes
- `llama_manager.py` with removed class definitions and new imports
- All 8 existing tests passing (`test_gpu_scanner.py`)

## Tests

- Verify all 8 existing tests in `test_gpu_scanner.py` pass
- Verify `python3 -m py_compile model_manager.py` succeeds
- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify app starts without errors

## Success Criteria

- All 8 existing tests in `test_gpu_scanner.py` passing
- `model_manager.py` compiles cleanly
- `llama_manager.py` compiles cleanly
- App starts without errors
- Test coverage maintained (no regression)
