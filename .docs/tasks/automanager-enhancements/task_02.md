---
status: pending
title: Extract gpu_manager.py
type: refactor
complexity: low
dependencies:
  - task_01
---

# Extract gpu_manager.py

## Overview

Extract the `GPUDetector` class from `llama_manager.py` (lines 279–366) into a new `gpu_manager.py` module. Add the `GPUInfo` dataclass as defined in the TechSpec. No behavioral changes; pure code movement with updated import paths.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.2.
- Focus on WHAT: extract the class exactly as it is. Do NOT modify behavior.
- Minimize code changes: only move lines and fix imports.
- Tests must pass before considering this task complete.

## Requirements

1. MUST create `gpu_manager.py` in the project root.
2. MUST extract `GPUDetector` class (lines 279–366) exactly as-is.
3. MUST add `GPUInfo` dataclass with fields: `index: int`, `name: str`, `vram: int`.
4. MUST keep required imports: `subprocess`, `re`, `logging`, `typing.List`, `typing.Dict`, `typing.Any`.
5. MUST NOT change any method signatures, class attributes, or logic.
6. MUST extract `LLAMA_SERVER_BIN` constant (line 43) as it's used by `GPUDetector`.
7. MUST update all references in `llama_manager.py` to import from `gpu_manager` instead of using local class definition.

## Subtasks

- [ ] Create `gpu_manager.py` with GPUInfo dataclass
- [ ] Copy GPUDetector class (lines 279–366) to `gpu_manager.py`
- [ ] Copy required constants (LLAMA_SERVER_BIN) to `gpu_manager.py`
- [ ] Remove class definition from `llama_manager.py`
- [ ] Add import statements to `llama_manager.py`: `from gpu_manager import GPUDetector, GPUInfo`
- [ ] Update service singleton initialization in `llama_manager.py`
- [ ] Run `python3 -m py_compile gpu_manager.py` to verify syntax
- [ ] Run existing tests: `pytest tests/unit/test_gpu_scanner.py`

## Implementation Details

### File Paths to Create
- `gpu_manager.py` — new file

### File Paths to Modify
- `llama_manager.py` — remove lines 279–366, add imports

### Integration Points
- `gpu_detector = GPUDetector()` at line 1003 → unchanged
- All API routes referencing `gpu_detector` → unchanged

### Relevant Files
- `llama_manager.py` lines 279–366 — source class to extract
- `llama_manager.py` line 43 — LLAMA_SERVER_BIN constant
- `tests/unit/test_gpu_scanner.py` — existing tests (8 tests) for GPUDetector

### Dependent Files
- `llama_manager.py` — will import from gpu_manager
- `process_manager.py` (task_04) — depends on GPUInfo
- `tests/unit/test_gpu_scanner.py` — imports must update

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `gpu_manager.py` with GPUDetector class and GPUInfo dataclass
- `llama_manager.py` with removed class definition and new imports
- All 8 existing tests passing (`test_gpu_scanner.py`)

## Tests

- Verify all 8 existing tests in `test_gpu_scanner.py` pass
- Verify `python3 -m py_compile gpu_manager.py` succeeds
- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify GPUDetector.detect_gpus() and get_metrics() produce same results as before

## Success Criteria

- All 8 existing tests in `test_gpu_scanner.py` passing
- `gpu_manager.py` compiles cleanly
- `llama_manager.py` compiles cleanly
- App starts without errors
- Test coverage maintained (no regression from 8 tests)
