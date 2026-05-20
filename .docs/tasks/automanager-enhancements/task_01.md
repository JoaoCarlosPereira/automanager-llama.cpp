---
status: pending
title: Extract config_manager.py
type: refactor
complexity: low
dependencies: []
---

# Extract config_manager.py

## Overview

Extract the `ConfigManager`, `TokenManager`, and `AuthManager` classes from `llama_manager.py` (lines 115–271) into a new `config_manager.py` module. This is the foundational extraction — other modules depend on these services. No behavioral changes; pure code movement with updated import paths.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.1.
- Focus on WHAT: extract the classes exactly as they are. Do NOT modify behavior.
- Minimize code changes: only move lines and fix imports.
- Tests must pass before considering this task complete.

## Requirements

1. MUST create `config_manager.py` in the project root.
2. MUST extract ConfigManager class (lines 115–167) exactly as-is.
3. MUST extract TokenManager class (lines 175–204) exactly as-is.
4. MUST extract AuthManager class (lines 212–271) exactly as-is.
5. MUST extract related module-level constants used by these classes: `CONFIG_PATH`, `MANAGER_LOG_PATH` (for AuthManager password init).
6. MUST keep `sys` and `pathlib` imports only if they are used by these classes (currently unused — remove them).
7. MUST NOT change any method signatures, class attributes, or logic.
8. MUST update all references in `llama_manager.py` to import from `config_manager` instead of using local class definitions.

## Subtasks

- [ ] Copy ConfigManager class (lines 115–167) to `config_manager.py`
- [ ] Copy TokenManager class (lines 175–204) to `config_manager.py`
- [ ] Copy AuthManager class (lines 212–271) to `config_manager.py`
- [ ] Copy required constants (CONFIG_PATH, MANAGER_LOG_PATH) to `config_manager.py`
- [ ] Remove class definitions from `llama_manager.py`
- [ ] Add import statements to `llama_manager.py`: `from config_manager import ConfigManager, TokenManager, AuthManager`
- [ ] Update service singleton initialization in `llama_manager.py` to use imported classes
- [ ] Run `python3 -m py_compile config_manager.py` to verify syntax
- [ ] Run existing tests: `pytest tests/unit/test_config_token.py`

## Implementation Details

### File Paths to Create
- `config_manager.py` — new file

### File Paths to Modify
- `llama_manager.py` — remove lines 115–271, add imports

### Integration Points
- `token_manager = TokenManager(config_manager)` at line 1001 → unchanged
- `auth_manager = AuthManager(config_manager, token_manager)` at line 1002 → unchanged
- All API routes referencing `config_manager`, `token_manager`, `auth_manager` → unchanged (they reference service instances, not classes)

### Relevant Files
- `llama_manager.py` lines 115–271 — source classes to extract
- `llama_manager.py` lines 39–42 — constants used by extracted classes
- `tests/unit/test_config_token.py` — existing tests (16 tests) for ConfigManager and TokenManager
- `tests/conftest.py` lines 13–16, 64–74 — fixtures creating ConfigManager, TokenManager, AuthManager

### Dependent Files
- `llama_manager.py` — will import from config_manager instead of defining locally
- `tests/unit/test_config_token.py` — imports must update
- `tests/conftest.py` — fixture imports must update

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `config_manager.py` with all three classes extracted
- `llama_manager.py` with removed class definitions and new imports
- All 16 existing tests passing (`test_config_token.py`)
- No behavioral changes to any API endpoint

## Tests

- Verify all 16 existing tests in `test_config_token.py` pass
- Verify `python3 -m py_compile config_manager.py` succeeds
- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify app starts: no import errors, no missing symbols

## Success Criteria

- All 16 existing tests in `test_config_token.py` passing
- `config_manager.py` compiles cleanly
- `llama_manager.py` compiles cleanly
- App starts without errors
- Test coverage maintained (no regression from 16 tests)
