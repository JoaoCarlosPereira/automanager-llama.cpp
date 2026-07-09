---
status: completed
title: "Hybrid backend identity and config foundations"
type: backend
complexity: medium
dependencies: []
---

# Task 1: Hybrid backend identity and config foundations

## Overview
This task establishes the backend identity and persistence foundation for hybrid local and platform backends. It keeps existing local model settings keyed by normalized `model_path` while adding platform-safe config and request schemas keyed by `backend_id`.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: AutoManager MUST introduce stable platform backend identifiers without passing platform IDs through `normalize_model_path`.
- Requirement 2: AutoManager MUST persist platform preferences separately from local `model_configs`.
- Requirement 3: Smart proxy settings MUST support `primary_backend_id` while preserving legacy `primary_model_path` behavior.
- Requirement 4: Existing local model config migration and defaults MUST remain backward compatible.
- Requirement 5: API schemas MUST accept platform `backend_id` for proxy settings without breaking existing `model_path` callers.
</requirements>

## Subtasks
- [x] 1.1 Add platform config defaults and migration-safe load behavior.
- [x] 1.2 Add helper methods for reading and updating platform settings by `backend_id`.
- [x] 1.3 Extend smart proxy settings with backend-aware primary selection.
- [x] 1.4 Extend request schemas for backend-aware proxy configuration.
- [x] 1.5 Add unit tests for config migration and backend-aware settings.

## Implementation Details
Create the durable contract described in the TechSpec "Core Interfaces" and "Data Models" sections. Keep local model code paths operational by treating `model_path` as the legacy local key and `backend_id` as the platform key.

### Relevant Files
- `config_manager.py` - owns config defaults, migration, local model settings, and smart proxy settings.
- `schemas.py` - defines FastAPI request models used by proxy config routes.
- `tests/unit/test_config_token.py` - contains existing config migration and default behavior tests.
- `tests/unit/test_proxy_config.py` - covers smart proxy config validation and persistence.

### Dependent Files
- `llama_manager.py` - will consume backend-aware request schemas in later tasks.
- `proxy_router.py` - will use `primary_backend_id` and platform config in later tasks.
- `static/js/proxy.js` - will send `backend_id` for platform proxy controls in later tasks.

### Related ADRs
- [ADR-003: Stable Backend Identity for Platform Integrations](adrs/adr-003.md) - Defines `backend_id`, `backend_type`, and platform config separation.
- [ADR-005: Startup-Only Platform Detection with Persisted Preferences](adrs/adr-005.md) - Requires persisting preferences without persisting detection state as truth.

## Deliverables
- Backend-aware config defaults and migrations.
- Platform settings helpers keyed by `backend_id`.
- Extended proxy request schemas for local and platform settings.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for config compatibility where route-level behavior is affected **(REQUIRED)**

## Tests
- Unit tests:
  - [x] Loading an empty config returns `platform_configs` defaults without removing `model_configs`.
  - [x] Updating `platform:codex` stores proxy preferences under `platform_configs` and never under a normalized path key.
  - [x] Updating `smart_proxy.primary_backend_id` preserves existing `primary_model_path` when no local primary is changed.
  - [x] Existing Windows and POSIX local path migrations continue to pass unchanged.
- Integration tests:
  - [x] POST `/proxy/config` continues to accept `primary_model_path` for local models after schema changes.
  - [x] POST `/models/proxy` continues to accept `model_path` for local model settings after schema changes.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- Platform IDs are not normalized as filesystem paths.
- Local model config behavior is unchanged for existing users.
