---
status: completed
title: "CLIProxyAPI sidecar lifecycle and platform start/stop API"
type: backend
complexity: high
dependencies:
  - task_01
  - task_02
---

# Task 3: CLIProxyAPI sidecar lifecycle and platform start/stop API

## Overview
This task implements the shared CLIProxyAPI sidecar manager and platform activation API. It lets the operator start and stop a detected platform integration while AutoManager supervises one localhost sidecar process.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: AutoManager MUST manage a single shared CLIProxyAPI sidecar for MVP platform integrations.
- Requirement 2: The sidecar MUST bind to `127.0.0.1` on an AutoManager-managed port.
- Requirement 3: Starting any detected platform MUST ensure the shared sidecar is running.
- Requirement 4: Stopping a platform MUST deactivate that platform and stop the sidecar when no platforms remain active.
- Requirement 5: Failures MUST return concise reasons suitable for UI display.
</requirements>

## Subtasks
- [x] 3.1 Add a sidecar manager with port allocation, process start, health, and stop behavior.
- [x] 3.2 Add generated CLIProxyAPI runtime config handling under AutoManager data paths.
- [x] 3.3 Add platform runtime state for active, running, not-ready, and error states.
- [x] 3.4 Add authenticated start and stop routes for platform `backend_id`s.
- [x] 3.5 Add unit tests with subprocess and health-check boundaries mocked.

## Implementation Details
Keep process supervision isolated from `ProcessManager`, which remains responsible for `llama-server`. Reference the TechSpec "Integration Points" section and ADR-002 for sidecar ownership.

### Relevant Files
- `process_manager.py` - existing process lifecycle patterns and port checks for local models.
- `llama_manager.py` - route registration and authenticated endpoint patterns.
- `paths.py` - config/log/data path conventions.
- `tests/unit/test_process_manager_extended.py` - process lifecycle test style.
- `tests/unit/test_llama_manager_routes.py` - FastAPI route test style.

### Dependent Files
- Platform catalog module from task 02 - provides platform definitions and detection state.
- `config_manager.py` - supplies persisted platform preferences from task 01.
- `log_manager.py` - may be used if sidecar output is streamed or recorded.

### Related ADRs
- [ADR-002: CLIProxyAPI HTTP Sidecar for Platform Backends](adrs/adr-002.md) - Chooses the HTTP sidecar approach.
- [ADR-004: Shared Sidecar Lifecycle and Real Model Discovery](adrs/adr-004.md) - Requires one shared sidecar process.

## Deliverables
- Shared sidecar lifecycle manager.
- `POST /platforms/{backend_id}/start` and `POST /platforms/{backend_id}/stop`.
- Platform runtime state exposed from backend services.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for platform start/stop route contracts **(REQUIRED)**

## Tests
- Unit tests:
  - [x] Starting `platform:codex` starts the sidecar when detected and inactive.
  - [x] Starting a missing platform returns a 400 or 409 with a concise reason.
  - [x] Starting a second platform reuses the existing sidecar port.
  - [x] Stopping the last active platform stops the sidecar process.
  - [x] Sidecar health failure marks runtime state as not-ready with `last_error`.
- Integration tests:
  - [x] POST `/platforms/platform:codex/start` requires auth.
  - [x] POST `/platforms/platform:codex/stop` returns stable JSON when the platform is already inactive.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- Platform start creates an operational backend state.
- No provider credentials are requested or stored by AutoManager.
