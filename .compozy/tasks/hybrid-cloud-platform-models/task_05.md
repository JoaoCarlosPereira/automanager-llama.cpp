---
status: completed
title: "Smart proxy support for platform backends"
type: backend
complexity: high
dependencies:
  - task_01
  - task_03
  - task_04
---

# Task 5: Smart proxy support for platform backends

## Overview
This task makes the smart proxy backend-aware so platform integrations can participate only when the operator explicitly enables them. It keeps existing local routing behavior while allowing `backend_id`-based primary and eligibility controls.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: Platform backends MUST be excluded from smart proxy routing unless explicitly proxy-eligible.
- Requirement 2: Smart proxy primary selection MUST support platform `backend_id` and legacy local `model_path`.
- Requirement 3: Proxy backend snapshots MUST distinguish local and platform backends.
- Requirement 4: Routing decisions MUST forward platform requests to the shared sidecar port.
- Requirement 5: Existing sticky session and local least-busy behavior MUST remain stable.
</requirements>

## Subtasks
- [x] 5.1 Add backend-aware primary lookup for local paths and platform IDs.
- [x] 5.2 Add platform config flags to proxy eligibility and parallelism lookup.
- [x] 5.3 Extend backend snapshots with backend identity and provider metadata.
- [x] 5.4 Extend routing and session state to tolerate platform backends.
- [x] 5.5 Add smart proxy route tests for platform eligibility and routing.

## Implementation Details
Reference the TechSpec "API Endpoints" and "Technical Considerations" sections. Keep the router's port-based forwarding intact, but resolve eligibility and primary status through a backend key abstraction.

### Relevant Files
- `proxy_router.py` - smart proxy selection, sessions, snapshots, in-flight counts.
- `llama_manager.py` - admin proxy routes and smart proxy forwarding.
- `config_manager.py` - platform and local proxy settings.
- `tests/unit/test_proxy_router.py` - router-level behavior tests.
- `tests/unit/test_smart_proxy_routes.py` - route-level smart proxy behavior tests.

### Dependent Files
- `schemas.py` - backend-aware request schema from task 01.
- Sidecar runtime state from task 03 - supplies platform sidecar port.
- Hybrid status from task 04 - supplies platform instances or backend descriptors.

### Related ADRs
- [ADR-003: Stable Backend Identity for Platform Integrations](adrs/adr-003.md) - Requires backend-aware identity.
- [ADR-004: Shared Sidecar Lifecycle and Real Model Discovery](adrs/adr-004.md) - Routes platform backends to the shared sidecar port.

## Deliverables
- Backend-aware smart proxy selection and snapshots.
- Platform `backend_id` support in `/proxy/config` and `/models/proxy`.
- Tests proving cloud backends are opt-in for proxy routing.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for platform-backed smart proxy routes **(REQUIRED)**

## Tests
- Unit tests:
  - [x] Platform backend with `proxy_eligible: false` is not selected as a secondary backend.
  - [x] Platform backend selected as primary can route through the sidecar port.
  - [x] Local `primary_model_path` behavior remains unchanged.
  - [x] Backend snapshots include `backend_type: platform` and provider metadata.
- Integration tests:
  - [x] POST `/models/proxy` with `backend_id: platform:codex` updates platform proxy eligibility.
  - [x] POST `/proxy/resolve` returns a sidecar backend when platform primary is active and eligible.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- Cloud integrations never become proxy fallback capacity silently.
- Proxy UI/API can explain local versus platform backend participation.
