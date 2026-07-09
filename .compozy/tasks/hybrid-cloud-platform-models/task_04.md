---
status: completed
title: "Hybrid `/models`, `/status`, and `/v1` model availability"
type: backend
complexity: high
dependencies:
  - task_01
  - task_02
  - task_03
---

# Task 4: Hybrid `/models`, `/status`, and `/v1` model availability

## Overview
This task wires active platform integrations into AutoManager's existing catalog, status, and OpenAI-compatible model discovery surface. It preserves local model behavior while adding platform entries and sidecar-backed `/v1/models` aggregation.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: `GET /models` MUST return local models plus a platform catalog array.
- Requirement 2: `GET /status` MUST include platform runtime state without removing existing `instances`.
- Requirement 3: `GET /v1/models` MUST include models returned by the sidecar for active platform integrations.
- Requirement 4: Sidecar model IDs MUST be preserved in API responses.
- Requirement 5: Local-only behavior MUST remain stable when no platform integration is active.
</requirements>

## Subtasks
- [x] 4.1 Extend model catalog responses with platform entries.
- [x] 4.2 Extend status responses with platform runtime and hybrid backend information.
- [x] 4.3 Extend `/v1/models` aggregation to query the sidecar when active.
- [x] 4.4 Extend non-proxy `/v1` request forwarding for sidecar model IDs.
- [x] 4.5 Add route tests with a fake sidecar model endpoint.

## Implementation Details
Use the existing `_aggregate_models_response` and `openai_proxy` patterns as the integration point. Reference TechSpec "API Endpoints" for the intended contract.

### Relevant Files
- `llama_manager.py` - owns `/models`, `/status`, and `/v1/{path}`.
- `process_manager.py` - provides existing local instance status shape.
- `tests/unit/test_smart_proxy_routes.py` - contains route-level `/v1/models` tests.
- `tests/integration/test_api_endpoints.py` - covers API contracts.

### Dependent Files
- Sidecar manager from task 03 - supplies active sidecar port and runtime state.
- Platform catalog from task 02 - supplies platform list for `/models`.
- `static/js/models.js` - will consume `platforms` in task 06.

### Related ADRs
- [ADR-001: Unified Catalog MVP for Hybrid Platform Models](adrs/adr-001.md) - Requires cloud integrations in the existing model/API flow.
- [ADR-004: Shared Sidecar Lifecycle and Real Model Discovery](adrs/adr-004.md) - Requires preserving sidecar model IDs from `/v1/models`.

## Deliverables
- Hybrid `/models` and `/status` API responses.
- Sidecar-aware `/v1/models` aggregation.
- Sidecar-aware direct `/v1` forwarding for selected platform model IDs.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for mixed local and platform model discovery **(REQUIRED)**

## Tests
- Unit tests:
  - [x] `GET /models` returns `platforms` with all three MVP integrations.
  - [x] `GET /status` includes platform runtime state and keeps existing local instances.
  - [x] `GET /v1/models` returns local models when no platform is active.
  - [x] `GET /v1/models` merges fake sidecar models when a platform is active.
  - [x] Duplicate sidecar and local model IDs are de-duplicated deterministically.
- Integration tests:
  - [x] POST `/v1/chat/completions` with a sidecar model ID forwards to the sidecar port.
  - [x] Sidecar `/v1/models` failure logs a warning and does not break local model listing.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- Started platform integrations appear in `/v1/models`.
- Existing local clients can continue using AutoManager unchanged.
