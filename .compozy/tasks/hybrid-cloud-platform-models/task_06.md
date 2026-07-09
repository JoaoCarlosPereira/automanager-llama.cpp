---
status: completed
title: "Frontend platform cards, start flow, and proxy controls"
type: frontend
complexity: high
dependencies:
  - task_04
  - task_05
---

# Task 6: Frontend platform cards, start flow, and proxy controls

## Overview
This task updates the dashboard to render platform integrations beside local model cards. It adds platform-specific start/stop actions, not-ready reasons, and backend-aware smart proxy controls without changing the familiar local model card workflow.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: The model catalog UI MUST show platform integrations in the same catalog area as local models.
- Requirement 2: Platform cards MUST show provider identity, state, and concise reason text when unavailable.
- Requirement 3: Platform start and stop controls MUST call backend-aware platform endpoints.
- Requirement 4: Platform proxy controls MUST send `backend_id` instead of `model_path`.
- Requirement 5: Local model cards and tabs MUST remain visually and behaviorally unchanged.
</requirements>

## Subtasks
- [x] 6.1 Render platform cards from the `platforms` array returned by `/models`.
- [x] 6.2 Add platform start and stop UI actions with loading and error states.
- [x] 6.3 Add backend-aware proxy primary, eligibility, and parallelism controls for platform cards.
- [x] 6.4 Update proxy monitoring labels to distinguish local and platform backends.
- [x] 6.5 Add UI contract tests for platform card and proxy control markup.

## Implementation Details
Follow existing sidebar card density and control styling in `static/js/models.js` and `static/js/proxy.js`. Reference the TechSpec "System Architecture" and PRD "UX Requirements" sections for required states.

### Relevant Files
- `static/js/models.js` - renders model catalog cards and start/stop actions.
- `static/js/proxy.js` - handles proxy primary and eligibility controls.
- `static/js/state.js` - shared active instance/runtime state.
- `tests/unit/test_html_contract.py` - dashboard markup contract tests.
- `tests/unit/test_ui_proxy.py` - proxy UI behavior test patterns.

### Dependent Files
- `/models` response from task 04 - supplies platform catalog.
- `/status` response from task 04 - supplies platform runtime state.
- `/models/proxy` and `/proxy/config` from task 05 - accept platform `backend_id`.

### Related ADRs
- [ADR-001: Unified Catalog MVP for Hybrid Platform Models](adrs/adr-001.md) - Requires unified catalog cards.
- [ADR-003: Stable Backend Identity for Platform Integrations](adrs/adr-003.md) - Requires platform UI controls to use `backend_id`.

## Deliverables
- Platform cards in the existing model catalog UI.
- Platform start/stop UI actions.
- Backend-aware proxy controls for platform cards.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for UI contract and route payloads where practical **(REQUIRED)**

## Tests
- Unit tests:
  - [x] Platform card markup includes provider label, status label, and reason text.
  - [x] Disabled or missing platform cards do not expose an enabled start action.
  - [x] Platform proxy eligibility control sends `backend_id`.
  - [x] Local model card markup still includes existing rename, delete, auto-start, and proxy controls.
- Integration tests:
  - [x] Loading `/models` with a platform entry renders the platform in the model list container.
  - [x] Starting a platform card updates status and refreshes `/v1/models` visibility through mocked API responses.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- Operators can start detected cloud integrations from the UI.
- Unavailable platform cards show a reason instead of disappearing.
