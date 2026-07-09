---
status: completed
title: "Hybrid flow hardening, documentation, and regression coverage"
type: docs
complexity: medium
dependencies:
  - task_02
  - task_03
  - task_04
  - task_05
  - task_06
---

# Task 7: Hybrid flow hardening, documentation, and regression coverage

## Overview
This task closes the MVP by documenting the hybrid platform flow and adding final regression coverage around the end-to-end operator journey. It verifies that local model behavior, platform detection, platform activation, `/v1/models`, and smart proxy opt-in behavior work together.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: Documentation MUST explain startup-only detection and the restart requirement for newly installed tools.
- Requirement 2: Documentation MUST state that AutoManager does not collect provider credentials for the MVP.
- Requirement 3: Regression tests MUST cover the primary MVP success metric: activate a detected integration and see it in `/v1/models`.
- Requirement 4: Regression tests MUST cover local-model behavior when no platform is detected or active.
- Requirement 5: Final verification MUST run the relevant task and repository checks available in the project.
</requirements>

## Subtasks
- [x] 7.1 Add concise documentation for the hybrid platform MVP.
- [x] 7.2 Add regression tests for platform activation and `/v1/models` discovery.
- [x] 7.3 Add regression tests for local-only behavior.
- [x] 7.4 Add regression tests for explicit smart proxy eligibility.
- [x] 7.5 Run validation and update task tracking after verification.

## Implementation Details
Use the TechSpec "Monitoring and Observability" and "Known Risks" sections as documentation input. Keep documentation concise and aligned with existing README tone.

### Relevant Files
- `README.md` - general project documentation.
- `README.pt-BR.md` - Portuguese project documentation.
- `tests/unit/test_smart_proxy_routes.py` - route-level regression tests.
- `tests/integration/test_api_endpoints.py` - integration API behavior tests.
- `.compozy/tasks/hybrid-cloud-platform-models/_tasks.md` - master tracking file.

### Dependent Files
- Backend and frontend changes from tasks 02 through 06.
- Task files `task_02.md` through `task_06.md` for status and checklist updates.
- ADRs in `adrs/` for documented constraints.

### Related ADRs
- [ADR-001: Unified Catalog MVP for Hybrid Platform Models](adrs/adr-001.md) - Defines the MVP success metric.
- [ADR-002: CLIProxyAPI HTTP Sidecar for Platform Backends](adrs/adr-002.md) - Documents sidecar architecture.
- [ADR-005: Startup-Only Platform Detection with Persisted Preferences](adrs/adr-005.md) - Documents detection lifecycle and restart requirement.

## Deliverables
- Hybrid platform MVP documentation.
- End-to-end regression tests for activation and `/v1/models` visibility.
- Regression tests preserving local-only behavior.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for the hybrid operator flow **(REQUIRED)**

## Tests
- Unit tests:
  - [x] Local-only `/models` and `/v1/models` responses remain compatible with existing tests.
  - [x] Platform activation records a running runtime state and model availability.
  - [x] Smart proxy excludes platform backends until proxy eligibility is enabled.
- Integration tests:
  - [x] Operator flow: detected platform, start platform, fake sidecar `/v1/models`, AutoManager `/v1/models` includes sidecar model.
  - [x] Documentation mentions no AutoManager provider credential collection.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- The MVP success metric is covered by an automated regression test.
- Documentation states the detection and credential boundaries clearly.
