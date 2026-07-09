---
status: completed
title: "Startup platform detection and catalog service"
type: backend
complexity: medium
dependencies:
  - task_01
---

# Task 2: Startup platform detection and catalog service

## Overview
This task adds startup-only detection for Codex, Claude Code, Google Antigravity, and the CLIProxyAPI executable. It exposes an in-memory platform catalog that later API, sidecar, and UI tasks can consume.

<critical>
- ALWAYS READ the PRD and TechSpec before starting
- REFERENCE TECHSPEC for implementation details - do not duplicate here
- FOCUS ON "WHAT" - describe what needs to be accomplished, not how
- MINIMIZE CODE - show code only to illustrate current structure or problem areas
- TESTS REQUIRED - every task MUST include tests in deliverables
</critical>

<requirements>
- Requirement 1: Detection MUST run once during AutoManager startup or module initialization for the MVP.
- Requirement 2: The platform catalog MUST include Codex, Claude Code, and Google Antigravity.
- Requirement 3: Missing or not-ready platforms MUST remain visible with a concise reason.
- Requirement 4: Detection MUST not require provider credential validation or AutoManager-specific login.
- Requirement 5: Platform catalog entries MUST include `backend_id`, `backend_type`, `provider`, display name, status, and reason fields.
</requirements>

## Subtasks
- [x] 2.1 Add platform definitions for the three MVP providers.
- [x] 2.2 Add executable detection for platform tools and CLIProxyAPI.
- [x] 2.3 Add a catalog service that merges detection with persisted platform preferences.
- [x] 2.4 Expose platform catalog state through a testable Python API.
- [x] 2.5 Add unit tests for detected, missing, and preference-merged states.

## Implementation Details
Create a small platform module rather than expanding `model_manager.py` with provider-specific logic. Reference the TechSpec "System Architecture" and "Integration Points" sections for boundaries.

### Relevant Files
- `paths.py` - provides installation/data path conventions.
- `model_manager.py` - owns current local model scan and will be extended in task 04.
- `llama_manager.py` - initializes module-level services and will expose catalog data in task 04.
- `tests/unit/test_model_manager.py` - shows scanner-oriented test patterns.

### Dependent Files
- `config_manager.py` - supplies platform preference helpers from task 01.
- `tests/conftest.py` - provides reusable fixtures for isolated config paths.
- `static/js/models.js` - will render the platform catalog in task 06.

### Related ADRs
- [ADR-001: Unified Catalog MVP for Hybrid Platform Models](adrs/adr-001.md) - Requires platform cards beside local model cards.
- [ADR-005: Startup-Only Platform Detection with Persisted Preferences](adrs/adr-005.md) - Defines startup-only detection behavior.

## Deliverables
- Platform catalog service with startup-only detection.
- Testable detection results for detected and missing providers.
- Preference merge from `platform_configs`.
- Unit tests with 80%+ coverage **(REQUIRED)**
- Integration tests for catalog shape where exposed through service APIs **(REQUIRED)**

## Tests
- Unit tests:
  - [x] Detector marks Codex as detected when a configured executable resolver returns a path.
  - [x] Detector marks Claude Code as missing with a reason when no command is found.
  - [x] Catalog includes Google Antigravity even when missing.
  - [x] Platform preferences override default proxy eligibility without changing detection status.
- Integration tests:
  - [x] A catalog service initialized with mixed detected and missing providers returns all three MVP platform entries.
- Test coverage target: >=80%
- All tests must pass

## Success Criteria
- All tests passing
- Test coverage >=80%
- All required platform integrations appear in the catalog.
- Detection state is in memory and preferences are durable.
