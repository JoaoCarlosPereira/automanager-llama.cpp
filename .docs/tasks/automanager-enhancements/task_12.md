---
status: pending
title: OFFLINE initial state + GPU meter dimming
type: frontend
complexity: medium
dependencies:
  - task_08
---

# OFFLINE initial state + GPU meter dimming

## Overview

Implement accurate status indicators so the dashboard shows OFFLINE by default when no model is running. This implements PRD Feature 4 and TechSpec Section 5.4. Changes the initial HTML state, adds OFFLINE→ONLINE transitions, and dims GPU meters when no process is active.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 5.4.
- Focus on WHAT: dashboard shows OFFLINE when no process is running.
- Do NOT change existing status update logic — only add initial state and dimming.
- Tests must pass before considering this task complete.

## Requirements

1. MUST set initial HTML state to OFFLINE (slate text + dot, no glow/pulse) in the dashboard.
2. MUST ensure `updateStatus()` always sets correct class: `status-online` when running, `status-offline` when not.
3. MUST add `dimmed` CSS class for GPU meters when OFFLINE.
4. MUST hide or disable "ABRIR CHAT" button when OFFLINE.
5. MUST hide active-model-card or show "No model loaded" when OFFLINE.
6. MUST implement dual verification: client-side fetch('/status') + initial OFFLINE state.
7. MUST NOT change existing polling intervals (2s metrics, 3s status, 5s models).
8. MUST NOT change status text labels (ONLINE, REALOCANDO..., FALHA) — only fix initial state.

## Subtasks

- [ ] Update _build_html() in ui_renderer.py to show OFFLINE as initial status badge
- [ ] Add .status-offline CSS class (slate text, gray dot, no glow)
- [ ] Add .status-online CSS class (gradient, emerald glow, pulse)
- [ ] Modify updateStatus() JS to always set correct class based on data.running
- [ ] Add .dimmed CSS class for GPU meters
- [ ] Modify updateMetrics() JS to add/remove .dimmed class based on running state
- [ ] Hide/disable "ABRIR CHAT" button when OFFLINE
- [ ] Show "No model loaded" in active-model-card when OFFLINE
- [ ] Write unit tests for UI state transitions
- [ ] Write integration test: verify page loads with OFFLINE state

## Implementation Details

### File Paths to Modify
- `ui_renderer.py` — update _build_html() HTML structure and CSS
- `ui_renderer.py` — update inline JavaScript (updateStatus, updateMetrics functions)

### Integration Points
- `initDashboard()` JS → calls updateStatus() immediately, initial state is OFFLINE
- `updateStatus()` JS → reads data.running from /status response
- `updateMetrics()` JS → checks running state before updating meters

### Relevant Files
- `ui_renderer.py` — HTML template, CSS, JS
- `tests/unit/test_ui_renderer_new.py` — new test file

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-006](adrs/adr-006.md) — Status OFFLINE — Dual Verification

## Deliverables

- Dashboard loads with OFFLINE state
- Status transitions correctly (OFFLINE → ONLINE → OFFLINE)
- GPU meters dimmed when OFFLINE
- "ABRIR CHAT" disabled when OFFLINE
- 3+ new tests

## Tests

### Unit Tests
- [ ] `test_render_dashboard_initial_state_is_offline` — verify HTML contains status-offline class
- [ ] `test_status_badge_online_class_when_running` — verify status-online class when data.running=true
- [ ] `test_status_badge_offline_class_when_not_running` — verify status-offline class when data.running=false

### Integration Tests
- [ ] `test_status_offline_initial` — verify GET / returns HTML with status-offline class
- [ ] `test_gpu_meters_dimmed_when_offline` — verify .dimmed class applied to meters when not running

## Success Criteria

- All new tests passing (5+)
- Dashboard loads with OFFLINE state (slate color, no glow)
- Status transitions work correctly
- GPU meters dim when no process running
- All existing tests still passing
