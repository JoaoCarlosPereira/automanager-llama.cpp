---
status: pending
title: Pac-Man background + gradient CSS overlay
type: frontend
complexity: medium
dependencies:
  - task_07
---

# Pac-Man background + gradient CSS overlay

## Overview

Integrate the Pac-Man canvas animation as a subtle background layer and apply the design's blue-to-pink gradient styling to the dashboard. This implements PRD Feature 3 and TechSpec Section 5.3. Uses the overlay approach — no Bootstrap migration, just visual accents.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 5.3.
- Focus on WHAT: canvas as background, gradients on buttons/status/cards.
- Do NOT migrate to Bootstrap. Do NOT add portfolio-specific components.
- Respect `prefers-reduced-motion` for accessibility.
- Tests must pass before considering this task complete.

## Requirements

1. MUST copy Pac-Man canvas animation from `design/js/scripts.js` → `static/js/pacman_bg.js` (extract canvas animation only, remove portfolio-specific code).
2. MUST inject `<canvas id="pacman-background">` at z-index 0 with opacity 0.35 and `pointer-events: none` in `ui_renderer.py`.
3. MUST apply design's gradient (`#1e30f3 → #e21e80`) to primary action buttons.
4. MUST apply gradient to status badge (ONLINE state).
5. MUST apply gradient to section headers and headings.
6. MUST apply gradient to progress bars in GPU metrics.
7. MUST adopt portfolio card styling: rounded corners, generous padding, shadow.
8. MUST respect `prefers-reduced-motion` — render static frame when enabled.
9. MUST NOT add Bootstrap dependency.
10. MUST NOT add profile cards, contact cards, or portfolio-specific components.

## Subtasks

- [ ] Create `static/js/pacman_bg.js` from `design/js/scripts.js` (extract canvas animation only)
- [ ] Inject canvas HTML element into ui_renderer.py with correct CSS positioning
- [ ] Add canvas script tag to HTML (loaded before </body>)
- [ ] Extract gradient CSS variables from design/css/styles.css
- [ ] Apply gradient to primary buttons in Tailwind config
- [ ] Apply gradient to status badge (ONLINE state)
- [ ] Apply gradient to section headers and headings
- [ ] Apply gradient to progress bars
- [ ] Apply card styling (rounded-4, shadow, padding) to dashboard cards
- [ ] Verify prefers-reduced-motion is respected
- [ ] Manual verification: visit dashboard and confirm visual appearance

## Implementation Details

### File Paths to Create
- `static/js/pacman_bg.js` — new file (extracted canvas animation, ~500 lines)

### File Paths to Modify
- `ui_renderer.py` — inject canvas HTML, update CSS, update Tailwind config
- `static/` — directory (created at project root)

### Integration Points
- `_build_html()` → inject canvas before </body>
- Tailwind config → extend colors with primary (#1e30f3) and secondary (#e21e80)
- Dashboard cards → apply rounded-4, shadow classes

### Relevant Files
- `design/js/scripts.js` — source canvas animation
- `design/css/styles.css` — source gradient definitions (lines 1–253)
- `ui_renderer.py` — target for injection

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-005](adrs/adr-005.md) — Pac-Man Background Integration — Fixed Overlay

## Deliverables

- `static/js/pacman_bg.js` with canvas animation
- Canvas visible at z-index 0 with 0.35 opacity
- Gradient styling applied to buttons, status, headers, progress bars
- Card styling applied to dashboard cards
- prefers-reduced-motion respected
- All existing tests passing

## Tests

### Unit Tests
- [ ] `test_inject_pacman_canvas` — verify canvas element present in rendered HTML
- [ ] `test_gradient_css_applied` — verify gradient classes present in rendered HTML

### Integration Tests
- [ ] `test_pacman_canvas_injected` — verify GET / returns HTML with canvas element
- [ ] `test_gradients_render_correctly` — manual verification in browser

## Success Criteria

- All new tests passing (4+)
- Canvas visible as subtle background (not distracting)
- Gradients applied to buttons, status, headers, progress bars
- Card styling consistent with portfolio design
- prefers-reduced-motion respected (static frame rendered)
- All existing tests still passing
- App starts without errors
