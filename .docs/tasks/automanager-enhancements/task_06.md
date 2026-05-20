---
status: pending
title: Extract ui_renderer.py
type: refactor
complexity: medium
dependencies:
  - task_02
  - task_04
---

# Extract ui_renderer.py

## Overview

Extract the `index()` route handler (lines 1236–1442) and `_build_html()` function (lines 1444–2313) from `llama_manager.py` into a new `ui_renderer.py` module. Create a `UIRenderer` class that encapsulates HTML generation. This is the largest single extraction (~870 lines). No behavioral changes.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 3.5.
- Focus on WHAT: extract exactly as-is. Do NOT modify behavior.
- The `_build_html()` function contains ~870 lines of inline HTML, CSS, and JS — this is the largest extraction.
- Tests must pass before considering this task complete.

## Requirements

1. MUST create `ui_renderer.py` in the project root.
2. MUST create `UIRenderer` class with `render_dashboard()` method wrapping `index()`.
3. MUST extract `_build_html()` as a method of `UIRenderer`.
4. MUST extract all inline JavaScript (lines 1732–2310) into the HTML template — NO behavioral changes to JS.
5. MUST extract all inline CSS (lines 1493–1506) into the HTML template — NO behavioral changes to CSS.
6. MUST NOT change any HTML structure, JS logic, or CSS styles.
7. MUST keep required constants used by the UI: `SERVER_PORT`, `MANAGER_PORT`, `LLAMA_SERVER_BIN`.

## Subtasks

- [ ] Create `ui_renderer.py` with UIRenderer class
- [ ] Copy index() function logic into UIRenderer.render_dashboard()
- [ ] Copy _build_html() function into UIRenderer._build_html()
- [ ] Copy inline JS (lines 1732–2310) — keep exactly as-is
- [ ] Copy inline CSS (lines 1493–1506) — keep exactly as-is
- [ ] Remove index() and _build_html() from `llama_manager.py`
- [ ] Add import statements to `llama_manager.py`
- [ ] Update GET "/" route to call `ui_renderer.render_dashboard()`
- [ ] Run `python3 -m py_compile ui_renderer.py` to verify syntax
- [ ] Verify app starts and renders correctly

## Implementation Details

### File Paths to Create
- `ui_renderer.py` — new file (~900 lines)

### File Paths to Modify
- `llama_manager.py` — remove lines 1236–2313, add imports, update GET "/" route

### Integration Points
- `@app.get("/", response_class=HTMLResponse)` at line 1235 → will call `ui_renderer.render_dashboard()`
- `index()` depends on: model_scanner, gpu_detector, config_manager, process_manager, token_manager, auth_manager
- `_build_html()` depends on: all data passed from `index()`

### Relevant Files
- `llama_manager.py` lines 1236–2313 — source functions to extract
- `llama_manager.py` line 1235 — route decorator to update
- `design/js/scripts.js` — referenced in TechSpec for future Pac-Man integration

### Dependent Files
- `llama_manager.py` — will import from ui_renderer
- `gpu_manager.py` (task_02) — UIRenderer may need GPUInfo
- `process_manager.py` (task_04) — UIRenderer needs process status

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-002](adrs/adr-002.md) — Flat Module Structure in Project Root

## Deliverables

- `ui_renderer.py` with UIRenderer class
- `llama_manager.py` with removed functions and new imports
- App renders the same HTML as before

## Tests

- Verify `python3 -m py_compile ui_renderer.py` succeeds
- Verify `python3 -m py_compile llama_manager.py` succeeds
- Verify app starts without errors
- Manual verification: visit `http://localhost:8000` and confirm UI renders identically

## Success Criteria

- `ui_renderer.py` compiles cleanly
- `llama_manager.py` compiles cleanly
- App starts without errors
- Dashboard UI renders identically to before (pixel-perfect HTML comparison)
- No missing variables, no broken JavaScript, no broken CSS
