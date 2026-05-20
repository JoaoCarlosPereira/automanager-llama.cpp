---
status: pending
title: Update SSEStreamer to project-local path
type: refactor
complexity: low
dependencies:
  - task_10
---

# Update SSEStreamer to project-local path

## Overview

Update the SSE log streaming endpoint to read from the project-local `logs/server.log` instead of the hardcoded `/root/llama_server.log`. This follows naturally from task_10 (log rotation) and ensures the web UI displays the same logs that are written to the project directory.

## <critical>

- Read the PRD and TechSpec before starting.
- Focus on WHAT: SSE stream reads from project-local path.
- Minimal change — only the file path reference needs updating.
- Tests must pass before considering this task complete.

## Requirements

1. MUST update `LogManager.stream_logs()` to read from `log_manager.get_server_log_path()` (which returns `logs/server.log`).
2. MUST NOT change the SSE event format (`data: {line}\n\n`).
3. MUST NOT change the 500-line initial burst or 0.5s polling interval.
4. MUST handle missing file gracefully (return "Arquivo de log nao encontrado." message).
5. MUST NOT change any other SSEStreamer behavior.

## Subtasks

- [ ] Verify LogManager.stream_logs() reads from log_manager.get_server_log_path()
- [ ] Verify SSE endpoint at /logs still works
- [ ] Run existing tests: `pytest tests/integration/test_api_endpoints.py`
- [ ] Manual verification: curl /logs and confirm log lines appear

## Implementation Details

### File Paths to Modify
- `log_manager.py` — update stream_logs() method

### Integration Points
- `@app.get("/logs")` route → calls `log_mgr.stream_logs()`
- Browser JS `startLogs()` function → unchanged (consumes SSE events)

### Relevant Files
- `log_manager.py` — stream_logs() method
- `tests/integration/test_api_endpoints.py` — SSE endpoint test (401 check only)

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-004](adrs/adr-004.md) — Log Rotation — Hybrid Handler Approach

## Deliverables

- SSE stream reads from `logs/server.log`
- No behavioral changes to SSE protocol
- All existing tests passing

## Tests

- Verify `pytest tests/integration/test_api_endpoints.py` passes
- Manual: start a model and verify /logs stream shows server output

## Success Criteria

- SSE stream endpoint functional
- Stream reads from `logs/server.log`
- No regression in SSE behavior
- All existing tests passing
