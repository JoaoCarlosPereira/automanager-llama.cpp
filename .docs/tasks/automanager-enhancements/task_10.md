---
status: pending
title: Project-local log files with rotation
type: backend
complexity: medium
dependencies:
  - task_08
---

# Project-local log files with rotation

## Overview

Implement project-local log persistence with automatic rotation. This implements PRD Feature 2 and TechSpec Section 5.2. Creates a `logs/` directory, writes logs with `RotatingFileHandler` (10MB max, 3 backups), while maintaining backward compatibility with existing system log paths.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Section 5.2.
- Focus on WHAT: logs written to `logs/` directory with rotation.
- Maintain backward compatibility — existing system paths still receive output.
- Every change MUST have corresponding tests.

## Requirements

1. MUST create `logs/` directory at project root on LogManager initialization.
2. MUST add `RotatingFileHandler` to the "automanager" logger in `LogManager.setup_logging()`.
3. MUST configure `RotatingFileHandler` with `maxBytes=10*1024*1024` (10MB) and `backupCount=3`.
4. MUST write to `logs/manager.log` via the rotating handler.
5. MUST redirect `llama-server` stdout to `logs/server.log` in `ProcessManager.start()`.
6. MUST keep existing system log paths (`/root/manager.log`, `/root/llama_server.log`) as secondary destinations (dual-write).
7. MUST add `LogManager.get_server_log_path()` returning project-local path.
8. MUST add `LogManager.rotate_server_log()` method.
9. MUST add `LogManager.clear_server_log()` method.
10. MUST NOT change any logging format or log level.

## Subtasks

- [ ] Add log path constants to log_manager.py: LOGS_DIR, SERVER_LOG_PATH, MANAGER_LOG_PATH, MAX_LOG_SIZE, LOG_BACKUP_COUNT
- [ ] Add os.makedirs(self._logs_dir, exist_ok=True) in LogManager.__init__
- [ ] Add RotatingFileHandler to LogManager.setup_logging()
- [ ] Add LogManager.get_server_log_path() returning logs/server.log
- [ ] Add LogManager.rotate_server_log() method
- [ ] Add LogManager.clear_server_log() method
- [ ] Update ProcessManager.start() to use log_manager.get_server_log_path() for server log
- [ ] Update ProcessManager.stop() to use log_manager for log cleanup
- [ ] Write unit tests for LogManager
- [ ] Verify log rotation behavior

## Implementation Details

### File Paths to Create
- `logs/` — directory (created at runtime)
- None (all changes to existing files)

### File Paths to Modify
- `log_manager.py` — add constants, new methods, RotatingFileHandler
- `process_manager.py` — update log path references

### Integration Points
- `ProcessManager.start()` → uses log_manager.get_server_log_path()
- `ProcessManager.stop()` → uses log_manager.clear_server_log()
- OOMWatchdog._check_log() → reads from log_manager path (via SERVER_LOG_PATH constant)

### Relevant Files
- `log_manager.py` — new methods and constants
- `process_manager.py` — updated log path references

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- [ADR-004](adrs/adr-004.md) — Log Rotation — Hybrid Handler Approach

## Deliverables

- `logs/` directory created at startup
- RotatingFileHandler for manager.log
- Server log redirected to project-local path
- All existing tests passing
- 5+ new unit tests

## Tests

### Unit Tests
- [ ] `test_log_manager_creates_logs_directory` — verify logs/ dir created on init
- [ ] `test_log_manager_setup_logging_adds_rotating_handler` — verify handler added
- [ ] `test_log_manager_get_server_log_path_returns_project_local` — verify path
- [ ] `test_log_rotation_creates_backup` — verify .1 file created when size exceeded
- [ ] `test_log_rotation_respects_backup_count` — verify max 3 backup files

## Success Criteria

- All new tests passing (5+)
- All existing tests still passing
- `logs/` directory created on app startup
- Log files rotate at 10MB with 3 backups
- Server log written to `logs/server.log`
- Manager log written to `logs/manager.log` with rotation
