---
status: pending
title: Quick-Install script, README redesign, LICENSE
type: docs
complexity: high
dependencies:
  []
---

# Quick-Install script, README redesign, LICENSE

## Overview

Create the Quick-Install bash script (`installer/setup.sh`), redesign the README with 12 sections, and add the Apache 2.0 LICENSE file. This implements PRD Features 5 and 6 and TechSpec Section 5.5/5.6. Independent of other tasks — no code dependencies.

## <critical>

- Read the PRD and TechSpec before starting. Reference TechSpec Sections 5.5 and 5.6.
- Focus on WHAT: installation script, comprehensive README, license file.
- The README must accurately reflect ALL features implemented by this project (including the new ones).
- The setup.sh must be idempotent and work on Ubuntu 22.04+ / Debian 11+.

## Requirements

### Quick-Install Script (setup.sh)
1. MUST create `installer/setup.sh` in the project root.
2. MUST detect OS and exit with error if not Linux (Ubuntu/Debian).
3. MUST verify and require `sudo` privileges.
4. MUST install system dependencies: `python3-pip`, `python3-venv`, `curl`, `git`, `lsb-release`.
5. MUST verify `llama-server` is in PATH (warn if missing).
6. MUST verify `nvidia-smi` is accessible and at least one GPU detected.
7. MUST create Python venv in `.venv/`.
8. MUST run `pip install -r requirements.txt` inside venv.
9. MUST create `logs/` directory and `/root/` config directory.
10. MUST generate systemd service unit file at `/etc/systemd/system/llama-manager.service`.
11. MUST enable and start the systemd service.
12. MUST run health check: `curl localhost:8000/status` and report success/failure.
13. MUST be idempotent (safe to run multiple times).
14. MUST print a summary of what was installed and dashboard URL.

### README Redesign
1. MUST restructure README with 12 sections as defined in PRD Section 4 Feature 6.
2. MUST include: header, badges, TOC, quick-start, features table, architecture diagram, API reference, hardware requirements, installation, configuration, troubleshooting, development.
3. MUST correct all factual inaccuracies (context size default is 65536, not 128K).
4. MUST reference the Quick-Install script in installation steps.
5. MUST include API reference table matching CLAUDE.md.
6. MUST include architecture text diagram.

### LICENSE
1. MUST create `LICENSE` file with full Apache License 2.0 text.
2. MUST reference the license in README under a dedicated "License" section.

## Subtasks

- [ ] Create `installer/setup.sh` with all 14 requirements
- [ ] Make setup.sh executable: `chmod +x installer/setup.sh`
- [ ] Create `README.md` with all 12 sections
- [ ] Add Apache 2.0 `LICENSE` file
- [ ] Verify setup.sh syntax: `bash -n installer/setup.sh`
- [ ] Verify README renders correctly (markdown validation)
- [ ] Verify LICENSE contains full Apache 2.0 text

## Implementation Details

### File Paths to Create
- `installer/setup.sh` — new file (~150 lines)
- `README.md` — overwritten with redesigned version
- `LICENSE` — new file (Apache 2.0)

### File Paths to Modify
- `README.md` — complete redesign
- `LICENSE` — new file

### Integration Points
- `README.md` → references `installer/setup.sh` and all API endpoints
- `LICENSE` → top-level project file
- `setup.sh` → installs from `requirements.txt`, creates `logs/` dir

### Relevant Files
- `CLAUDE.md` — API reference table to reference
- `requirements.txt` — dependency list for setup.sh
- `start_llama.sh` — existing script (compare for reference)
- `design/design.md` — design system docs

### Related ADRs
- [ADR-001](adrs/adr-001.md) — Modular Refactoring Strategy
- PRD Section 10.5 — License (Apache 2.0)

## Deliverables

- `installer/setup.sh` — executable, idempotent, tested
- `README.md` — 12-section comprehensive guide
- `LICENSE` — Apache 2.0 full text

## Tests

- [ ] `bash -n installer/setup.sh` — syntax validation passes
- [ ] Manual review: setup.sh runs correctly on Ubuntu 22.04 (if available)
- [ ] Verify README has all 12 sections
- [ ] Verify LICENSE contains "Apache License, Version 2.0"
- [ ] Verify no factual inaccuracies in README (context size, features, etc.)

## Success Criteria

- setup.sh syntax validates
- README has all 12 sections with accurate content
- LICENSE file contains full Apache 2.0 text
- All installation steps in README match setup.sh behavior
- API reference table accurate
