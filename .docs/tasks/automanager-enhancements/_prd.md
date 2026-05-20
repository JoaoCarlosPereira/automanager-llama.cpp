# Product Requirements Document: Automanager Llama.cpp Enhancements

**Slug:** automanager-enhancements  
**Date:** 2026-05-19  
**Status:** Draft — Pending Approval  
**Project:** Automanager Llama.cpp  
**Author:** AI Assistant  
**Language:** English

---

## 1. Overview

The Automanager Llama.cpp is a FastAPI-based web control plane for orchestrating `llama-server` instances with multi-GPU tensor split management, OOM auto-recovery, and real-time hardware monitoring. Currently in alpha, the project consists of a single ~2400-line monolithic Python file (`llama_manager.py`) targeting a specific hardware setup (RTX 3090 + Tesla P100) on Linux.

This PRD defines six enhancements aimed at improving hardware isolation, observability, visual identity, deployment automation, and documentation. The changes will be implemented through a **modular refactoring strategy** that extracts the monolith into dedicated modules before applying each feature.

---

## 2. Goals

| # | Goal | Metric |
|---|------|--------|
| G1 | Guarantee that `llama-server` never uses GPUs outside explicit user selection | Zero unauthorized GPU access incidents |
| G2 | Persist all terminal/server logs to project-local files for auditability and debugging | Logs available in `logs/` directory, with rotation |
| G3 | Incorporate the visual identity of the `design/` portfolio into the manager UI | Consistent use of Pac-Man theme gradients, background animation, and card styles |
| G4 | Eliminate false "Active"/"Online" status indicators when no model is running | Status UI accurately reflects running state (OFFLINE by default) |
| G5 | Provide a one-command Quick-Install script for zero-touch deployment | `./setup.sh` installs all dependencies and configures systemd in under 5 minutes |
| G6 | Deliver a comprehensive, professionally structured README reflecting actual functionality | README includes TOC, quick-start, features table, architecture diagram, API reference, troubleshooting |

---

## 3. User Stories

### GPU Enforcement
- **US-1:** As an administrator, I want to explicitly select which GPUs process model tensors, so that I prevent unwanted GPUs from being used.
- **US-2:** As an administrator, I want to be confident that disabled GPUs are invisible to the `llama-server` process, so that I can reliably manage VRAM allocation across mixed hardware.

### Log Persistence
- **US-3:** As a developer, I want server and manager logs saved to files within the project directory, so that I can review them without SSH access.
- **US-4:** As an administrator, I want logs to rotate automatically, so that disk space is not consumed indefinitely.

### Design Integration
- **US-5:** As a user, I want the manager UI to share the visual identity of the portfolio design, so that the experience feels cohesive and professional.
- **US-6:** As a user, I want subtle background animations and consistent gradient styling, so that the interface feels modern without being distracting.

### Status Accuracy
- **US-7:** As an operator, I want the dashboard to show OFFLINE when no model is running, so that I am not misled into thinking the server is active.
- **US-8:** As an operator, I want all status indicators (badges, meters, labels) to be synchronized with the actual process state.

### Quick-Install
- **US-9:** As a sysadmin, I want a single script that installs all Python dependencies, verifies `llama-server` and `nvidia-smi` are available, and sets up the systemd service, so that I can deploy the manager in under 5 minutes.
- **US-10:** As a sysadmin, I want the installer to validate prerequisites before installing, so that I get clear error messages instead of cryptic failures.

### Documentation
- **US-11:** As a new user, I want a comprehensive README with a table of contents, quick-start guide, and API reference, so that I can understand and deploy the project without reading source code.
- **US-12:** As a contributor, I want the README to accurately reflect the current feature set and installation steps, so that I don't waste time on outdated information.

---

## 4. Core Features

### Feature 1: Strict GPU Enforcement via CUDA_VISIBLE_DEVICES

**Problem:** Currently, the system builds a `--tensor-split` array for ALL detected GPUs (indices 0 through max_idx). GPUs marked `active=False` or with `weight=0` receive a split value of `0.0000`, but `llama-server` can still enumerate them and potentially attempt allocation.

**Requirements:**
- GPUs with `active=False` or `weight=0` must be completely excluded from the `--tensor-split` array.
- The `CUDA_VISIBLE_DEVICES` environment variable must be set to only the indices of GPUs with `active=True` and `weight > 0`.
- This ensures `llama-server` literally cannot see disabled GPUs.
- Example: if GPU 0 and GPU 2 are active (indices 0 and 1 in the filtered view), and GPU 1 is inactive:
  - `CUDA_VISIBLE_DEVICES=0,1`
  - `--tensor-split=0.7,0.3` (only 2 values)
- If no GPUs are active or all weights are 0, the start operation must be rejected with a clear error message.

**Non-Goals:**
- UI lockdown while model is running (deferred per user choice).
- Dynamic GPU toggling without restart (existing behavior to be preserved for now).

---

### Feature 2: Project-Local Log Files with Rotation

**Problem:** Logs are currently written only to hardcoded system paths (`/root/llama_server.log`, `/root/manager.log`) with no rotation. They are inaccessible without SSH and can grow unbounded.

**Requirements:**
- Create a `logs/` directory within the project root.
- Write server logs to `logs/server.log` and manager logs to `logs/manager.log`.
- Implement Python `logging.RotatingFileHandler` for both log files:
  - Max file size: 10 MB
  - Backup count: 3
- Retain the existing system log paths as secondary destinations (dual-write) for backward compatibility.
- Log rotation must be automatic and transparent — no user action required.

**Non-Goals:**
- Log aggregation or remote log shipping (Syslog, ELK, etc.).
- Log viewer UI within the dashboard (existing SSE stream remains unchanged).

---

### Feature 3: Selective Design Integration (Overlay Components)

**Problem:** The manager UI (Tailwind CSS, embedded in `llama_manager.py`) and the portfolio design (`design/`, Bootstrap 5.2.3, Pac-Man theme) are visually disconnected. A full migration to Bootstrap is too risky.

**Approach chosen by user:** Overlay — keep the existing UI structure and selectively incorporate design elements.

**Requirements:**
- **Background animation:** Embed the Pac-Man Canvas animation (`scripts.js`) as a subtle, semi-transparent background layer behind the dashboard content. Respect `prefers-reduced-motion`.
- **Gradient styling:** Apply the design's blue-to-pink gradient (`#1e30f3 → #e21e80`) to:
  - Primary action buttons (Start, Load, Download)
  - Status badge (ONLINE uses gradient, OFFLINE uses neutral slate)
  - Section headers and headings
  - Progress bars in GPU metrics
- **Card styling:** Adopt the portfolio card design (rounded corners `rounded-4`, generous padding, shadow) for status and GPU monitor cards.
- **Typography:** Optionally integrate the "Plus Jakarta Sans" font for headings as a visual accent.
- **Implementation:** All design elements must be inline (no external Bootstrap dependency). CSS will be extracted from `design/css/styles.css` and merged into the Tailwind CDN configuration within `llama_manager.py`.

**Non-Goals:**
- Migrating the entire UI to Bootstrap.
- Adding portfolio-specific components (profile cards, contact cards, etc.).
- Integrating the "Miriam AI" widget or games (Snake, arcade).

---

### Feature 4: Accurate Status Indicators (OFFLINE by Default)

**Problem:** The dashboard can display "ONLINE", "REALOCANDO...", or other active status states even when no model process is running. This creates confusion about the actual server state.

**Requirements:**
- Initial page load state must be OFFLINE with a neutral visual indicator (slate text + dot, no glow/pulse).
- Status transitions must be: OFFLINE → STARTING → ONLINE → (REALOCANDO → ONLINE/FALHA) → STOPPING → OFFLINE.
- All status-related DOM elements must be synchronized to the actual `llama-server` process state:
  - `status-badge`: Shows OFFLINE when `GET /status` returns `running: false`.
  - `active-model-card`: Hidden or shows "No model loaded" when no process is running.
  - GPU metrics meters: Dimmed/disabled visual state when OFFLINE.
  - "ABRIR CHAT" button: Disabled or hidden when OFFLINE.
- The polling mechanism (`GET /status` every 3s) must reliably detect state changes within 3 seconds.

**Non-Goals:**
- WebSocket-based real-time status (existing polling is sufficient).
- Email/push notifications on state changes.

---

### Feature 5: Quick-Install Script (Bash)

**Problem:** Deployment currently requires manual steps: SSH to server, clone repo, install Python packages, configure systemd, verify paths. There is no zero-touch deployment path.

**Requirements:**
- Create `setup.sh` in the project root.
- The script must:
  1. Detect OS and exit with error if not Linux (Ubuntu/Debian).
  2. Verify and prompt for `sudo` privileges.
  3. Install system dependencies: `python3-pip`, `python3-venv`, `nvidia-utils-<version>` (if not present), `curl`, `git`.
  4. Verify `llama-server` is in PATH or detect its location and warn if missing.
  5. Verify `nvidia-smi` is accessible and at least one GPU is detected.
  6. Create a Python virtual environment in `.venv/`.
  7. Run `pip install -r requirements.txt` inside the venv.
  8. Create required directories: `/root/` config files, `logs/` in project.
  9. Generate a systemd service unit file at `/etc/systemd/system/llama-manager.service`.
  10. Enable and start the service.
  11. Run a health check: `curl localhost:8000/status` and report success/failure.
- Script must be idempotent (safe to run multiple times).
- Script must exit with clear error messages at each validation step.
- Script must print a summary of what was installed and how to access the dashboard.

**Non-Goals:**
- Windows or macOS support (project is Linux-only per design).
- Docker containerization.
- Automated `llama-server` binary download (user must provide the binary).
- Configuration of GPU drivers or CUDA toolkit (assumes already installed).

---

### Feature 6: Complete README Redesign

**Problem:** The current README is a simple markdown file with basic feature descriptions. It mentions 128K context as default but the code uses 65536. It lacks installation steps, API reference, and architecture overview.

**Requirements:**
- Restructure the README with the following sections:
  1. **Project header** with emoji banner and tagline
  2. **Badges** (Alpha status, Python version, License placeholder)
  3. **Table of Contents** (auto-generated structure with anchor links)
  4. **Quick Start** (3-step: clone, install via setup.sh, access dashboard)
  5. **Features** (table format: Feature | Description | Status)
  6. **Architecture** (text diagram showing Manager → llama-server → GPU flow)
  7. **API Reference** (table: Method | Path | Description — matching the canonical API table from CLAUDE.md)
  8. **Hardware Requirements** (GPU list, Linux distro, nvidia-smi)
  9. **Installation** (detailed steps: prerequisites, Quick-Install, manual install, systemd configuration)
  10. **Configuration** (config file path, model settings, log paths)
  11. **Troubleshooting** (common issues with solutions)
  12. **Development** (testing, code structure, contributing guidelines)
- Correct all factual inaccuracies (context size default, feature capabilities).
- Ensure all installation steps match the Quick-Install script behavior.
- Language: English. Tone: professional, technical, consistent with existing project artifacts.

**Non-Goals:**
- Multi-language translations.
- Contribution guidelines file (separate from README).
- Changelog or version history.

---

## 5. User Experience

### Dashboard Before/After (Feature 4)

**Before:** Page loads with potentially stale "ONLINE" status from previous session. GPU metrics show values from last poll regardless of process state.

**After:** Page loads with clean OFFLINE state. All GPU meters are dimmed. Status badge shows "OFFLINE" in slate color. "Start Model" controls are the primary call-to-action. Status transitions are animated and clear.

### Quick-Install Experience

**Before:** User must SSH, manually install packages, create systemd unit, start service, verify.

**After:** User runs `./setup.sh`, answers "yes" to privilege confirmation, waits ~3 minutes, receives dashboard URL.

### Log Access Experience

**Before:** Logs accessible only via SSE stream in browser or SSH to `/root/llama_server.log`.

**After:** Logs available in `logs/server.log` and `logs/manager.log` on local filesystem, with automatic rotation at 10MB.

---

## 6. Non-Goals

The following are explicitly out of scope for this PRD:

- Migration of the entire UI to Bootstrap (overlay approach only).
- Windows/macOS support for Quick-Install (Linux-only).
- Docker containerization.
- Log aggregation, remote logging, or log viewer UI enhancements.
- Dynamic GPU toggling without restart (existing behavior preserved).
- WebSocket-based real-time status (polling remains).
- Automated `llama-server` binary download.
- GPU driver or CUDA toolkit installation.
- Multi-language support for the UI or README.
- CI/CD pipeline setup.
- Database migration (JSON config remains).

---

## 7. Phased Rollout Plan

### Phase 1: Modular Foundation (Refactoring)
Extract the monolithic `llama_manager.py` into dedicated modules:
- `gpu_manager.py` — GPU detection, metrics, tensor split logic
- `log_manager.py` — Log streaming, file handlers, rotation
- `ui_renderer.py` — HTML generation, CSS/JS embedding
- `process_manager.py` — llama-server process lifecycle
- `config_manager.py` — Config persistence (extracted from existing ConfigManager)
- `installer/` — Quick-Install script and templates
- `llama_manager.py` — Slimmed entry point (API routes, FastApp setup)

**Deliverables:** All modules, existing tests passing, no feature changes.

### Phase 2: Core Features
- Implement GPU strict enforcement (Feature 1) in `gpu_manager.py`.
- Implement log file persistence with rotation (Feature 2) in `log_manager.py`.
- Implement accurate status indicators (Feature 4) in `ui_renderer.py` + JS.

**Deliverables:** 3 features complete, tests added/updated.

### Phase 3: Visual & Deployment
- Implement design overlay components (Feature 3) in `ui_renderer.py`.
- Create Quick-Install script (Feature 5) in `installer/setup.sh`.
- Redesign README (Feature 6).

**Deliverables:** All 6 features complete, full test suite passing.

### Phase 4: Validation & Documentation
- End-to-end testing on target hardware (RTX 3090 + Tesla P100).
- Quick-Install script testing on clean Ubuntu/Debian.
- README accuracy audit.
- Final review and release preparation.

**Deliverables:** Tested release, updated documentation.

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| GPU isolation | 100% of disabled GPUs invisible to llama-server | `nvidia-smi` process list verification |
| Log availability | 100% of logs written to `logs/` directory | File existence and content verification |
| Log rotation | Files never exceed 30MB total | File size monitoring |
| Status accuracy | OFFLINE state correct within 3s of process stop | Automated status polling test |
| Quick-Install success rate | 100% on clean Ubuntu 22.04+ | Script execution on VM/containe |
| Quick-Install time | Under 5 minutes (excl. system package download) | Timing measurement |
| README completeness | All 12 sections present and accurate | Manual audit |

---

## 9. Risks and Mitigations

| # | Risk | Impact | Mitigation |
|---|------|--------|-----------|
| R1 | Modular refactoring breaks existing integrations | High | Comprehensive test suite before and after refactoring; incremental PRs |
| R2 | `CUDA_VISIBLE_DEVICES` changes break existing GPU detection logic | High | Thorough testing on multi-GPU hardware; rollback plan via git |
| R3 | Log file permission issues on target system | Medium | Create log directory with appropriate permissions in startup; fall back to existing paths |
| R4 | Design overlay conflicts with Tailwind utility classes | Low | Use prefixed CSS classes; test on multiple screen sizes |
| R5 | Quick-Install script fails on non-standard Ubuntu/Debian versions | Medium | Support Ubuntu 22.04+ and Debian 11+; clear error messages for unsupported versions |
| R6 | Canvas animation impacts dashboard performance | Medium | Throttle animation FPS; respect `prefers-reduced-motion`; make it optional via config |
| R7 | Dual-write logs cause confusion during migration | Low | Document clearly in README; add deprecation note for system paths |

---

## 10. Architecture Decision Records

| ADR | Title | Summary |
|-----|-------|---------|
| ADR-001 | Modular Refactoring Strategy | Extract monolith into dedicated modules before implementing features |

---

## 10.5. License

**Decision:** The project will adopt the **Apache License 2.0**.

**Rationale:**
- Apache 2.0 is the most widely adopted open-source license for server-side and infrastructure projects.
- It is permissive: allows free use, modification, distribution, and commercial use.
- It provides an explicit grant of patent rights from contributors to users (unlike MIT/BSD).
- It requires preservation of copyright notices and a NOTICE file (if applicable).
- It is compatible with the project's "alpha, use at your own risk" positioning — no warranty, but clear usage rights.
- The license file (`LICENSE`) must be added to the project root alongside this PRD's implementation.

**Implementation:**
- Add `LICENSE` file with full Apache 2.0 text.
- Add `LICENSE-APACHE` and `NOTICE` placeholders if needed.
- Reference the license in the README under a dedicated "License" section.
- Add an Apache 2.0 SPDX identifier to `requirements.txt` or a dedicated `pyproject.toml`/`setup.py` if project metadata is added later.

---

## 11. Open Questions

| # | Question | Context |
|---|----------|---------|
| O1 | Should the Quick-Install script backup existing config files before overwriting? | The script creates systemd unit and config directories — existing configs should be preserved. |
| O2 | Should the Pac-Man background animation be enabled by default or opt-in via config? | Performance consideration — heavy canvas animation may impact dashboard responsiveness on low-end hardware. |
| O3 | Should the dual-write log approach (system paths + project logs) be temporary or permanent? | Backward compatibility vs. long-term simplicity. |
| O4 | Are there additional hardware configurations beyond RTX 3090 + Tesla P100 that need testing? | GPU detection and split logic should work generically, but specific configs may need validation. |
| O5 | Should the installer detect and configure the correct NVIDIA driver version? | Out of scope per Non-Goals, but should provide clear guidance if driver is missing. |
