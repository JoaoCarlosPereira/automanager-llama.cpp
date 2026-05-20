# Automanager Enhancements — Task List

## Tasks

| # | Title | Status | Complexity | Dependencies |
|---|-------|--------|------------|--------------|
| 01 | Extract config_manager.py | completed | low | — |
| 02 | Extract gpu_manager.py | completed | low | task_01 |
| 03 | Extract log_manager.py | completed | low | — |
| 04 | Extract process_manager.py | completed | medium | task_02, task_03 |
| 05 | Extract model_manager.py | completed | low | task_01 |
| 06 | Extract ui_renderer.py | cancelled | medium | task_02, task_04 |
| 07 | Refactor llama_manager.py entry point | completed | critical | task_01–06 |
| 08 | Fix tests after refactoring | completed | medium | task_07 |
| 09 | GPU strict enforcement via CUDA_VISIBLE_DEVICES | completed | high | task_08 |
| 10 | Project-local log files with rotation | completed | medium | task_08 |
| 11 | Update SSEStreamer to project-local path | completed | low | task_10 |
| 12 | OFFLINE initial state + GPU meter dimming | completed | medium | task_08 |
| 13 | Pac-Man background + gradient CSS overlay | completed | medium | task_07 |
| 14 | Quick-Install script, README redesign, LICENSE | completed | high | — |
| 15 | Full test suite + E2E validation | completed | medium | task_09, task_10, task_12, task_13, task_14 |

## Notes

- **task_06** deferred: UI remains embedded in `llama_manager.py` (~1.4k lines); extraction is optional follow-up.
- **task_15**: 66 unit/integration tests pass locally (`pytest -q`). E2E on Linux GPU host remains manual.

## Dependency Graph

```
task_01 ──→ task_02 ──→ task_04 ──→ task_06 ──→ task_07 ──→ task_08 ──→ task_09 ──→ task_15
  │           │           │                             │
  │           │           │                             └─────────────→ task_13 ──→ task_15
  │           │           └─────────────────────────────────────────────→ task_12
  │           └─────────────────────────────────────────────────────────→ task_15
  └─────────────────────────────────────────────────────────────────────→ task_05
                                                                          
task_03 ──→ task_04 ────────────────────────────────────────────────────→ task_07
  │           │
  │           └─────────────────────────────────────────────────────────→ task_10 ──→ task_11 ──→ task_15
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                                                          
task_08 ──→ task_10 ────────────────────────────────────────────────────→ task_15
  │           │
  └───────────→ task_12 ────────────────────────────────────────────────→ task_15
                                                                          
task_14 (independent) ──────────────────────────────────────────────────→ task_15
```

## Phases

- **Phase 1 (Refactoring):** task_01 → task_08 — done
- **Phase 2 (Core Features):** task_09 → task_12 — done
- **Phase 3 (Visual & Deployment):** task_13 → task_14 — done
- **Phase 4 (Validation):** task_15 — tests green locally
