# Hybrid Cloud Platform Models for AutoManager — Task List

## Tasks

| # | Title | Status | Complexity | Dependencies |
|---|-------|--------|------------|--------------|
| 01 | Hybrid backend identity and config foundations | completed | medium | — |
| 02 | Startup platform detection and catalog service | completed | medium | task_01 |
| 03 | CLIProxyAPI sidecar lifecycle and platform start/stop API | completed | high | task_01, task_02 |
| 04 | Hybrid `/models`, `/status`, and `/v1` model availability | completed | high | task_01, task_02, task_03 |
| 05 | Smart proxy support for platform backends | completed | high | task_01, task_03, task_04 |
| 06 | Frontend platform cards, start flow, and proxy controls | completed | high | task_04, task_05 |
| 07 | Hybrid flow hardening, documentation, and regression coverage | completed | medium | task_02, task_03, task_04, task_05, task_06 |
