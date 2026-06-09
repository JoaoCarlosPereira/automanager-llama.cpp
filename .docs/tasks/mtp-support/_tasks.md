# Suporte MTP — Lista de Tarefas

## Tarefas

| # | Título | Status | Complexidade | Dependências |
|---|--------|--------|--------------|--------------|
| 01 | Extender schemas.py com campos MTP | completed | low | — |
| 02 | Persistir mtp_enabled/mtp_draft_tokens no config_manager | completed | low | task_01 |
| 03 | Implementar detect_model_mtp() no gpu_manager | completed | low | — |
| 04 | Implementar mtp_cli_args() e integrar no process_manager.start | completed | medium | task_01, task_03 |
| 05 | Propagar MTP em llama_manager (/start, /status, auto-start) | completed | medium | task_01, task_02, task_04 |
| 06 | Injetar campos MTP no HTML do dashboard | completed | low | task_05 |
| 07 | Atualizar gpu.js (badge, reset, listeners) | completed | low | task_06 |
| 08 | Atualizar models.js (applyModelConfig, startModel) | completed | medium | task_07 |
| 09 | Atualizar metrics.js (sync config running) | completed | low | task_07 |
| 10 | Propagar MTP no fluxo de auto-balance | completed | medium | task_04, task_05 |
