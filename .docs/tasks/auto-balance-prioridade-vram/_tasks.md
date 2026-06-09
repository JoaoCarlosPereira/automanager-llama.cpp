# Maximização de VRAM por Prioridade no Auto-Balance — Lista de Tarefas

## Tarefas

| # | Título | Status | Complexidade | Dependências |
|---|--------|--------|--------------|--------------|
| 01 | Implementar cascata estrita por MB em `LoadDistributor.distribute` | completed | medium | — |
| 02 | Adaptar `compute_offload_plan` para delegar à cascata e remover caminho proporcional/legado | completed | medium | task_01 |
| 03 | Aposentar probing empírico e lógica proporcional em `auto_balance.py` | completed | high | task_01, task_02 |
| 04 | Backend ignora `pinned` quando `auto_balance=true` | completed | low | task_02 |
| 05 | Frontend limpa pins ao ativar o Auto-Balance | completed | low | task_04 |
