# CPU Offload — Lista de Tarefas

## Tarefas

| # | Título | Status | Complexidade | Dependências |
|---|--------|--------|--------------|--------------|
| 01 | Extender GPUWeight com campo `device` | completed | low | — |
| 02 | Implementar `detect_cpu_info()` no gpu_manager | completed | low | task_01 |
| 03 | Estender `/metrics` com RAM e cpu_name | completed | low | task_02 |
| 04 | Implementar `compute_n_gpu_layers()` e `validate_weights()` | completed | medium | task_01 |
| 05 | Integrar CPU ao Auto Balance com priorização GPU e limite 70% | completed | high | task_01, task_03 |
| 06 | Tornar `--n-gpu-layers` dinâmico no process_manager | completed | medium | task_04, task_05 |
| 07 | Injetar linha da CPU no HTML do dashboard | completed | low | task_02 |
| 08 | Bind de eventos CPU em gpu.js + redistribuição de pesos | completed | medium | task_07 |
| 09 | Atualizar metrics.js para renderizar CPU usage e RAM | completed | low | task_07 |
| 10 | Estender models.js para coletar peso da CPU no startModel() | completed | medium | task_08 |
| 11 | Testes de integração — fluxo completo CPU offload | completed | medium | task_01–task_10 |
| 12 | Validações, tratamento de erros e casos de borda | completed | medium | task_04, task_11 |
