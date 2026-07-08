# Modo Proxy Inteligente — Lista de Tarefas

## Tarefas

| # | Título | Status | Complexidade | Dependências |
|---|--------|--------|--------------|--------------|
| 01 | Configuração e schemas do proxy (smart_proxy global + flags por modelo) | completed | medium | — |
| 02 | ProxyRouter: extração de afinidade e sessões sticky (TTL + persistência) | completed | medium | task_01 |
| 03 | ProxyRouter: elegibilidade, seleção least-busy, contadores e reassign | completed | high | task_02 |
| 04 | Integração do roteamento no catch-all /v1 | completed | high | task_03 |
| 05 | Reescrita SSE por linha e headers de telemetria | completed | medium | task_04 |
| 06 | Endpoints administrativos /proxy/* e /models/proxy | completed | medium | task_01, task_03 |
| 07 | UI: controles de configuração do proxy | completed | medium | task_06 |
| 08 | UI: painel de monitoramento "Proxy Inteligente" | completed | medium | task_06 |
| 09 | Observabilidade (logs [proxy]) e testes de integração ponta a ponta | completed | medium | task_04, task_05, task_06 |
