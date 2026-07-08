---
status: completed
title: "UI: painel de monitoramento \"Proxy Inteligente\""
type: frontend
complexity: medium
dependencies:
  - task_06
---

# UI: painel de monitoramento "Proxy Inteligente"

## Visão Geral
Adiciona ao dashboard a seção de leitura "Proxy Inteligente" (PRD F9): estado do modo, modelo exposto, lista de backends com papel/GPU/status/requisições ativas e tabela de sessões sticky (tag, backend/GPU, contagem de requisições, último uso, tokens quando disponíveis), atualizada junto ao polling de status existente.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. O card DEVE exibir: modo ON/OFF, modelo exposto, backends (papel principal/secundário, modelo, GPU, estado, in-flight/max) e sessões sticky (chave/tag, backend/GPU, request_count, last_used_at, tokens_processed) — PRD F9.
- 2. A atualização DEVE ocorrer junto ao ciclo de polling existente (`updateStatus()` em `static/js/metrics.js:11+`, 1 s), consultando `GET /proxy/status` e `GET /proxy/sessions` SOMENTE quando o modo estiver ligado (sem tráfego extra com modo OFF).
- 3. Com o modo desligado, o card DEVE mostrar estado compacto "Modo Proxy Inteligente: OFF" sem listas.
- 4. As linhas de sessão DEVEM oferecer ações de intervenção (remover sessão, reassign) chamando os endpoints da tarefa 06, com confirmação via `showConfirm`.
- 5. A renderização DEVE seguir o padrão DOM existente (innerHTML/innerText em elementos do HTML inline, sem framework).
- 6. O contrato de HTML (`tests/unit/test_html_contract.py`) DEVE ser atualizado com o novo card.
</requirements>

## Subtarefas
- [x] 8.1 Adicionar o card "Proxy Inteligente" (containers vazios + cabeçalho) ao HTML inline em `_build_html`.
- [x] 8.2 Implementar em `static/js/proxy.js` a renderização de backends e sessões a partir de `/proxy/status` e `/proxy/sessions`.
- [x] 8.3 Integrar a atualização ao ciclo de `updateStatus()` com guarda de modo ligado.
- [x] 8.4 Implementar ações por sessão (remover, reassign) com confirmação e toast de resultado.
- [x] 8.5 Atualizar contrato de HTML e escrever testes de renderização/fluxo.

## Detalhes de Implementação
Ver PRD F9 (conteúdo do painel, exemplo visual) e TechSpec "Endpoints de API" (payloads de `/proxy/status`, `/proxy/backends`, `/proxy/sessions`). Polling existente: `metrics.js:11-18` popula `state.activeInstances` a cada 1 s; padrão de renderização por DOM em `metrics.js:47-92`. Nome de GPU vem pronto no payload (tarefa 06) — a UI não consulta métricas de GPU.

### Arquivos Relevantes
- `llama_manager.py` — `_build_html` (card novo)
- `static/js/proxy.js` — renderização e ações (mesmo módulo da tarefa 07)
- `static/js/metrics.js` — gancho no ciclo de polling
- `static/js/auth.js` — `apiFetch`/`showConfirm`/`showToast`

### Arquivos Dependentes
- `tests/unit/test_html_contract.py` — contrato do HTML (atualizar)
- `static/js/state.js` — estado compartilhado se necessário para o modo

### ADRs Relacionados
- [ADR-001: Least-busy sticky](../adrs/adr-001.md) — o painel exibe os contadores in-flight que fundamentam a decisão
- [ADR-005: Sessões sticky](../adrs/adr-005.md) — campos exibidos por sessão

## Entregáveis
- Card "Proxy Inteligente" com backends e sessões em tempo real (polling 1 s)
- Ações de intervenção por sessão funcionais
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração do contrato HTML e dos fluxos de intervenção **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [x] HTML gerado contém o card "Proxy Inteligente" com os containers de backends e sessões (contrato atualizado)
  - [x] Payload de `/proxy/status` com 3 backends (1 principal RTX 3090, 2 secundários P100) renderiza papel, GPU e estado correto por linha
  - [x] Payload de `/proxy/sessions` com sessão `agent:sql-reviewer:*` renderiza tag, backend, contagem e último uso
  - [x] Modo OFF → card compacto sem chamadas a `/proxy/sessions` (verificar ausência de fetch)
- Testes de integração:
  - [x] Fluxo remover sessão: DELETE bem-sucedido remove a linha e exibe toast; erro 404 exibe toast de erro sem quebrar o polling
  - [x] Fluxo reassign: POST retorna nova decisão e a linha atualiza o backend exibido
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Administrador enxerga em tempo real qual GPU/modelo atende cada sessão (métrica de sucesso do PRD)
- Zero requisições `/proxy/*` de polling com o modo desligado
