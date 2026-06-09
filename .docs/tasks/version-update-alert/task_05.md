---
status: completed
title: Módulo version.js, wire-up e testes Jest
type: frontend
complexity: medium
dependencies:
  - task_03
  - task_04
---

# Módulo version.js, wire-up e testes Jest

## Visão Geral

Implementa a lógica frontend de verificação de versão em `static/js/version.js`: chama o endpoint uma vez por page load, abre o modal automaticamente quando há update disponível, e persiste dismiss em `sessionStorage`. Integra com `initDashboard()` e registra exports em `index.js`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE criar `static/js/version.js` com `checkForUpdates()`, `showVersionModal()` e `dismissVersionModal()`
- DEVE chamar `apiFetch('/api/system/version-check')` apenas uma vez por carregamento (flag interna `checked`)
- DEVE verificar `sessionStorage.getItem('version-update-dismissed')` antes de abrir o modal
- DEVE abrir modal automaticamente quando `update_available === true` e não dispensado
- DEVE gravar `sessionStorage.setItem('version-update-dismissed', '1')` ao dispensar
- DEVE suportar fechamento por botão, tecla Esc e clique no backdrop
- DEVE popular lista de commits com mensagem, autor e data formatada
- DEVE invocar `checkForUpdates()` ao final de `initDashboard()` em `models.js`
- DEVE registrar funções em `index.js` e incluir `version.js` em `_dashboard_js_version()`
- NÃO DEVE exibir modal quando `status` for `unavailable` ou `error`
- NÃO DEVE acionar `POST /api/system/update` — apenas informar (coexistência com botão ATUALIZAR)
</requirements>

## Subtarefas

- [x] 5.1 Criar `version.js` com lógica de fetch, render e dismiss
- [x] 5.2 Adicionar chamada `checkForUpdates()` no final de `initDashboard()` em `models.js`
- [x] 5.3 Importar e expor funções em `index.js` (padrão `window.*`)
- [x] 5.4 Incluir `version.js` na tupla de `_dashboard_js_version()` em `llama_manager.py`
- [x] 5.5 Criar `static/js/version.test.js` com mocks de `fetch`, DOM e `sessionStorage`
- [x] 5.6 Implementar trap de foco básico e listener de Esc no modal

## Detalhes de Implementação

Reutilizar `apiFetch` de `auth.js` para tratamento de 401. Padrão de testes: `auth.test.js`. Ver seções **Fluxo de dados**, **Frontend version.js** e **Abordagem de Testes** do TechSpec.

### Arquivos Relevantes

- `static/js/version.js` — novo módulo (criar)
- `static/js/version.test.js` — testes Jest (criar)
- `static/js/auth.js` — `apiFetch`, fluxo pós-login chama `initDashboard()`
- `static/js/models.js` — `initDashboard()` ponto de integração
- `static/js/index.js` — re-exports no `window`
- `llama_manager.py` — `_dashboard_js_version()` lista de arquivos JS

### Arquivos Dependentes

- `llama_manager.py` — markup do modal (tarefa 04)
- Endpoint `GET /api/system/version-check` (tarefa 03)

### ADRs Relacionados

- [ADR-003: Frontend version.js com modal e sessionStorage](adrs/adr-003.md) — decisão principal desta tarefa
- [ADR-001: Modal Automático na Abertura do Dashboard](adrs/adr-001.md) — modal automático e dismiss por sessão

## Entregáveis

- Módulo `static/js/version.js` funcional
- Wire-up em `models.js`, `index.js` e `_dashboard_js_version()`
- Arquivo `static/js/version.test.js` com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários (Jest):
  - [ ] `checkForUpdates()` chama `fetch`/`apiFetch` exatamente uma vez mesmo com dupla invocação
  - [ ] Resposta `update_available=true` exibe modal (`#version-update-modal` visível)
  - [ ] Resposta `update_available=false` não exibe modal
  - [ ] Resposta `status=error` não exibe modal
  - [ ] Resposta `status=unavailable` não exibe modal
  - [ ] `dismissVersionModal()` define `sessionStorage['version-update-dismissed']='1'` e oculta modal
  - [ ] Com dismiss prévio em sessionStorage, `checkForUpdates()` não reabre modal
  - [ ] Lista de commits renderiza N itens com message, author e date
  - [ ] Tecla Esc fecha o modal
  - [ ] Clique no backdrop fecha o modal
- Meta de cobertura: >= 80%
- Todos os testes devem passar (`npm test -- version.test.js`)

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% em `version.js`
- Fluxo completo: login → initDashboard → modal automático quando backend reporta update
- Dispensar persiste na aba até fechar; nova aba sem dismiss mostra modal novamente
- Dashboard continua operacional (polling de métricas/status) durante verificação async
