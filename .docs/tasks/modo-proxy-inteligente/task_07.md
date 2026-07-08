---
status: completed
title: "UI: controles de configuração do proxy"
type: frontend
complexity: medium
dependencies:
  - task_06
---

# UI: controles de configuração do proxy

## Visão Geral
Adiciona ao dashboard os controles de escrita do Modo Proxy Inteligente: checkbox global "Ativar Modo Proxy Inteligente", marcação "Principal" por aba de modelo (exclusiva — apenas um por vez) e checkbox por aba "Usar como backend secundário no proxy", além do campo de limite de concorrência por modelo. Tudo persiste via os endpoints da tarefa 06.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. O checkbox global DEVE chamar `POST /proxy/config {enabled}` e refletir o estado atual carregado no boot da página (PRD F1).
- 2. A marcação "Principal" por aba DEVE ser mutuamente exclusiva: marcar um modelo desmarca visualmente o anterior e persiste via `POST /proxy/config {primary_model_path}` (PRD F2).
- 3. O checkbox "Usar como backend secundário no proxy" (default ligado) e o campo `max_parallel_requests` DEVEM persistir via `POST /models/proxy`, seguindo o padrão de `persistThinkingEnabled` (`static/js/models.js:1045-1058`).
- 4. Com o modo ligado e sem principal definido, a UI DEVE exibir orientação clara ao administrador (PRD, Riscos: adoção).
- 5. Os controles DEVEM usar `apiFetch` (auth) e `showToast` para feedback de sucesso/erro, no padrão de `auth.js`.
- 6. Textos DEVEM ser em pt-BR hardcoded, consistentes com o restante do dashboard (não há i18n).
- 7. O contrato de HTML testado em `tests/unit/test_html_contract.py` DEVE ser atualizado junto.
</requirements>

## Subtarefas
- [x] 7.1 Adicionar o bloco de configuração global (checkbox + orientações) ao HTML inline gerado em `_build_html`.
- [x] 7.2 Adicionar controles "Principal" e "Participar do proxy" + limite de concorrência ao template de aba de modelo.
- [x] 7.3 Criar `static/js/proxy.js` com carga inicial do estado (`GET /proxy/status` + `/config`) e os handlers de persistência.
- [x] 7.4 Implementar exclusividade visual do "Principal" e estados desabilitados coerentes (ex.: controles por aba inertes com modo OFF).
- [x] 7.5 Atualizar o contrato de HTML e escrever testes dos novos elementos.

## Detalhes de Implementação
Ver "Visão dos Componentes" (linha `static/js/proxy.js + _build_html`) e "Endpoints de API" do TechSpec. HTML do dashboard é gerado inline em `_build_html` (`llama_manager.py:940+`), com template de aba `<template id="model-tab-template">` (`llama_manager.py:1397-1744`); padrão de checkbox persistente: `setDefaultModel` (`static/js/models.js:1241-1251`) e `persistThinkingEnabled` (`models.js:1045-1058`); imports versionados `./auth.js?v=...` (`models.js:2`).

### Arquivos Relevantes
- `llama_manager.py` — `_build_html` e template de aba (HTML inline)
- `static/js/proxy.js` — módulo novo (estado + handlers)
- `static/js/models.js` — ponto de integração dos controles por aba
- `static/js/auth.js` — `apiFetch`/`showToast` reutilizados

### Arquivos Dependentes
- `tests/unit/test_html_contract.py` — contrato do HTML gerado (atualizar)
- `static/js/index.js` — registro/carga do novo módulo

### ADRs Relacionados
- [ADR-002: Sem regras tag→backend](../adrs/adr-002.md) — a UI de configuração NÃO inclui mapeamento de tags
- [ADR-005: Configuração](../adrs/adr-005.md) — chaves persistidas pelos controles

## Entregáveis
- Checkbox global, marcação "Principal" exclusiva e participação por instância funcionais no dashboard
- `static/js/proxy.js` novo integrado ao carregamento existente
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração do contrato HTML **(OBRIGATÓRIO)**

## Testes
- Testes unitários (pytest sobre o HTML gerado + rotas mockadas):
  - [x] HTML gerado contém o checkbox global com id/atributo esperado e o bloco "Proxy Inteligente"
  - [x] Template de aba contém os controles "Principal" e "Participar do proxy" com handlers referenciando `proxy.js`
  - [x] `test_html_contract.py` atualizado passa com os novos elementos
- Testes de integração (rotas + estado):
  - [x] Sequência via API simulando a UI: marcar modelo A como principal → marcar modelo B → `GET /proxy/status` mostra apenas B como principal
  - [x] Desmarcar "Participar do proxy" do modelo C → `GET /proxy/backends` reporta `not_eligible` para C
  - [x] Ligar modo sem principal definido → resposta da API usada pela UI carrega indicação para exibir a orientação
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Administrador configura o proxy inteiro pela UI sem editar arquivos
- Apenas um "Principal" marcado por vez em qualquer sequência de cliques
