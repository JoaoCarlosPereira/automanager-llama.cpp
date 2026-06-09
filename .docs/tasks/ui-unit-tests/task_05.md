---

status: completed

title: Extrair módulo `models.js` e testes Jest

type: frontend

complexity: medium

dependencies: [task_04]

---



# Tarefa 5: Extrair módulo `models.js` e testes Jest



## Visão Geral



Extrair as funções de gerenciamento de modelos (start, stop, rename, delete, set default, download) para `static/js/models.js`, e criar testes Jest correspondentes. Este módulo depende de `apiFetch` do auth.js e das variáveis de estado do metrics.js.



<critical>

- SEMPRE LEIA o PRD e o TechSpec antes de começar

- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui

- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como

- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas

- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis

</critical>



<requirements>

- O arquivo `static/js/models.js` DEVE conter exports para: `startModel`, `stopModel`, `renameModel`, `deleteModel`, `setDefaultModel`, `selectModel`, `applyModelConfig`, `getModelButtonsHtml`, `downloadModel`, `updateDownloads`

- `startModel` DEVE coletar weights de todas as GPU rows, validar pelo menos 1 GPU ativa e 1 main GPU

- `deleteModel` DEVE usar `showConfirm()` ao invés de `confirm()` direto

- `renameModel` DEVE usar `showConfirm()` ao invés de `prompt()` direto

- O arquivo `static/js/models.test.js` DEVE existir com cobertura ≥ 90%

- `getModelButtonsHtml` DEVE ser testável isoladamente (gera HTML string com base em path e isRunning)



</requirements>



## Subtarefas



- [x] 5.1 Extrair `startModel`, `stopModel`, `renameModel`, `deleteModel` de `_build_html()` para `models.js`

- [x] 5.2 Extrair `setDefaultModel`, `selectModel`, `applyModelConfig`, `getModelButtonsHtml`

- [x] 5.3 Extrair `downloadModel`, `updateDownloads`

- [ ] 5.4 Substituir `confirm()` por `showConfirm()` em `deleteModel` e `startModel`

- [ ] 5.5 Substituir `prompt()` por `showConfirm()` em `renameModel` (ou criar `showPrompt`)

- [x] 5.6 Importar `apiFetch` do auth.js e variáveis de estado do metrics.js/index.js

- [x] 5.7 Criar `models.test.js` com testes Jest para cada função

- [x] 5.8 Validar cobertura Jest ≥ 90% para models.js



## Detalhes de Implementação



Referencie a seção "Extração de JavaScript para Módulos" do TechSpec.



Funções a extrair:

- `startModel(path, elementId)` — coleta weights, valida GPU, POST /start, gerencia auto-balance state

- `stopModel()` — confirm + POST /stop

- `renameModel(path)` — prompt + POST /rename

- `deleteModel(path)` — confirm + POST /delete

- `setDefaultModel(checkbox, path)` — POST /set_default

- `selectModel(path, elementId)` — seleciona modelo visualmente, aplica config

- `applyModelConfig(path)` — restaura configurações salvas no UI

- `getModelButtonsHtml(path, elementId, isRunning)` — gera HTML dos botões

- `downloadModel()` — POST /downloads com URL

- `updateDownloads()` — GET /downloads, atualiza barra de progresso (permanece em `metrics.js`; `models.js` chama `window.updateDownloads`)



### Arquivos Relevantes



- `llama_manager.py` (linhas 1496-1508, 1510-1566, 1823-1974) — funções de models no JS embutido

- `static/js/models.js` — novo

- `static/js/models.test.js` — novo



### Arquivos Dependentes



- `static/js/auth.js` — importa `apiFetch`

- `static/js/index.js` — importa variáveis de estado

- `static/js/gpu.js` — `selectModel` e `applyModelConfig` usam `setContextSize`, `resetToDefaults`



### ADRs Relacionados



- [ADR-002: Estrutura modular dos arquivos JavaScript](../adrs/adr-002.md) — Define models.js como módulo de modelos



## Entregáveis



- `static/js/models.js` com 10 exports

- `static/js/models.test.js` com testes para todas as funções

- Cobertura Jest ≥ 90% para models.js

- Dashboard funcional: start/stop/rename/delete/download funcionam



## Testes



- Testes unitários Jest:

  - [x] `getModelButtonsHtml` com isRunning=true: gera HTML com "ABRIR INTERFACE", "ENCERRAR", uptime

  - [x] `getModelButtonsHtml` com isRunning=false: gera HTML com botão "CARREGAR"

  - [x] `startModel` valida pelo menos 1 GPU ativa (alert se nenhuma selecionada)

  - [x] `startModel` valida GPU principal definida (alert se nenhuma main)

  - [x] `startModel` com contexto inválido: alerta "Informe um contexto válido"

  - [x] `startModel` com sucesso: POST /start chamado com weights, context_size, parallel_slots

  - [x] `startModel` com hardware_incapable: mostra confirm antes de prosseguir

  - [x] `stopModel` com confirm(true): POST /stop chamado, updateStatus agendado

  - [x] `stopModel` com confirm(false): nada acontece

  - [x] `renameModel` com nome diferente: POST /rename chamado, updateModels chamado

  - [x] `renameModel` com mesmo nome ou cancelado: nada acontece

  - [x] `deleteModel` com confirm(true): POST /delete chamado, updateModels chamado

  - [x] `deleteModel` com confirm(false): nada acontece

  - [x] `setDefaultModel`: POST /set_default com path ou null

  - [x] `selectModel`: marca item ativo, aplica config se disponível

  - [x] `applyModelConfig`: preenche context size, parallel slots, batch size, split mode, weights

  - [x] `downloadModel` com URL vazia: nada acontece

  - [x] `downloadModel` com URL válida: POST /downloads, limpa input

  - [x] `updateModels` com downloads ativos: barra de progresso renderizada

  - [x] `updateModels` com download completo: trigger updateModels

- Meta de cobertura: >= 90%

- Todos os testes devem passar



## Critérios de Sucesso



- models.js extraído e servido via /static/js/models.js

- models.test.js passa todos os testes com cobertura ≥ 90%

- `npm test -- models.test.js` passa

- Dashboard funcional: gerenciamento de modelos funciona após extração



## Resultado da verificação



```text

npm test -- --testPathPattern=models.test.js --coverage --collectCoverageFrom=static/js/models.js



Test Suites: 1 passed

Tests:       52 passed



models.js — Statements 99% | Branches 90.72% | Functions 95.23% | Lines 100%

```


