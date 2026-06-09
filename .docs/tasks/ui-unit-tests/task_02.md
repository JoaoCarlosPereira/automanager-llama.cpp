---
status: pending
title: Refatorar `_build_html()` para scripts externos
type: refactor
complexity: high
dependencies: [task_01]
---

# Tarefa 2: Refatorar `_build_html()` para scripts externos

## Visão Geral

Extrair o JavaScript embutido na f-string de `llama_manager.py` (linhas 969-2074, ~1100 linhas) para módulos ES6 separados em `static/js/` e substituir o `<script>` embutido por referências `<script type="module">` externas. Esta é a tarefa de maior risco pois modifica o comportamento runtime da aplicação — a ordem de scripts deve ser preservada e a dashboard deve continuar funcional após a refatoração.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O `<script>` embutido de 1100 linhas DEVE ser removido de `_build_html()` (linhas 969-2074 de `llama_manager.py`)
- 5 referências `<script type="module" src="/static/js/{auth,models,metrics,gpu,index}.js">` DEVEM substituir o script embutido
- O IP fixo (`fixedIp`), os valores de contexto (`CONTEXT_PRESET_VALUES`, `DEFAULT_CONTEXT_SIZE`, `CONTEXT_K_MULTIPLIER`), e as variáveis Python injetadas NO HTML DEVEM permanecer injetadas via `window.*` no `index.js`
- A ordem de carregamento DEVE ser: auth.js → models.js → metrics.js → gpu.js → index.js
- A dashboard DEVE continuar funcional após a refatoração (carregar, exibir componentes, manter comportamento original)
- Nenhum comportamento runtime DEVE ser alterado — a refatoração é puramente de organização, não de lógica

</requirements>

## Subtarefas

- [ ] 2.1 Copiar o JavaScript completo de `_build_html()` (linhas 969-2074) para um arquivo temporário de referência
- [ ] 2.2 Identificar as variáveis Python injetadas no JS (fixedIp, CONTEXT_PRESET_VALUES, DEFAULT_CONTEXT_SIZE, CONTEXT_K_MULTIPLIER, DEFAULT_PARALLEL_SLOTS, DEFAULT_BATCH_SIZE)
- [ ] 2.3 Criar `static/js/` diretório
- [ ] 2.4 Substituir o `<script>` embutido por 5 `<script type="module" src="/static/js/*.js">` no `llama_manager.py`
- [ ] 2.5 Adicionar `window.fixedIp`, `window.CONTEXT_PRESET_VALUES`, etc. no `index.js` para manter compatibilidade
- [ ] 2.6 Validar que `GET /` retorna HTML com 5 scripts externos ao invés de 1 embutido
- [ ] 2.7 Validar que a dashboard carrega e funciona após a refatoração (testar manualmente ou via task_10)

## Detalhes de Implementação

A f-string em `_build_html()` injeta variáveis Python dentro do JavaScript:
- `{local_ip}` → `fixedIp`
- `{json.dumps(CONTEXT_PRESET_VALUES)}` → `CONTEXT_PRESET_VALUES`
- `{DEFAULT_CONTEXT_SIZE}` → `DEFAULT_CONTEXT_SIZE`
- `{CONTEXT_K_MULTIPLIER}` → `CONTEXT_K_MULTIPLIER`
- `{DEFAULT_PARALLEL_SLOTS}` → `DEFAULT_PARALLEL_SLOTS`
- `{DEFAULT_BATCH_SIZE}` → `DEFAULT_BATCH_SIZE`

Estas variáveis devem continuar sendo injetadas no HTML mas lidas via `window.*` no `index.js`.

Referencie a seção "Extração de JavaScript para Módulos" do TechSpec para a estrutura dos módulos.

### Arquivos Relevantes

- `llama_manager.py` (linhas 969-2074) — modificado para remover script embutido e adicionar imports externos
- `static/js/index.js` — novo, injeta variáveis window e importa outros módulos
- `static/js/auth.js` — novo (task_03)
- `static/js/models.js` — novo (task_05)
- `static/js/metrics.js` — novo (task_04)
- `static/js/gpu.js` — novo (task_06)

### Arquivos Dependentes

- `static/` — diretório existente, subpasta `js/` será criada
- `tests/unit/test_html_contract.py` — task_10 validará que HTML contém 5 scripts externos

### ADRs Relacionados

- [ADR-001: Extração de JavaScript para testes unitários com Jest + Playwright E2E](../adrs/adr-001.md) — Decide extração do JS
- [ADR-002: Estrutura modular dos arquivos JavaScript](../adrs/adr-002.md) — Define a estrutura de 5 módulos
- [ADR-004: Configuração Jest e Playwright para testes de UI](../adrs/adr-004.md) — Define carregamento como módulos ES6

## Entregáveis

- `llama_manager.py` modificado: script embutido removido, 5 scripts externos adicionados
- `static/js/index.js` injetando variáveis window (fixedIp, CONTEXT_PRESET_VALUES, etc.)
- `GET /` retorna HTML com `src="/static/js/auth.js"`, `src="/static/js/models.js"`, `src="/static/js/metrics.js"`, `src="/static/js/gpu.js"`, `src="/static/js/index.js"`
- `GET /` NÃO contém mais a seção `<script>...</script>` com todo o JS embutido
- Dashboard continua funcional (login, dashboard visível, componentes renderizados)

## Testes

- Testes de refatoração:
  - [ ] `GET /` retorna 200 com HTML contendo 5 `<script type="module" src="/static/js/*.js">`
  - [ ] `GET /` retorna HTML NÃO contendo o `<script>` embutido original
  - [ ] `GET /static/js/index.js` retorna 200 (arquivo servido corretamente pelo StaticFiles)
  - [ ] Dashboard carrega sem erros de JS no console (index.js importa os outros 4 módulos)
  - [ ] Variáveis injetadas (fixedIp, context presets) estão acessíveis via `window.*` no navegador

## Critérios de Sucesso

- Dashboard funcional após refatoração (login, dashboard, componentes visíveis)
- 5 scripts externos referenciados no HTML
- Script embutido completamente removido de `_build_html()`
- Nenhum comportamento runtime alterado
- Todos os scripts carregam sem erro 404
