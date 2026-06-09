---

status: completed

title: Criar testes E2E de start/stop e gerenciamento de modelos

type: test

complexity: high

dependencies: [task_09]

---



# Tarefa 11: Criar testes E2E de start/stop e gerenciamento de modelos



## Visão Geral



Criar `tests/e2e/models.spec.ts` e `tests/e2e/metrics.spec.ts` — testes E2E que validam os fluxos de start/stop de modelos, atualização de métricas, SSE log streaming, download, rename e delete de modelos. Estes completam a cobertura E2E do PRD.



<critical>

- SEMPRE LEIA o PRD e o TechSpec antes de começar

- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui

- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como

- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas

- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis

</critical>



<requirements>

- O arquivo `tests/e2e/models.spec.ts` DEVE conter testes para: start modelo, stop modelo, rename modelo, delete modelo

- O arquivo `tests/e2e/metrics.spec.ts` DEVE conter testes para: métricas atualizam, logs SSE funcionam, download progresso

- Os testes DEVEM usar condicion-based waits (`waitForSelector`, `waitForResponse`) ao invés de sleeps fixos

- Os testes DEVEM conectar ao mock server via `baseURL` do playwright.config.ts



</requirements>



## Subtarefas



- [x] 11.1 Criar `tests/e2e/models.spec.ts`

- [x] 11.2 Implementar teste "start modelo mostra ONLINE e active card"

- [x] 11.3 Implementar teste "stop modelo mostra OFFLINE e esconde active card"

- [x] 11.4 Implementar teste "rename modelo atualiza nome na lista"

- [x] 11.5 Implementar teste "delete modelo remove da lista"

- [x] 11.6 Criar `tests/e2e/metrics.spec.ts`

- [x] 11.7 Implementar teste "métricas aparecem com valores não-zero"

- [x] 11.8 Implementar teste "logs SSE exibe linhas no terminal"

- [x] 11.9 Implementar teste "download progresso aparece após iniciar download"

- [x] 11.10 Validar que todos os testes passam com mock server



## Detalhes de Implementação



Referencie a seção "Testes E2E Playwright — Exemplo de Estrutura" do TechSpec.



Fluxos models.spec.ts:



Start modelo:

1. Login sucesso

2. `page.locator('.model-item-container').first().click()` — seleciona primeiro modelo

3. `page.locator('button:has-text("CARREGAR")').click()` — clica load

4. `expect(page.locator('#status-badge')).toContainText('ONLINE')` — status ONLINE

5. `expect(page.locator('#active-card')).toBeVisible()` — card ativo visível



Stop modelo:

1. Mock server já em running=true

2. `page.locator('button:has-text("ENCERRAR")').click()` — clica encerrar

3. `page.locator('#login-error').click()` — confirma no confirm mock

4. `expect(page.locator('#status-badge')).toContainText('OFFLINE')` — status OFFLINE



Fluxos metrics.spec.ts:



Métricas atualizam:

1. Login sucesso

2. `expect(page.locator('#cpu-val')).toHaveText(/\d+%/)` — CPU tem valor

3. `expect(page.locator('#ram-val')).toHaveText(/\d+%/)` — RAM tem valor



Logs SSE:

1. Mock server retorna SSE stream fake

2. `expect(page.locator('#log-box')).toContainText('llama server')` — log presente



### Arquivos Relevantes



- `tests/e2e/models.spec.ts` — novo

- `tests/e2e/metrics.spec.ts` — novo

- `tests/e2e/helpers.ts` — login, reset mock, refresh status

- `tests/e2e/mock_server.py` — endpoints mock (incl. `POST /__e2e/reset`, login 401 para `invalid`)



### Arquivos Dependentes



- `tests/e2e/mock_server.py` — mock server com endpoints mock

- `tests/e2e/login.spec.ts` — login como pré-requisito



### ADRs Relacionados



- [ADR-003: Estratégia de mock para testes E2E com Playwright](../adrs/adr-003.md) — Mock total via dependency_overrides



## Entregáveis



- `tests/e2e/models.spec.ts` com 4+ testes E2E

- `tests/e2e/metrics.spec.ts` com 3+ testes E2E

- Todos os testes passam com mock server

- Condicion-based waits usados (sem sleep)

- `npx playwright test models.spec.ts metrics.spec.ts` passa



## Testes



- Testes E2E Playwright (models.spec.ts):

  - [x] Start modelo: seleciona → carrega → ONLINE → active card visível

  - [x] Stop modelo: encerra → OFFLINE → active card oculto

  - [x] Rename modelo: rename → nome atualizado na lista

  - [x] Delete modelo: delete → modelo removido da lista

- Testes E2E Playwright (metrics.spec.ts):

  - [x] Métricas atualizam: CPU e RAM aparecem com valores não-zero

  - [x] Logs SSE: terminal exibe linhas de log do mock

  - [x] Download progresso: barra de progresso aparece após iniciar download

- Meta: todos os testes devem passar com mock server

- Zero sleeps fixos (usar waitForSelector, waitForResponse)



## Critérios de Sucesso



- models.spec.ts e metrics.spec.ts passam todos os testes

- Start/stop: status badge atualiza corretamente (ONLINE/OFFLINE)

- Rename/delete: modelo é atualizado/removido da lista visualmente

- Métricas: valores não-zero aparecem

- Logs SSE: terminal exibe linhas

- `npx playwright test --reporter=list` passa em < 2 min

- Sem sleeps fixos nos testes

