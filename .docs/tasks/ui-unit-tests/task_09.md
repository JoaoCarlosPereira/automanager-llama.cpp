---

status: completed

title: Criar testes E2E de login com Playwright

type: test

complexity: medium

dependencies: [task_08]

---



# Tarefa 9: Criar testes E2E de login com Playwright



## Visão Geral



Criar `tests/e2e/login.spec.ts` — os primeiros testes E2E que validam o fluxo de login (sucesso e falha) com Playwright conectando ao mock server. Este é o fluxo P0 mais crítico do PRD.



<critical>

- SEMPRE LEIA o PRD e o TechSpec antes de começar

- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui

- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como

- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas

- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis

</critical>



<requirements>

- O arquivo `tests/e2e/login.spec.ts` DEVE existir com testes Playwright

- O teste de login com sucesso DEVE verificar: overlay visível → preencher credenciais → submit → dashboard visível → overlay oculto → componentes da dashboard presentes

- O teste de login com falha DEVE verificar: overlay visível → preencher credenciais inválidas → submit → mensagem de erro visível

- Os testes DEVEM usar condicion-based waits (`waitForSelector`) ao invés de sleeps fixos

- Os testes DEVEM conectar ao mock server via `baseURL` do playwright.config.ts



</requirements>



## Subtarefas



- [x] 9.1 Criar `tests/e2e/login.spec.ts`

- [x] 9.2 Implementar teste "login com credenciais válidas mostra dashboard"

- [x] 9.3 Implementar teste "login com credenciais inválidas mostra mensagem de erro"

- [x] 9.4 Implementar teste "overlay de login é visível ao acessar a página"

- [x] 9.5 Validar que os testes passam com mock server rodando



## Detalhes de Implementação



Referencie a seção "Testes E2E Playwright — Exemplo de Estrutura" do TechSpec.



Fluxo de login sucesso:

1. `page.goto('/')` — acessa dashboard

2. `expect(page.locator('#login-overlay')).toBeVisible()` — overlay visível

3. `page.locator('#login-username').fill('admin')` — preenche username

4. `page.locator('#login-password').fill('qualquer-senha')` — preenhe senha

5. `page.locator('#login-form').submit()` — submete

6. `expect(page.locator('#dashboard')).toBeVisible()` — dashboard visível

7. `expect(page.locator('#status-badge')).toBeVisible()` — status badge presente

8. `expect(page.locator('#cpu-val')).toBeVisible()` — métricas presentes

9. `expect(page.locator('#model-list-container')).toBeVisible()` — modelo presentes



Fluxo de login falha:

1. Mock server configurado para retornar 401 para credenciais inválidas

2. Preencher credenciais inválidas

3. Verificar que `#login-error` exibe mensagem



### Arquivos Relevantes



- `tests/e2e/login.spec.ts` — novo



### Arquivos Dependentes



- `tests/e2e/mock_server.py` — mock server com endpoint /api/auth/login

- `playwright.config.ts` — baseURL para mock server



### ADRs Relacionados



- [ADR-003: Estratégia de mock para testes E2E com Playwright](../adrs/adr-003.md) — Mock total via dependency_overrides



## Entregáveis



- `tests/e2e/login.spec.ts` com 3+ testes E2E

- Todos os testes passam com mock server

- Condicion-based waits usados (sem sleep)

- `npx playwright test login.spec.ts` passa



## Testes



- Testes E2E Playwright:

  - [x] Login sucesso: overlay → preenher → dashboard visível com todos os componentes

  - [x] Login falha: credenciais inválidas → mensagem de erro

  - [x] Overlay visível ao acessar página (antes de login)

  - [x] Após logout (handleLogout), overlay aparece novamente

- Meta: todos os testes devem passar com mock server

- Zero sleeps fixos (usar waitForSelector, waitForResponse)



### Execução verificada



```bash

npx playwright test tests/e2e/login.spec.ts --reporter=list --workers=1

```



4 testes passaram em ~19s (ambiente Windows, Chromium).



**Notas:**



- Testes em `mode: 'serial'` reutilizam o mesmo `BrowserContext`; `beforeEach` chama `context.clearCookies()` e `POST /__e2e/reset`.

- Mock rejeita login com `password === "wrong"` ou `username === "invalid"` (401 + `Credenciais invalidas`).

- Com `reuseExistingServer: true` (dev local), reinicie o processo na porta 8001 após alterar `mock_server.py`, ou use `CI=true` para subir servidor novo.



## Critérios de Sucesso



- login.spec.ts passa todos os testes

- Login sucesso: dashboard carrega com todos os componentes visíveis

- Login falha: mensagem de erro aparece no overlay

- Login falha: dashboard NÃO aparece

- Sem sleeps fixos nos testes

- `npx playwright test login.spec.ts --reporter=list` passa em < 30s

