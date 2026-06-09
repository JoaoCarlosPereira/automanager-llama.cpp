---

status: completed

title: Extrair módulo `auth.js` e testes Jest

type: frontend

complexity: medium

dependencies: [task_02]

---



# Tarefa 3: Extrair módulo `auth.js` e testes Jest



## Subtarefas



- [x] 3.1 Extrair funções de autenticação para `auth.js`

- [x] 3.2 Adicionar `showAlert()` e `showConfirm()`

- [x] 3.3 Substituir validações de `changePassword` por `showAlert()`

- [x] 3.4 Manter `window.initDashboard` e `window.startDashboardPolling`

- [x] 3.5 Criar `auth.test.js`

- [x] 3.6 Cobertura Jest ≥ 90% para `auth.js`



## Verificação



- `npm test -- --testPathPattern=auth.test.js --coverage --collectCoverageFrom=static/js/auth.js` — 22 testes, 100% statements/lines, 90.62% branches

- `pytest tests/` — 93 passed

