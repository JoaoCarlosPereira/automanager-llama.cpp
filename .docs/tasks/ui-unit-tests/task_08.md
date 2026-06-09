---
status: completed
title: Criar servidor mock FastAPI para E2E
type: test
complexity: high
dependencies: [task_02]
---

# Tarefa 8: Criar servidor mock FastAPI para E2E

## Visão Geral

Criar um servidor FastAPI mock em `tests/e2e/mock_server.py` que substitui todas as respostas da API por dados determinísticos controlados pelos testes. Este mock permite que o Playwright interaja com a dashboard real (JS real, DOM real) sem depender de GPU, llama-server ou qualquer estado externo.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O arquivo `tests/e2e/mock_server.py` DEVE existir e exportar uma aplicação FastAPI com mocks para: `GET /status`, `POST /stop`, `POST /start`, `GET /metrics`, `GET /models`, `GET /logs`, `POST /api/auth/login`, `POST /api/auth/logout`, `GET /downloads`, `POST /downloads`, `POST /rename`, `POST /delete`, `POST /set_default`, `GET /api/key`, `POST /api/key/renew`, `POST /api/auth/change-password`
- O mock de `GET /status` DEVE retornar `{ running: boolean, model: string|null, config: object, recovery: object|null }`
- O mock de `GET /metrics` DEVE retornar `{ cpu: number, ram: number, gpus: [{ index, util, temp, power, mem_used, mem_total, vram_pct }] }`
- O mock de `GET /models` DEVE retornar `{ models: [{ id, name, path, dir }], projectors: [] }` com dados fake
- O mock de `POST /api/auth/login` DEVE criar cookie de sessão válido para qualquer credencial
- O mock de `GET /logs` DEVE retornar SSE stream com linhas controladas
- O mock de `GET /status` DEVE permitir controle de estado (running=true/false, model set) via API ou fixture
- O mock DEVE aplicar `dependency_overrides` no app principal para autenticação

</requirements>

## Subtarefas

- [x] 8.1 Criar `tests/e2e/` diretório com `__init__.py`
- [x] 8.2 Criar `mock_server.py` com FastAPI app contendo todos os endpoints mock
- [x] 8.3 Implementar `_mock_state` para controlar estado do mock (running, model, start_time)
- [x] 8.4 Implementar `_FAKE_MODELS` com 2-3 modelos fake
- [x] 8.5 Implementar `POST /api/auth/login` com cookie de sessão
- [x] 8.6 Implementar `GET /status`, `POST /stop`, `POST /start` com estado mock
- [x] 8.7 Implementar `GET /metrics` com dados fake de GPU
- [x] 8.8 Implementar `GET /models` com dados fake
- [x] 8.9 Implementar `GET /logs` com SSE stream fake
- [x] 8.10 Implementar `GET /downloads`, `POST /downloads` com estado vazio
- [x] 8.11 Implementar `POST /rename`, `POST /delete` com resposta `{ ok: true }`
- [x] 8.12 Implementar `POST /set_default` com resposta `{ ok: true }`
- [x] 8.13 Aplicar `dependency_overrides` para autenticação
- [x] 8.14 Criar `tests/e2e/conftest.py` com fixture para iniciar mock server

## Detalhes de Implementação

Referencie a seção "Mock Server para E2E" do TechSpec.

O mock server deve ser iniciado como webServer pelo Playwright via `playwright.config.ts`:
```
command: "python -m tests.e2e.mock_server"
url: "http://127.0.0.1:8001/"
```

O servidor mock deve rodar na porta 8001 (diferente da porta 8000 do app real).

### Arquivos Relevantes

- `tests/e2e/mock_server.py` — novo
- `tests/e2e/conftest.py` — novo, fixture para webServer
- `playwright.config.ts` — modificado para apontar webServer para mock_server.py

### Arquivos Dependentes

- `llama_manager.py` — mock importa app, auth_manager para dependency_overrides
- `playwright.config.ts` — configura webServer para iniciar mock

### ADRs Relacionados

- [ADR-003: Estratégia de mock para testes E2E com Playwright](../adrs/adr-003.md) — Define mock total via dependency_overrides

## Entregáveis

- `tests/e2e/mock_server.py` com 14+ endpoints mock
- `tests/e2e/conftest.py` com fixture para mock server
- `playwright.config.ts` configurado com webServer mock
- Mock server inicia na porta 8001 e responde a todos os endpoints
- Estado do mock é controlável via API (para cenários de teste)

## Testes

- Testes do mock server:
  - [x] Mock server inicia e responde em http://127.0.0.1:8001/
  - [x] `POST /api/auth/login` com qualquer credencial retorna cookie de sessão
  - [x] `GET /status` retorna estado mock controlável
  - [x] `POST /start` atualiza estado para running=true
  - [x] `POST /stop` atualiza estado para running=false
  - [x] `GET /metrics` retorna dados fake de CPU/RAM/GPU
  - [x] `GET /models` retorna lista fake de modelos
  - [x] `GET /logs` retorna SSE stream com dados
  - [x] `POST /rename` retorna { ok: true }
  - [x] `POST /delete` retorna { ok: true }
  - [x] `POST /set_default` retorna { ok: true }
  - [x] Dependency overrides aplicados (autenticação passa sem credenciais)

Suite automatizada: `pytest tests/e2e/test_mock_server.py`

## Critérios de Sucesso

- Mock server roda e responde a todos os endpoints
- Playwright consegue conectar ao mock server via webServer config
- Estado do mock é controlável (start → running=true, stop → running=false)
- Login com qualquer credencial funciona via cookie mock
- SSE stream fake funciona corretamente
