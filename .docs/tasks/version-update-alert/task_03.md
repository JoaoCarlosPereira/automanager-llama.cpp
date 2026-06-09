---
status: completed
title: Endpoint GET /api/system/version-check
type: backend
complexity: medium
dependencies:
  - task_01
  - task_02
---

# Endpoint GET /api/system/version-check

## Visão Geral

Expõe a verificação de versão via endpoint autenticado em `llama_manager.py`, delegando a `version_manager.check_for_updates()` e retornando `VersionCheckResponse`. Inclui testes de integração no padrão de `test_api_endpoints.py`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE adicionar `GET /api/system/version-check` na seção System Management de `llama_manager.py`
- DEVE exigir autenticação via `Depends(get_current_auth)` (sessão ou API key)
- DEVE chamar `check_for_updates(paths.INSTALL_ROOT)` e mapear resultado para `VersionCheckResponse`
- DEVE retornar HTTP 200 com JSON em todos os status (`ok`, `unavailable`, `error`) — erros git não viram 500 no MVP
- DEVE retornar HTTP 401 quando não autenticado
- NÃO DEVE alterar `POST /api/system/update` existente
- DEVE documentar o endpoint na tabela de API do README se o projeto mantiver essa convenção
</requirements>

## Subtarefas

- [x] 3.1 Importar `check_for_updates` e `VersionCheckResponse` em `llama_manager.py`
- [x] 3.2 Implementar handler async que converte `VersionCheckResult` → `VersionCheckResponse`
- [x] 3.3 Adicionar testes em `tests/integration/test_api_endpoints.py` com `FakeAuthManager`
- [x] 3.4 Adicionar mock de `version_manager.check_for_updates` nos testes de integração
- [x] 3.5 Verificar que endpoint não bloqueia outras rotas (sem lock global)

## Detalhes de Implementação

Posicionar a rota junto a `/api/system/shutdown` e `/api/system/update` (linhas ~414–431). Ver seção **Endpoints de API** do TechSpec. Padrão de auth: `get_current_auth` usado em `/status` e demais rotas protegidas.

### Arquivos Relevantes

- `llama_manager.py` — nova rota FastAPI
- `schemas.py` — `VersionCheckResponse` (tarefa 01)
- `version_manager.py` — `check_for_updates()` (tarefa 02)
- `paths.py` — `INSTALL_ROOT`
- `tests/integration/test_api_endpoints.py` — padrão `FakeAuthManager` + `authenticated_client`

### Arquivos Dependentes

- `static/js/version.js` — consumirá este endpoint (tarefa 05)
- `README.md` / `CLAUDE.md` — tabela de endpoints (opcional, se mantida atualizada)

### ADRs Relacionados

- [ADR-002: Módulo version_manager com subprocess git](adrs/adr-002.md) — backend subprocess
- [ADR-001: Modal Automático na Abertura do Dashboard](adrs/adr-001.md) — verificação apenas para usuários autenticados

## Entregáveis

- Rota `GET /api/system/version-check` funcional em `llama_manager.py`
- Testes de integração em `test_api_endpoints.py` com cobertura do endpoint **(OBRIGATÓRIO)**

## Testes

- Testes de integração:
  - [ ] `GET /api/system/version-check` sem cookie/token → HTTP 401
  - [ ] Com `authenticated_client` e mock retornando `update_available=True` → HTTP 200, JSON com `commits` array
  - [ ] Com mock retornando `status=unavailable` → HTTP 200, `update_available=False`
  - [ ] Com mock retornando `status=error` → HTTP 200, `error_message` presente no payload
  - [ ] Resposta JSON contém campos `current_ref`, `remote_ref`, `branch` quando `status=ok`
- Meta de cobertura: >= 80% no handler da rota
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% no código da rota
- Endpoint acessível apenas com sessão ou API key válida
- Payload JSON validado contra `VersionCheckResponse`
