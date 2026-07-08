---
status: completed
title: Endpoints administrativos /proxy/* e /models/proxy
type: backend
complexity: medium
dependencies:
  - task_01
  - task_03
---

# Endpoints administrativos /proxy/* e /models/proxy

## Visão Geral
Expõe a superfície administrativa do proxy: configuração global (`/proxy/config`), flags por modelo (`/models/proxy`), inspeção (`/proxy/status`, `/proxy/backends`, `/proxy/sessions`), intervenção (limpeza/reassign de sessões, enable/disable de backends) e o simulador de roteamento `/proxy/resolve`. Dá ao administrador controle e diagnóstico completos sem tocar em arquivos.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. Todas as rotas DEVEM seguir o padrão canônico do projeto: `Depends(require_auth)` + `raise HTTPException(401)` (ver `/set_default` em `llama_manager.py:479-484`) e schemas pydantic da tarefa 01.
- 2. A superfície DEVE implementar exatamente a tabela "Endpoints de API" do TechSpec (métodos, caminhos, formatos de request/response e códigos de status).
- 3. `POST /proxy/resolve` DEVE usar `ProxyRouter.resolve(dry_run=True)` e retornar o formato do PRD (`proxy_enabled`, `external_model`, `detected_tag`, `affinity_key`, `selected_backend`, `internal_model`, `reason`, `sticky_hit`) sem criar sessão nem chamar backend.
- 4. `GET /proxy/backends` DEVE derivar os estados `online|offline|busy|disabled|not_eligible` e incluir `gpu` (nome + índice a partir de `config.gpu_weights[]` da instância, priorizando `is_main`), `in_flight`, `max_parallel` e `ctx_per_slot`.
- 5. `POST /proxy/config` DEVE validar `primary_model_path` contra os modelos conhecidos e aplicar mudanças com efeito imediato (sem reiniciar instâncias) — PRD F1/F2.
- 6. `DELETE /proxy/sessions/{affinity_key}` e `POST /proxy/sessions/{affinity_key}/reassign` DEVEM retornar 404 para chave inexistente.
- 7. Definir um novo principal DEVE ser refletido nas decisões seguintes; sessões existentes NÃO são migradas automaticamente (permanecem sticky até TTL/limpeza).
</requirements>

## Subtarefas
- [x] 6.1 Implementar `POST /proxy/config` e `POST /models/proxy` gravando via `ConfigManager` (tarefa 01).
- [x] 6.2 Implementar `GET /proxy/status`, `GET /proxy/backends` e `GET /proxy/sessions` compondo router + `get_status()` + nomes de GPU.
- [x] 6.3 Implementar `DELETE /proxy/sessions[/{key}]`, `POST /proxy/sessions/{key}/reassign` e `POST /proxy/backends/{port}/enable|disable`.
- [x] 6.4 Implementar `POST /proxy/resolve` com `dry_run=True`.
- [x] 6.5 Escrever testes de rota para toda a superfície, incluindo auth e códigos de erro.

## Detalhes de Implementação
Ver tabela "Endpoints de API" do TechSpec. Padrão de rota autenticada: `llama_manager.py:479-484`; nome de GPU por instância: `get_status().instances[].config.gpu_weights[].name`/`index` (sem novas chamadas a nvidia-smi). Rotas novas agrupadas em bloco próprio no `llama_manager.py` (ADR-004: lógica no router, rotas finas no manager).

### Arquivos Relevantes
- `llama_manager.py` — novas rotas `/proxy/*` e `/models/proxy`
- `proxy_router.py` — métodos `sessions`, `clear_sessions`, `reassign`, `set_backend_enabled`, `resolve(dry_run)` (tarefas 02/03)
- `schemas.py` — `ProxyConfigRequest`, `SetModelProxyRequest` (tarefa 01)

### Arquivos Dependentes
- `static/js/proxy.js` (tarefas 07/08) — consumirá `/proxy/status`, `/proxy/config`, `/models/proxy`
- `tests/unit/test_smart_proxy_routes.py` — casos desta tarefa

### ADRs Relacionados
- [ADR-004: Rotas finas no llama_manager](../adrs/adr-004.md) — wiring rota→router
- [ADR-005: Configuração smart_proxy + flags por modelo](../adrs/adr-005.md) — persistência consumida por `/proxy/config` e `/models/proxy`
- [ADR-003: Exposição do principal](../adrs/adr-003.md) — validação de principal definido

## Entregáveis
- 10 rotas administrativas funcionais e autenticadas
- `/proxy/resolve` operacional para depuração sem inferência
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração config→roteamento (mudança via API refletida na decisão) **(OBRIGATÓRIO)**

## Testes
- Testes de rota (TestClient + dependency_overrides):
  - [x] Todas as 10 rotas retornam 401 sem autenticação
  - [x] `POST /proxy/config {"enabled":true,"primary_model_path":"<path válido>"}` persiste e `GET /proxy/status` reflete `enabled=true` e o principal
  - [x] `POST /proxy/config` com `primary_model_path` desconhecido retorna 400/422 com mensagem clara
  - [x] `POST /models/proxy {"model_path":X,"proxy_eligible":false}` → `GET /proxy/backends` mostra estado `not_eligible` para as instâncias de X
  - [x] `GET /proxy/backends` com 3 instâncias mockadas retorna `gpu` com nome correto (ex.: "Tesla P100" índice 1) e `ctx_per_slot = context_size // parallel_slots`
  - [x] `POST /proxy/backends/{port}/disable` → novas resoluções não usam a porta; `/enable` restaura
  - [x] `DELETE /proxy/sessions/{key}` inexistente → 404; existente → 200 e some de `GET /proxy/sessions`
  - [x] `POST /proxy/sessions/{key}/reassign` muda o backend da sessão e retorna a `RouteDecision`
  - [x] `POST /proxy/resolve` com tag `[AGENT:sql-reviewer]` retorna os 8 campos do contrato do PRD e NÃO cria sessão (verificar `GET /proxy/sessions` vazio)
- Testes de integração:
  - [x] Fluxo: definir principal via API → resolve dry_run aponta para ele → trocar principal → nova resolução (nova chave) aponta para o novo, sessão antiga permanece no backend original
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Administrador consegue ligar o modo, definir principal e depurar roteamento apenas via API
- Nenhuma rota administrativa acessível sem autenticação
