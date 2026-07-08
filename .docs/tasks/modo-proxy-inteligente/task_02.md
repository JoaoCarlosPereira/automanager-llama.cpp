---
status: completed
title: "ProxyRouter: extração de afinidade e sessões sticky (TTL + persistência)"
type: backend
complexity: medium
dependencies:
  - task_01
---

# ProxyRouter: extração de afinidade e sessões sticky (TTL + persistência)

## Visão Geral
Cria o módulo `proxy_router.py` com a metade "estado" do roteador: dataclasses (`StickySession`, `RouteDecision`), extração da `affinity_key` em 5 camadas e a tabela de sessões sticky com TTL de inatividade e persistência JSON atômica em `data/proxy_sessions.json`. É a fundação que garante que a mesma conversa/subagente sempre resolva para o mesmo backend.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. A extração de `affinity_key` DEVE seguir a ordem exata do PRD F5/TechSpec: header `x-automanager-session-id` → header `x-automanager-agent-id` → `metadata.session_id`/`metadata.agent_id` → tag regex `\[AGENT:([A-Za-z0-9_-]+)\]` (primeira ocorrência, mensagens system primeiro) → hash estável `sha256(primeiro system + primeira user + modelo externo + IP + User-Agent)[:16]`, com os prefixos de chave definidos no TechSpec (`sid:`, `aid:`, `agent:<tag>:<hash8>`, `hash:<hash16>`).
- 2. A chave NÃO DEVE incorporar valores voláteis (timestamp, request id) — a mesma entrada DEVE produzir sempre a mesma chave.
- 3. `StickySession` DEVE conter os campos definidos em "Modelos de Dados" do TechSpec, com timestamps ISO-8601 UTC e `backend_model_path` como identificador durável.
- 4. Toda mutação da tabela DEVE ocorrer sob `asyncio.Lock`; expiração por TTL DEVE ser lazy (verificada no acesso) usando `ttl_minutes` do `smart_proxy`.
- 5. A persistência DEVE usar escrita atômica (`.tmp` + `os.replace`) em `data/proxy_sessions.json`; gravação em criação/remoção de sessão e oportunista (a cada 20 requisições ou shutdown) para contadores.
- 6. No carregamento, sessões com TTL vencido DEVEM ser descartadas e sessões cuja porta não existe DEVEM ser re-vinculadas por `backend_model_path` quando houver instância online compatível.
- 7. O relógio (`now()`) DEVE ser injetável para permitir testes de TTL sem sleep.
- 8. O módulo NÃO DEVE importar `llama_manager` (dependências entram pelo construtor — ADR-004).
</requirements>

## Subtarefas
- [x] 2.1 Criar `proxy_router.py` com as dataclasses `StickySession` e `RouteDecision` e a exceção `ProxyError` (status + payload formato OpenAI).
- [x] 2.2 Implementar o extrator de afinidade em 5 camadas com os prefixos de chave e detecção da tag `[AGENT:...]`.
- [x] 2.3 Implementar a tabela sticky com `asyncio.Lock`, TTL lazy e API de consulta/limpeza (`sessions()`, `clear_sessions()`).
- [x] 2.4 Implementar persistência atômica (save debounced/oportunista, load no boot com descarte por TTL e re-vínculo por `model_path`).
- [x] 2.5 Escrever testes unitários das 5 camadas, estabilidade da chave, TTL e round-trip de persistência.

## Detalhes de Implementação
Ver seções "Interfaces Principais" e "Modelos de Dados" do TechSpec (assinaturas de `ProxyRouter`, `StickySession`, `RouteDecision`, prefixos de chave). Construtor recebe `get_status` (callable), `config_manager` e `sessions_path` por injeção. Padrão de escrita atômica em `config_manager.py:169-179`.

### Arquivos Relevantes
- `proxy_router.py` — módulo novo (dataclasses, extração, tabela sticky, persistência)
- `config_manager.py` — fonte de `ttl_minutes` via `get_smart_proxy_settings()` (tarefa 01)
- `paths.py` — referência para resolver `data/proxy_sessions.json` sob o diretório de dados

### Arquivos Dependentes
- `llama_manager.py` — instanciará o `ProxyRouter` (tarefa 04)
- `tests/unit/test_proxy_router.py` — novo arquivo de testes desta e da tarefa 03

### ADRs Relacionados
- [ADR-004: Módulo único proxy_router.py](../adrs/adr-004.md) — limites do módulo e injeção de dependências
- [ADR-005: Estado sticky e persistência](../adrs/adr-005.md) — formato do JSON, re-vínculo por `model_path`, TTL
- [ADR-002: Tag como chave de afinidade](../adrs/adr-002.md) — a tag não escolhe backend, apenas identifica a sessão

## Entregáveis
- `proxy_router.py` com dataclasses, extração de afinidade e tabela sticky persistente
- Arquivo `data/proxy_sessions.json` criado/gerido automaticamente
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração de persistência (round-trip disco) **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [x] Header `x-automanager-session-id: abc` gera chave `sid:abc` e vence todas as outras camadas presentes juntas
  - [x] `metadata.agent_id` vence tag no conteúdo; tag vence hash de fallback
  - [x] Tag `[AGENT:sql-reviewer]` em mensagem system é detectada e gera chave prefixada `agent:sql-reviewer:`
  - [x] Duas requisições idênticas (mesmo system/user/modelo/IP/UA) sem tag geram a MESMA chave `hash:`; mudar apenas o User-Agent muda a chave
  - [x] Sessão não usada por mais que `ttl_minutes` é considerada expirada no acesso (com `now()` injetado, sem sleep)
  - [x] Regex não casa `[AGENT:]` vazio nem tags com caracteres inválidos (ex.: espaços)
- Testes de integração:
  - [x] Round-trip: criar 3 sessões → persistir → recarregar em novo router → sessões idênticas; sessão expirada não retorna; sessão com porta morta re-vincula à instância online do mesmo `model_path`
  - [x] Arquivo corrompido (JSON inválido) no boot → router inicia vazio com warning, sem crash
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Mesma conversa/subagente sempre produz a mesma `affinity_key` entre requisições e restarts
- `proxy_router.py` importável sem FastAPI/llama_manager
