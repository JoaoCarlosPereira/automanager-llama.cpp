---
status: completed
title: Integração do roteamento no catch-all /v1
type: backend
complexity: high
dependencies:
  - task_03
---

# Integração do roteamento no catch-all /v1

## Visão Geral
Acopla o `ProxyRouter` ao handler `openai_proxy` existente: com o modo ativo e o modelo solicitado igual ao principal, a requisição é roteada pela decisão do router (com reescrita do `model` no body encaminhado); em qualquer outro caso o fluxo atual permanece intacto. Também filtra o `GET /v1/models` para expor somente o principal. É o ponto de maior risco de regressão da feature.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. Com `smart_proxy.enabled=false`, o comportamento do `/v1` DEVE permanecer byte-a-byte idêntico ao atual — a suíte `tests/unit/test_multi_model_proxy.py` DEVE passar sem alterações.
- 2. Com o modo ativo, requisição cujo `model` == modelo principal DEVE passar por `ProxyRouter.resolve()` e ser encaminhada à porta decidida; o body encaminhado DEVE ter `model` substituído pelo nome interno do backend, preservando TODOS os demais campos (conhecidos e desconhecidos) — PRD F8.
- 3. Requisição pedindo modelo secundário pelo nome real DEVE seguir o fluxo atual (sem sticky, sem reescrita) — ADR-003.
- 4. `GET /v1/models` com o modo ativo DEVE retornar somente o modelo principal no formato OpenAI (`{"object":"list","data":[...]}`); com modo inativo, agregação atual inalterada.
- 5. `acquire()` DEVE ocorrer antes do encaminhamento e `release()` em `finally` — inclusive quando o cliente aborta um stream no meio (o gerador de streaming DEVE decrementar no encerramento).
- 6. Erros do router (`ProxyError`) DEVEM virar respostas HTTP com corpo de erro formato OpenAI (503/404), e `httpx.RequestError` no backend roteado DEVE acionar o fluxo de reassign único (PRD F7) antes de responder 502.
- 7. Com modo ativo e sem principal definido/online, requisições ao principal DEVEM receber erro claro (PRD F2) e o `/v1/models` DEVE retornar lista vazia com log de aviso.
- 8. A resposta não-streaming roteada a secundário DEVE ter o campo `model` reescrito para o nome do principal (a parte SSE fica na tarefa 05, mas o caminho não-stream DEVE ser entregue aqui).
</requirements>

## Subtarefas
- [x] 4.1 Instanciar `ProxyRouter` em `llama_manager.py` (injeção de `process_manager.get_status`, `config_manager`, path de sessões) e carregar sessões no startup.
- [x] 4.2 Implementar o desvio no topo do `openai_proxy` (modo ativo + modelo principal) mantendo o caminho legado intocado.
- [x] 4.3 Encaminhar ao backend decidido com body reescrito, `acquire`/`release` em `finally` e reescrita do `model` na resposta não-streaming.
- [x] 4.4 Implementar reassign único em `httpx.RequestError` e o mapeamento de `ProxyError` para respostas formato OpenAI.
- [x] 4.5 Filtrar `_aggregate_models_response` para retornar somente o principal com o modo ativo.
- [x] 4.6 Escrever testes de rota cobrindo desvio, transparência, passthrough legado e erros.

## Detalhes de Implementação
Ver "Fluxo de dados" e "Endpoints de API" do TechSpec. Handler atual: `openai_proxy` em `llama_manager.py:616-698` (lê `body` bruto, `data.get("model")`, monta `target_url`, streaming via `client.stream` + `aiter_bytes`, erros em `:694-698`). Agregador: `_aggregate_models_response` em `llama_manager.py:585-613`. O encaminhamento httpx permanece no `llama_manager.py` parametrizado pela `RouteDecision` (ADR-004). Nesta tarefa o streaming roteado a secundário pode temporariamente repassar chunks brutos (reescrita SSE completa é a tarefa 05), mas o caminho não-stream já entrega transparência total.

### Arquivos Relevantes
- `llama_manager.py` — `openai_proxy` (`:616-698`), `_aggregate_models_response` (`:585-613`), instanciação de managers (~`:130-165`)
- `proxy_router.py` — API `resolve`/`acquire`/`release`/`ProxyError` (tarefas 02/03)

### Arquivos Dependentes
- `tests/unit/test_multi_model_proxy.py` — rede de segurança do comportamento legado (não pode regredir)
- `tests/unit/test_smart_proxy_routes.py` — novo arquivo de testes de rota desta tarefa
- `static/js/*` — nenhum impacto direto nesta tarefa

### ADRs Relacionados
- [ADR-003: Proxy no /v1 existente](../adrs/adr-003.md) — porta 8000, /v1/models só principal, secundário por nome real
- [ADR-004: Desvio no openai_proxy](../adrs/adr-004.md) — integração sem handler paralelo
- [ADR-001: Least-busy sticky](../adrs/adr-001.md) — semântica da decisão consumida

## Entregáveis
- Desvio de roteamento funcional no `/v1/chat/completions` (e demais paths POST do `/v1`)
- `/v1/models` filtrado pelo modo
- Tratamento de erros e reassign único integrados
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração do fluxo requisição→backend→resposta com mocks httpx **(OBRIGATÓRIO)**

## Testes
- Testes de rota (padrão `test_multi_model_proxy.py`: `TestClient(app)`, `app.dependency_overrides` para auth, `@patch("llama_manager.client...")`, mock de `process_manager.get_status`):
  - [x] Modo OFF: todos os testes existentes de `test_multi_model_proxy.py` passam sem modificação
  - [x] Modo ON + `model=principal` + tag `[AGENT:x]`: requisição é encaminhada à porta do backend decidido com body contendo o `model` interno; campos extras desconhecidos (ex.: `min_p`, `extra_body`) preservados
  - [x] Modo ON + `model=secundário (nome real)`: encaminhada direto à instância do secundário sem criar sessão sticky
  - [x] Modo ON: `GET /v1/models` retorna exatamente 1 item com `id` do principal; modo OFF retorna agregação completa
  - [x] Resposta não-stream de secundário chega ao cliente com `"model": <principal>`
  - [x] `httpx.RequestError` no backend decidido → reassign 1x para outro elegível e resposta 200; segunda falha consecutiva → 502 formato OpenAI
  - [x] Modo ON sem principal online → POST retorna erro claro formato OpenAI; contador in-flight volta a zero após cada requisição (sucesso e erro)
- Testes de integração:
  - [x] Sequência de 3 requisições da mesma tag → mesma porta nas 3 (sticky através do handler completo)
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando (incluindo suíte legada intacta)
- Cobertura de testes >= 80%
- Nenhuma mudança observável com o modo desligado
- Cliente externo nunca vê nome de modelo interno em respostas não-streaming
