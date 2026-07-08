# TechSpec — Modo Proxy Inteligente

## Resumo Executivo

O Modo Proxy Inteligente é implementado como um módulo novo `proxy_router.py` (classe `ProxyRouter`, com estado) acoplado por um desvio condicional ao catch-all `/v1` já existente em `llama_manager.py:616-698` (ADR-004). O router mantém a tabela de afinidade sticky em memória com persistência JSON atômica em `data/proxy_sessions.json`, contadores de requisições em andamento por porta, e decide o backend por least-busy interno (ADR-001, ADR-005). A configuração usa os padrões existentes: chave global `smart_proxy` no `automanager_config.json` e flags por modelo (`proxy_eligible`, `max_parallel_requests`) na whitelist de `update_model_settings`. A transparência para o cliente é garantida por reescrita do campo `model` linha a linha no streaming SSE, com buffer incremental que tolera eventos cortados entre chunks TCP (ADR-006).

**Trade-off principal**: a contagem de carga é interna ao manager (não consulta `/slots` do llama.cpp), o que torna a decisão de roteamento instantânea e sem dependências, ao custo de não enxergar tráfego enviado diretamente às portas dos backends — aceitável porque o `/v1` do manager é o único ponto de entrada externo pretendido.

## Arquitetura do Sistema

### Visão dos Componentes

| Componente | Tipo | Responsabilidade |
|---|---|---|
| `proxy_router.py` — `ProxyRouter` | novo | Extração de `affinity_key`, tabela sticky (TTL + persistência), contadores in-flight, elegibilidade, seleção least-busy, fallback/reassign, estado disable runtime |
| `llama_manager.py` — `openai_proxy` | modificado | Desvio no topo: proxy ativo + modelo pedido == principal → consulta `ProxyRouter`, encaminha ao backend escolhido com reescrita de `model`; senão fluxo atual intacto |
| `llama_manager.py` — `_aggregate_models_response` | modificado | Com modo ativo, retorna somente o modelo principal |
| `llama_manager.py` — rotas `/proxy/*` | novo | Endpoints administrativos (status, sessions, backends, resolve, config) |
| `config_manager.py` | modificado | `get/update_smart_proxy_settings()` (chave global) + 2 campos novos na whitelist por modelo |
| `schemas.py` | modificado | `DEFAULT_PROXY_ELIGIBLE`, `DEFAULT_MAX_PARALLEL_REQUESTS`, schemas pydantic das rotas novas |
| `static/js/proxy.js` + `_build_html` | novo/modificado | Card "Proxy Inteligente", checkbox global, controles por aba de modelo |

**Fluxo de dados (requisição ao modelo principal, modo ativo):**

```
Cliente → POST /v1/chat/completions {model: principal}
  → openai_proxy: proxy ativo? modelo == principal?
    → ProxyRouter.resolve(headers, body_json, client_ip, user_agent)
        1. extrai affinity_key (header → metadata → tag → hash estável)
        2. sticky hit? → mesmo backend (valida vivo/elegível; senão reassign 1x)
        3. novo → elegíveis (online, proxy_eligible, não disabled,
           ctx suficiente, abaixo de max_parallel) → main? principal : least-busy
        4. registra sessão, incrementa contador → RouteDecision
    → encaminha httpx à porta escolhida (body com model interno)
    → resposta/SSE reescrita para model = principal
       + headers x-automanager-backend[-model]
    → finally: decrementa contador, atualiza last_used_at/request_count
```

## Design de Implementação

### Interfaces Principais

```python
# proxy_router.py
class ProxyRouter:
    def __init__(self, get_status: Callable[[], dict],
                 config_manager: ConfigManager,
                 sessions_path: Path) -> None: ...

    async def resolve(self, *, headers: Mapping[str, str], body: dict,
                      client_ip: str, user_agent: str,
                      dry_run: bool = False) -> RouteDecision:
        """Decide o backend. Levanta ProxyError(status, payload_openai)
        quando não há backend disponível. dry_run não cria sessão
        nem incrementa contadores (usado por /proxy/resolve)."""

    async def acquire(self, port: int) -> None      # incrementa in-flight
    async def release(self, port: int, *, usage: dict | None) -> None
    async def sessions(self) -> list[StickySession]  # expira TTL lazy
    async def reassign(self, affinity_key: str) -> RouteDecision
    async def clear_sessions(self, affinity_key: str | None = None) -> int
    def set_backend_enabled(self, port: int, enabled: bool) -> None
```

```python
@dataclass
class RouteDecision:
    backend_port: int
    internal_model: str        # basename do gguf do backend
    external_model: str        # nome do principal (para reescrita)
    affinity_key: str
    detected_tag: str | None   # ex.: "sql-reviewer"
    sticky_hit: bool
    reason: str                # main_preference|least_busy|sticky|reassign_*
    rewrite: bool              # False quando backend == principal
```

**Extração de afinidade** (ordem do PRD F5): header `x-automanager-session-id` → `x-automanager-agent-id` → `metadata.session_id`/`metadata.agent_id` → tag por regex `\[AGENT:([A-Za-z0-9_-]+)\]` no `content` das mensagens (primeira ocorrência, system primeiro) → fallback `sha256(primeiro system + primeira user + modelo externo + IP + User-Agent)[:16]`. Prefixos de chave: `sid:`, `aid:`, `agent:<tag>:<hash8>`, `hash:<hash16>`.

**Seleção (nova sessão)**: `main` = tag ausente ou `[AGENT:main]`. Elegíveis = instâncias `running` com `proxy_eligible`, não desabilitadas, com `ctx_estimado ≤ config.context_size // max(1, parallel_slots)` e `in_flight < max_parallel_requests`. Main → principal se elegível (se ocupado, espera na fila curta); principal offline → least-busy secundário. Subagente → least-busy entre todos os elegíveis (empate: menor porta). Estimativa de tokens: `len(json.dumps(messages, ensure_ascii=False)) // 4`, com margem de 10%.

**Espera por backend ocupado** (PRD F7): loop `asyncio.sleep(0.25)` até liberar slot ou estourar `max_wait_seconds` (30 s); ao estourar: sessão sticky → HTTP 503 com corpo de erro formato OpenAI; sessão nova → tenta próximo elegível antes do erro.

### Modelos de Dados

```python
@dataclass
class StickySession:
    affinity_key: str
    backend_port: int
    backend_model_path: str    # identificador durável (ADR-005)
    external_model: str
    internal_model: str
    detected_tag: str | None
    created_at: str            # ISO-8601 UTC
    last_used_at: str
    request_count: int = 0
    tokens_processed: int = 0  # best-effort via usage do backend
```

**`data/proxy_sessions.json`** (escrita atômica `.tmp` + `os.replace`): `{"sessions": [StickySession...]}`. Persistido em criação/reassign/limpeza; `last_used_at`/contadores salvos de forma oportunista (a cada N=20 requisições ou no shutdown). No boot: sessões com TTL vencido descartadas; porta inexistente re-vinculada por `backend_model_path` quando possível.

**`automanager_config.json`** — nova chave global:

```json
"smart_proxy": {
  "enabled": false,
  "primary_model_path": null,
  "ttl_minutes": 180,
  "max_wait_seconds": 30
}
```

**Por modelo** (whitelist `update_model_settings` + `DEFAULT_*` em `schemas.py`): `proxy_eligible: true`, `max_parallel_requests: 1`.

### Endpoints de API

| Método | Caminho | Auth | Request | Response (200) |
|---|---|---|---|---|
| POST | `/proxy/config` | ✔ | `{enabled?, primary_model_path?, ttl_minutes?, max_wait_seconds?}` | `{message, smart_proxy}` |
| POST | `/models/proxy` | ✔ | `{model_path, proxy_eligible?, max_parallel_requests?}` | `{message}` |
| GET | `/proxy/status` | ✔ | — | `{enabled, primary:{model, port, gpu}, backends:[...], sessions_count}` |
| GET | `/proxy/backends` | ✔ | — | `[{port, model, model_path, gpu, role, state, in_flight, max_parallel, ctx_per_slot}]` |
| POST | `/proxy/backends/{port}/enable` / `disable` | ✔ | — | `{message, state}` |
| GET | `/proxy/sessions` | ✔ | — | `[StickySession + backend info]` |
| DELETE | `/proxy/sessions` | ✔ | — | `{removed: n}` |
| DELETE | `/proxy/sessions/{affinity_key}` | ✔ | — | `{removed: 1}` / 404 |
| POST | `/proxy/sessions/{affinity_key}/reassign` | ✔ | — | `RouteDecision` / 404 |
| POST | `/proxy/resolve` | ✔ | corpo estilo chat/completions | `{proxy_enabled, external_model, detected_tag, affinity_key, selected_backend, internal_model, reason, sticky_hit}` |

Estados de backend em `/proxy/backends`: `online`, `offline`, `busy` (in_flight ≥ max), `disabled`, `not_eligible`. `gpu` vem de `config.gpu_weights[]` (`name` + `index` da GPU com `is_main`/maior peso) — sem chamadas novas a nvidia-smi. Comportamento do `/v1` (modelo secundário pedido pelo nome real, `/v1/models` só principal, erro claro sem principal online) conforme ADR-003 e PRD F4. Schemas novos com `model_config = ConfigDict(protected_namespaces=())` (campos `model_*`).

## Pontos de Integração

- **llama-server (backends internos)**: encaminhamento httpx existente (`client` compartilhado, `timeout=None` em POST/stream); header `Authorization` repassado como hoje (backend valida `--api-key`). Rotas `/proxy/*` usam `Depends(require_auth)` do manager. Sem retry além do reassign único do PRD F7; `httpx.RequestError` → marca backend na decisão de reassign e responde 502/503 formato OpenAI.

## Análise de Impacto

| Componente | Tipo de Impacto | Descrição e Risco | Ação Necessária |
|---|---|---|---|
| `proxy_router.py` | novo | Toda a lógica de estado; risco baixo (isolado, testável) | Criar com testes unitários dedicados |
| `llama_manager.py` `openai_proxy` | modificado | Desvio condicional + gerador SSE com reescrita; **risco médio**: regressão no caminho atual | Guardas de modo desligado; manter caminho atual byte-a-byte; testes `test_multi_model_proxy.py` devem seguir verdes |
| `_aggregate_models_response` | modificado | Filtro para principal; risco baixo | Condicional em `smart_proxy.enabled` |
| `config_manager.py` / `schemas.py` | modificado | Chave global + 2 campos whitelist; risco baixo (merge preserva desconhecidos) | Novos métodos + defaults |
| `_build_html` + `static/js/` | modificado/novo | Card novo + controles por aba; risco baixo-médio (contrato HTML) | Atualizar `tests/unit/test_html_contract.py` |
| `tests/unit/*` | novo | Cobertura do router e das rotas | `test_proxy_router.py`, `test_smart_proxy_routes.py` |

## Abordagem de Testes

### Testes Unitários

- **`test_proxy_router.py`** (sem FastAPI): extração de afinidade nas 5 camadas e estabilidade da chave; sticky hit; seleção main→principal e subagente→least-busy; restrições de contexto e concorrência; TTL/expiração; reassign em backend morto; persistência (round-trip do JSON, re-vínculo por `model_path`); `dry_run`.
- **`test_smart_proxy_routes.py`** (padrão de `tests/unit/test_multi_model_proxy.py`: `TestClient(app)`, `app.dependency_overrides` para auth, `@patch("llama_manager.client...")` e mock de `process_manager.get_status`): rotas `/proxy/*`, `/v1/models` filtrado, desvio do `/v1/chat/completions` com reescrita, secundário pelo nome real sem sticky, erro sem principal online.
- **Reescrita SSE**: eventos `data: {json}` cortados no meio do nome do modelo entre dois chunks; `data: [DONE]`; linhas keep-alive; evento final com `usage`; garantia de fail-open em linha não-JSON.
- Mock de tempo (TTL) via injeção de `now()` no router — sem `sleep` em testes.

### Testes de Integração

- Cenário 3 backends fake (`httpx.MockTransport` ou mini-apps FastAPI): 3 subagentes tagueados distribuídos, cada um estável no seu backend por N requisições; queda de backend → reassign 1x com log; modo desligado → comportamento atual inalterado.
- Contrato HTML (`test_html_contract.py`) atualizado com o card "Proxy Inteligente". E2E Playwright existente não é pré-requisito desta entrega (suite não versiona config); validação manual no cenário real 3090 + 2× P100 conforme PRD.

## Sequenciamento de Desenvolvimento

### Ordem de Construção

1. **Config e schemas** — `smart_proxy` no `ConfigManager` (get/update), `proxy_eligible`/`max_parallel_requests` na whitelist, `DEFAULT_*` e schemas pydantic novos. Sem dependências.
2. **`proxy_router.py` núcleo** — dataclasses, extração de afinidade, tabela sticky + TTL + persistência, contadores, seleção/elegibilidade, reassign, disable runtime. Depende do passo 1 (lê config).
3. **Integração `/v1`** — desvio no `openai_proxy`, reescrita de body/resposta/SSE (buffer por linha), headers `x-automanager-backend*`, filtro do `/v1/models`, espera de backend ocupado, acquire/release em `finally`. Depende do passo 2.
4. **Endpoints `/proxy/*` e `/models/proxy`** — rotas + wiring com `ProxyRouter` e `ConfigManager`, incluindo `/proxy/resolve` (usa `resolve(dry_run=True)`). Depende dos passos 1 e 2 (independente do 3 para tudo exceto coerência do `resolve`).
5. **Frontend** — card "Proxy Inteligente" no `_build_html`, checkbox global, controles "Principal"/"Participar do proxy" por aba (padrão `persistThinkingEnabled`), `static/js/proxy.js` com polling de `/proxy/status` junto ao `updateStatus()`. Depende dos passos 3 e 4.
6. **Observabilidade e fechamento** — logs `[proxy]` completos, testes de integração do cenário PRD, atualização do contrato HTML. Depende dos passos 3, 4 e 5.

### Dependências Técnicas

- Nenhuma dependência externa nova (httpx, FastAPI, pydantic já presentes).
- Nenhum bloqueio de infraestrutura; feature toggle nasce desligado (`enabled: false`), permitindo merge incremental dos passos 1–4 sem impacto.

## Monitoramento e Observabilidade

- **Logs** (logger `automanager` existente, `manager.log`, formato `%(asctime)s - %(levelname)s - %(message)s`), prefixo `[proxy]` com pares `chave=valor`:
  - Decisão: `[proxy] route external_model=... internal_model=... backend=<porta> gpu=... affinity_key=... sticky_hit=... reason=... stream=... prompt_tokens_estimated=...`
  - Nova sessão: `[proxy] new sticky session affinity_key=... selected_backend=... reason=...`
  - Fallback: `[proxy] backend <porta> unavailable | reassigned affinity_key=... old_backend=... new_backend=... reason=backend_down`
- **Métricas via API**: `/proxy/status` e `/proxy/backends` expõem in-flight, sessões e estados — consumidos pelo painel (polling 1 s existente).
- Sem alerting novo (aplicação single-admin); erros de roteamento respondem em formato de erro OpenAI com `status` 503/502 e são logados em `ERROR`.

## Considerações Técnicas

### Decisões-Chave

- **Estimativa de tokens por `chars//4` com margem de 10%** — justificativa: sem tokenizer no manager; chamada ao `/tokenize` do backend adicionaria latência por requisição. Trade-off: imprecisão em textos não-ingleses; mitigada pela margem e pelo fallback ao principal (maior contexto). Alternativa rejeitada: `POST /tokenize` no backend por decisão.
- **Contexto efetivo por slot = `context_size // parallel_slots`** — coerente com o comportamento do llama-server; evita aceitar requisições que não cabem no slot.
- **Espera por polling (`asyncio.sleep(0.25)`)** em vez de `asyncio.Condition` — volume de concorrência é de um dígito; simplicidade vence.
- **Desvio dentro do `openai_proxy` existente** em vez de rota paralela — um único ponto de entrada `/v1` (ADR-003/004); rejeitado handler duplicado pelo risco de divergência.

### Riscos Conhecidos

- **Regressão no caminho `/v1` atual** (probabilidade baixa, impacto alto): mitigada por guarda de modo desligado no primeiro if e suíte `test_multi_model_proxy.py` intacta como rede de segurança.
- **Formatos SSE variantes entre versões do llama-server** (média/baixo): reescrita fail-open (linha não reconhecida passa intocada) + testes com fixtures reais.
- **Afinidade fraca para clientes sem tag/header** (média/médio): hash estável pode agrupar conversas distintas com prompts iniciais idênticos; `/proxy/resolve` e logs dão diagnóstico; documentação recomenda tags `[AGENT:...]`.
- **Concorrência sobre a tabela sticky** (baixa): todas as mutações sob `asyncio.Lock` único; operações são O(1) e rápidas.

## Registros de Decisão de Arquitetura

- [ADR-001: Seleção de backend por least-busy interno com afinidade sticky](adrs/adr-001.md) — novas sessões vão ao backend elegível menos ocupado, contado pelo manager.
- [ADR-002: Tag [AGENT:...] apenas como chave de afinidade](adrs/adr-002.md) — sem regras fixas tag→backend.
- [ADR-003: Proxy assume o /v1 existente (porta 8000) expondo somente o principal](adrs/adr-003.md) — sem porta nova; secundários acessíveis pelo nome real.
- [ADR-004: Módulo único `proxy_router.py` acoplado ao catch-all `/v1`](adrs/adr-004.md) — lógica com estado isolada e testável; rotas e encaminhamento permanecem no `llama_manager.py`.
- [ADR-005: Estado sticky em memória com persistência JSON atômica; config `smart_proxy` global + flags por modelo](adrs/adr-005.md) — `model_path` como identificador durável; sessões re-vinculam após restart.
- [ADR-006: Reescrita SSE por linha com buffer incremental; modelo real em `x-automanager-backend`](adrs/adr-006.md) — reescrita correta sob cortes de chunk; telemetria preservada.
