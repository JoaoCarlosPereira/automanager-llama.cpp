# TechSpec: Suporte MTP no Automanager

## Resumo Executivo

A feature adiciona dois campos de configuração por modelo (`mtp_enabled`, `mtp_draft_tokens`) propagados do dashboard até o comando `llama-server`, seguindo o pattern existente de `thinking_enabled` / `reasoning_cli_args`. A detecção de compatibilidade usa `llama-server --model-info` parseando `nextn_predict_layers > 0`; flags MTP só entram no cmd quando toggle ligado **e** modelo compatível.

**Trade-off principal:** subprocess extra no start (`--model-info`) em troca de conformidade com ADR-002 (ignorar silenciosamente sem falhar o start).

## Arquitetura do Sistema

### Visão dos Componentes

```
Dashboard (HTML/JS)
  mtp-toggle, mtp-draft-tokens
       │ POST /start { mtp_enabled, mtp_draft_tokens, ... }
       ▼
llama_manager.py ──► schemas.StartRequest (validação Pydantic)
       │
       ├── config_manager.update_model_settings() ──► automanager_config.json
       │
       └── process_manager.start()
              ├── gpu_manager.detect_model_mtp()  [se mtp_enabled]
              ├── mtp_cli_args() ──► [--spec-type draft-mtp, --spec-draft-n-max N]
              └── subprocess.Popen(llama-server ...)
```

| Componente | Responsabilidade |
|------------|------------------|
| `llama_manager.py` | HTML dos campos MTP; endpoint `/start`; auto-start default model |
| `schemas.py` | Campos + validação `mtp_draft_tokens` (1–6) |
| `config_manager.py` | Persistência em `model_configs`; defaults |
| `gpu_manager.py` | `detect_model_mtp()` via model-info |
| `process_manager.py` | `mtp_cli_args()`; integração no cmd; status/recovery/auto-balance |
| `static/js/gpu.js` | `resetToDefaults()`, `updateMtpBadge()` |
| `static/js/models.js` | `applyModelConfig()`, `startModel()` payload |
| `static/js/metrics.js` | Sync campos quando modelo running |

## Design de Implementação

### Interfaces Principais

```python
# schemas.py
DEFAULT_MTP_ENABLED = False
DEFAULT_MTP_DRAFT_TOKENS = 3

class StartRequest(BaseModel):
    # ... campos existentes ...
    mtp_enabled: bool = DEFAULT_MTP_ENABLED
    mtp_draft_tokens: int = Field(
        default=DEFAULT_MTP_DRAFT_TOKENS, ge=1, le=6
    )
```

```python
# process_manager.py
def mtp_cli_args(
    mtp_enabled: bool,
    mtp_draft_tokens: int,
    model_path: str,
    gpu_manager: GPUManager,
) -> List[str]:
    if not mtp_enabled:
        return []
    if not gpu_manager.detect_model_mtp(model_path):
        return []
    n = max(1, min(6, mtp_draft_tokens))
    return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]
```

```python
# gpu_manager.py
def detect_model_mtp(self, model_path: str) -> bool:
    # llama-server --model-info; regex nextn_predict_layers = N; return N > 0
```

### Modelos de Dados

**automanager_config.json** (`model_configs[<path>]`):

| Campo | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `mtp_enabled` | bool | `false` | Toggle MTP |
| `mtp_draft_tokens` | int | `3` | Tokens de predição (1–6) |

Campos omitidos em configs legadas: backend usa defaults; frontend trata ausência como off + 3.

**StartRequest** estende com os dois campos acima. Sem novos endpoints — `/start` e `/status` existentes absorvem os campos.

### Endpoints de API

| Método | Path | Alteração |
|--------|------|-----------|
| POST | `/start` | Aceita `mtp_enabled`, `mtp_draft_tokens`; persiste e propaga ao process_manager |
| GET | `/status` | Retorna `config.mtp_enabled`, `config.mtp_draft_tokens` quando running |
| GET | `/models` | `last_config` inclui campos MTP salvos |

Sem breaking change: campos opcionais com defaults.

## Pontos de Integração

**llama-server (binário externo):**

- Flags: `--spec-type draft-mtp`, `--spec-draft-n-max N`
- Introspecção: `llama-server --model-info "<path>"` → `nextn_predict_layers`
- Requer build llama.cpp com suporte MTP merged

## Análise de Impacto

| Componente | Tipo | Descrição e Risco | Ação |
|------------|------|-------------------|------|
| `schemas.py` | modificado | Novos campos StartRequest | Baixo |
| `config_manager.py` | modificado | merge `mtp_*` em update_model_settings | Baixo |
| `process_manager.py` | modificado | mtp_cli_args + start/auto-balance/OOM/status | Médio |
| `gpu_manager.py` | modificado | detect_model_mtp | Baixo |
| `llama_manager.py` | modificado | HTML + /start + startup auto-start | Médio |
| `static/js/gpu.js` | modificado | reset, badge MTP | Baixo |
| `static/js/models.js` | modificado | apply/start payload | Baixo |
| `static/js/metrics.js` | modificado | sync running config | Baixo |
| `tests/unit/test_mtp_cli_args.py` | novo | testes mtp_cli_args + detect_model_mtp | Baixo |
| `tests/unit/test_html_contract.py` | modificado | asserts mtp-toggle, mtp-draft-tokens | Baixo |
| Jest (`models.test.js`, `metrics.test.js`, `gpu.test.js`) | modificado | novos casos MTP | Baixo |

## Abordagem de Testes

### Testes Unitários (Python)

- `test_mtp_cli_args_enabled_compatible` → flags corretas
- `test_mtp_cli_args_disabled` → `[]`
- `test_mtp_cli_args_incompatible_model` → `[]` (mock detect_model_mtp=False)
- `test_detect_model_mtp_parses_nextn_predict_layers` → fixtures de output model-info
- `test_detect_model_mtp_fallback_on_error` → False

### Testes Unitários (Jest)

- `applyModelConfig` restaura `mtp_enabled` e `mtp_draft_tokens`
- `startModel` inclui campos no JSON POST `/start`
- `resetToDefaults` reseta toggle off + tokens 3
- `updateMtpBadge` ON/OFF

### Contrato HTML

- `id="mtp-toggle"`, `id="mtp-badge"`, `id="mtp-draft-tokens"` presentes

## Sequenciamento de Desenvolvimento

### Ordem de Construção

1. **Constantes + schemas** — `DEFAULT_MTP_*`, campos em `StartRequest` (sem dependências)
2. **config_manager** — persistência `mtp_enabled`, `mtp_draft_tokens` (depende do passo 1)
3. **gpu_manager.detect_model_mtp** — detecção model-info (sem dependências)
4. **process_manager.mtp_cli_args + integração start** — cmd llama-server (depende dos passos 1 e 3)
5. **llama_manager /start + status + auto-start** — propagar campos (depende dos passos 1, 2 e 4)
6. **HTML dashboard** — toggle + input numérico (depende do passo 5)
7. **JS gpu.js / models.js / metrics.js** — bind, apply, start, sync (depende do passo 6)
8. **Auto-balance** — propagar `mtp_*` em `_run_auto_balance` e prober (depende dos passos 4 e 5)
9. **Testes Python + Jest + HTML contract** — (depende dos passos 1–8)

### Dependências Técnicas

- llama-server com MTP merged e `--model-info` emitindo `nextn_predict_layers`
- Nenhuma infraestrutura nova

## Monitoramento e Observabilidade

- Log em `process_manager.start()`: incluir `mtp_enabled`, `mtp_draft_tokens`, `mtp_applied=True/False` na linha START existente
- Quando toggle on mas modelo incompatível: log INFO `"MTP requested but model has no MTP head, skipping flags"`
- Sem métricas novas no MVP

## Considerações Técnicas

### Decisões-Chave

| Decisão | Justificativa |
|---------|---------------|
| Detecção via model-info (ADR-003) | Metadado confiável; evita start failure |
| Helpers em gpu_manager + process_manager (ADR-004) | Consistência com thinking/layers |
| Nomes `mtp_enabled` / `mtp_draft_tokens` | Domínio desacoplado de flags CLI |
| Validação 1–6 tokens | Faixa prática da comunidade |
| `--spec-type draft-mtp` | Flag upstream atual pós-merge MTP |

### Riscos Conhecidos

| Risco | Mitigação |
|-------|----------|
| llama-server antigo sem MTP | Flags ignoradas ou erro no server — documentar versão mínima |
| model-info lento no start | Timeout 15s; só chama quando `mtp_enabled=True` |
| Config legada sem campos | Defaults seguros (off, 3) |

## Registros de Decisão de Arquitetura

- [ADR-001: Campos MTP sempre visíveis](adrs/adr-001.md) — Decisão de produto (UI)
- [ADR-002: Ignorar silenciosamente MTP incompatível](adrs/adr-002.md) — Decisão de produto (comportamento)
- [ADR-003: Detecção via model-info](adrs/adr-003.md) — `nextn_predict_layers > 0`
- [ADR-004: Helpers em gpu_manager + process_manager](adrs/adr-004.md) — Organização backend
