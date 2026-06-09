# TechSpec: Alerta de Nova Versão Disponível

**Slug:** version-update-alert  
**Data:** 2026-06-07  
**Baseado em:** [_prd.md](_prd.md)  
**Status:** Aprovado

---

## Resumo Executivo

Implementar verificação git local-vs-remoto em um módulo backend dedicado (`version_manager.py`) exposto via `GET /api/system/version-check`, e um módulo frontend (`static/js/version.js`) que consulta o endpoint uma vez por carregamento de página e abre modal automático com as notas dos commits.

**Trade-off principal:** subprocess git nativo (sem dependências) em troca de parsing manual de stdout e latência de `git fetch` a cada request — aceitável porque o frontend limita a uma chamada por sessão e falhas são tratadas de forma silenciosa.

**Nota de coexistência:** o botão ATUALIZAR e `POST /api/system/update` já existem no header; o MVP **não altera** esse fluxo. O modal apenas informa; o administrador pode usar o botão existente ou atualizar manualmente no terminal.

---

## Arquitetura do Sistema

### Componentes

```
Browser (version.js)
  │  GET /api/system/version-check (1x por page load, autenticado)
  ▼
llama_manager.py (rota FastAPI)
  │  delega
  ▼
version_manager.py
  │  subprocess: git fetch / rev-parse / log
  ▼
Repositório git em paths.INSTALL_ROOT
```

| Componente | Responsabilidade |
|------------|------------------|
| `version_manager.py` | Detectar work-tree, resolver branch, fetch remoto, listar commits ahead |
| `schemas.py` | `VersionCommit`, `VersionCheckResponse` |
| `llama_manager.py` | Rota autenticada + markup do modal em `_build_html()` |
| `static/js/version.js` | Fetch API, controle do modal, `sessionStorage` dismiss |
| `static/js/index.js` | Wire-up e chamada pós-`initDashboard()` |

### Fluxo de dados

1. Dashboard autenticado carrega → `initDashboard()` → `checkForUpdates()` (async).
2. `version.js` verifica flag interna `checked` e `sessionStorage` dismiss.
3. `apiFetch('/api/system/version-check')` → backend executa git → JSON response.
4. Se `update_available` e não dispensado → exibe `#version-update-modal`.
5. Dispensar → `sessionStorage.setItem('version-update-dismissed','1')` + oculta modal.

---

## Design de Implementação

### Interfaces Principais

```python
# version_manager.py
@dataclass(frozen=True)
class VersionCommit:
    sha: str
    message: str
    author: str
    date: str  # ISO 8601

@dataclass
class VersionCheckResult:
    status: Literal["ok", "unavailable", "error"]
    update_available: bool = False
    current_ref: str | None = None
    remote_ref: str | None = None
    branch: str | None = None
    commits: list[VersionCommit] = field(default_factory=list)
    error_message: str | None = None

def check_for_updates(install_root: str, fetch_timeout: int = 30) -> VersionCheckResult: ...
```

```python
# schemas.py
class VersionCommit(BaseModel):
    sha: str
    message: str
    author: str
    date: str

class VersionCheckResponse(BaseModel):
    status: Literal["ok", "unavailable", "error"]
    update_available: bool = False
    current_ref: Optional[str] = None
    remote_ref: Optional[str] = None
    branch: Optional[str] = None
    commits: List[VersionCommit] = Field(default_factory=list)
    error_message: Optional[str] = None
```

```javascript
// version.js
export async function checkForUpdates() { /* guard checked + sessionStorage */ }
export function dismissVersionModal() { /* sessionStorage + hide */ }
export function showVersionModal(data) { /* populate + display flex */ }
```

### Modelos de Dados

**VersionCheckResponse** — contrato da API:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `status` | `ok \| unavailable \| error` | Resultado da verificação |
| `update_available` | bool | `true` quando `remote_ref` ≠ `current_ref` e há commits ahead |
| `current_ref` | string? | SHA curto do HEAD local |
| `remote_ref` | string? | SHA curto de `origin/<branch>` |
| `branch` | string? | Branch do checkout local |
| `commits` | array | Commits em `HEAD..origin/<branch>`, ordem cronológica (antigo → recente) |
| `error_message` | string? | Detalhe quando `status=error` (não exibido ao usuário no MVP) |

**sessionStorage:** chave `version-update-dismissed`, valor `'1'`.

### Endpoints de API

#### `GET /api/system/version-check`

- **Auth:** `Depends(get_current_auth)` — sessão ou API key.
- **Request:** nenhum body.
- **Response 200:** `VersionCheckResponse`.
- **Response 401:** não autenticado (padrão existente).

**Comportamento backend (`version_manager`):**

1. `git -C <root> rev-parse --is-inside-work-tree` → se falhar: `unavailable`.
2. `git rev-parse --abbrev-ref HEAD` → branch atual.
3. `git fetch --quiet origin <branch>` (timeout 30s) → se falhar: `error`.
4. `rev-parse HEAD` e `rev-parse origin/<branch>`.
5. Se iguais: `ok`, `update_available=false`.
6. Se diferentes: `git log HEAD..origin/<branch> --format=%H%x1f%s%x1f%an%x1f%aI --reverse` → parse em commits.

---

## Pontos de Integração

| Sistema | Integração |
|---------|------------|
| Binário `git` no PATH | Invocado via subprocess; pré-requisito do instalador (`setup.sh` já instala git) |
| Remoto `origin` | Padrão git; credenciais SSH/HTTPS do ambiente do servidor |
| `paths.INSTALL_ROOT` | Diretório raiz do repositório |

Sem integrações externas de rede além do `git fetch` para o remoto já configurado.

---

## Análise de Impacto

| Componente | Impacto | Risco | Ação |
|------------|---------|-------|------|
| `version_manager.py` | Novo | Baixo | Criar módulo + testes |
| `schemas.py` | Modificado | Baixo | Adicionar 2 modelos Pydantic |
| `llama_manager.py` | Modificado | Médio | Nova rota + modal HTML + cache-bust JS |
| `static/js/version.js` | Novo | Baixo | Lógica modal |
| `static/js/index.js` | Modificado | Baixo | Import + chamada checkForUpdates |
| `static/js/models.js` | Modificado | Baixo | Invocar checkForUpdates no fim de initDashboard |
| `tests/unit/test_version_manager.py` | Novo | Baixo | Mocks subprocess |
| `tests/integration/test_api_endpoints.py` | Modificado | Baixo | Teste do endpoint |
| `static/js/version.test.js` | Novo | Baixo | Jest para modal/dismiss |
| `POST /api/system/update` | Inalterado | — | Coexistência documentada |

---

## Abordagem de Testes

### Unitários (`test_version_manager.py`)

- Work-tree válido, remoto ahead → commits parseados corretamente.
- HEAD == origin/branch → `update_available=false`.
- Não é repositório git → `unavailable`.
- `git fetch` timeout/erro → `error`.
- Branch com caracteres especiais.
- Lista longa de commits (sem truncamento).

Mock: `subprocess.run` com side_effect por comando.

### Integração (`test_api_endpoints.py`)

- `GET /api/system/version-check` sem auth → 401.
- Com auth + mock de `version_manager.check_for_updates` → 200 com payload esperado.

### Frontend (`version.test.js`)

- `checkForUpdates` chama API uma vez (flag `checked`).
- Modal abre quando `update_available=true`.
- Dismiss grava `sessionStorage` e não reabre.
- `status=unavailable/error` → sem modal.
- Esc fecha modal.

---

## Sequenciamento de Desenvolvimento

### Ordem de Construção

1. **Modelos Pydantic** em `schemas.py` — sem dependências.
2. **`version_manager.py`** — depende dos dataclasses internos; testes unitários.
3. **Rota `GET /api/system/version-check`** em `llama_manager.py` — depende de (1) e (2).
4. **Testes de integração** do endpoint — depende de (3).
5. **Markup do modal** em `_build_html()` — sem dependência de JS.
6. **`static/js/version.js`** — depende do contrato da API (3) e markup (5).
7. **Wire-up** em `index.js` + `models.js` + `_dashboard_js_version()` — depende de (6).
8. **Testes Jest** `version.test.js` — depende de (6) e (7).

### Dependências Técnicas

- `git` instalado e no PATH (já requisito do instalador).
- Remoto `origin` configurado com acesso de rede do servidor.
- Nenhuma dependência pip adicional.

---

## Monitoramento e Observabilidade

- **Log INFO:** verificação iniciada, branch, resultado (`ahead=N` ou `up-to-date`).
- **Log WARNING:** fetch falhou, timeout, não é work-tree.
- **Log ERROR:** exceção inesperada no subprocess.
- **Não logar:** mensagens completas de commit (volume desnecessário).
- **Métrica operacional:** latência do endpoint (observável via logs de request FastAPI existentes).

---

## Considerações Técnicas

### Decisões-Chave

| Decisão | Justificativa | Trade-off |
|---------|---------------|-----------|
| subprocess git | Padrão do projeto; zero deps | Parsing manual |
| Branch do checkout | Coerente com `git pull` do botão ATUALIZAR | Forks em branch não-main precisam estar no branch desejado |
| Sem cache server | Simplicidade; 1 call/sessão no frontend | Fetch repetido se Fase 2 adicionar botão manual |
| sessionStorage dismiss | Sobrevive reload na aba | Nova aba = nova sessão = modal reaparece |
| Sem limite de commits | Decisão do usuário; rolagem no modal | Modal longo se muito defasado |

### Riscos Conhecidos

| Risco | Prob. | Mitigação |
|-------|-------|-----------|
| `git fetch` lento | Média | Timeout 30s; chamada async no frontend |
| Credenciais git ausentes | Média | `status=error`; dashboard inalterado |
| SHAs locais divergentes por merge local | Baixa | Comparação por ref, não por tag semver |
| Modal compete com alertas existentes | Baixa | z-index abaixo ou igual ao login-overlay (z-50) |

---

## Registros de Decisão de Arquitetura

- [ADR-001: Modal Automático na Abertura do Dashboard](adrs/adr-001.md) — Decisão de produto: modal automático, verificação por sessão, update manual.
- [ADR-002: Módulo version_manager com subprocess git](adrs/adr-002.md) — Backend via subprocess, branch do checkout, sem cache.
- [ADR-003: Frontend version.js com modal e sessionStorage](adrs/adr-003.md) — Módulo JS dedicado, dismiss em sessionStorage, markup no template.
