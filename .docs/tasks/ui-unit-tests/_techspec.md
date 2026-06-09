# TechSpec: Testes de Interface para AutoManager Llama.cpp

> **Idioma:** escreva todo o conteúdo de `_techspec.md` em **português brasileiro (PT-BR)**.

## Resumo Executivo

Esta especificação define a implementação de uma suíte de testes para a dashboard web do AutoManager Llama.cpp. A estratégia consiste em três camadas: **(1)** testes unitários Jest para funções JavaScript extraídas de `llama_manager.py` para arquivos modulares em `static/js/`, **(2)** testes de contrato HTML via Python (pytest + httpx TestClient) validando a resposta HTML gerada pelo `_build_html()`, e **(3)** testes E2E Playwright (Chromium) validando interações do usuário com um servidor FastAPI mock.

**Trade-off principal:** A extração do JS de f-string para módulos ES6 aumenta o esforço inicial de refatoração, mas elimina a principal limitação do projeto — a impossibilidade de testar funções JS isoladamente. O custo de manter dois arquivos por função (produção + teste) é compensado pela detecção imediata de regressões e pela possibilidade de refatorar com segurança.

## Arquitetura do Sistema

### Visão dos Componentes

A implementação introduz três novos componentes de teste sem alterar a lógica de negócio do backend:

| Componente | Responsabilidade | Tipo |
|------------|-----------------|------|
| `static/js/*.js` | Módulos JS extraídos da f-string (auth, models, metrics, gpu, index) | Novo |
| `static/js/*.test.js` | Testes Jest para cada módulo JS | Novo |
| `tests/e2e/mock_server.py` | Servidor FastAPI com dependency_overrides para E2E testing | Novo |
| `tests/e2e/*.spec.ts` | Testes E2E Playwright (login, start/stop, monitoring, models) | Novo |
| `jest.config.js` | Configuração Jest (JSDOM, módulos, cobertura) | Novo |
| `playwright.config.ts` | Configuração Playwright (Chromium, webServer mock) | Novo |
| `package.json` | Dependências npm (jest, playwright) e scripts | Novo |
| `_build_html()` | Modificado para servir scripts externos ao invés de embutidos | Modificado |

**Fluxo de dados:**

```
llama_manager.py (_build_html modificado)
  → serve static/js/{auth,models,metrics,gpu,index}.js via StaticFiles
  → index.js importa módulos e inicializa dashboard

Jest (node)
  → importa módulos de static/js/ via ES module
  → executa em JSDOM (DOM fake)
  → valida comportamento de funções isoladas

Playwright (chromium)
  → conecta a mock_server.py (FastAPI com dependency_overrides)
  → navega na dashboard real com JS real
  → interage com DOM real (clicar, digitar, verificar)

pytest (Python)
  → usa httpx TestClient em llama_manager.app
  → valida HTML retornado por GET / (contract tests)
```

## Design de Implementação

### Extração de JavaScript para Módulos

O JavaScript atual em `llama_manager.py` (linhas 969-2074) será extraído para 5 módulos ES6. Cada função que atualmente está no escopo global do `<script>` será convertida para `export function` e referenciada via `window` para compatibilidade com `onclick` no HTML.

#### Módulo: `auth.js`

```javascript
// static/js/auth.js

export let sessionExpiredHandled = false;

export function showAlert(msg) { alert(msg); }
export function showConfirm(msg) { return confirm(msg); }

export async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, password}),
        });
        if (res.ok) {
            sessionExpiredHandled = false;
            const errEl = document.getElementById('login-error');
            if (errEl) {
                errEl.textContent = '';
                errEl.classList.add('hidden');
            }
            document.getElementById('login-overlay').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            window.initDashboard();
            window.startDashboardPolling();
        } else {
            const err = await res.json();
            const el = document.getElementById('login-error');
            el.textContent = err.detail || 'Erro no login';
            el.classList.remove('hidden');
        }
    } catch (e) {
        const el = document.getElementById('login-error');
        el.textContent = 'Erro de rede';
        el.classList.remove('hidden');
    }
}

export async function handleLogout() {
    try { await fetch('/api/auth/logout', {method: 'POST'}); } catch (e) {}
    location.reload();
}

export async function changePassword() { /* ... */ }
export function handleSessionExpired(message) { /* ... */ }

export async function apiFetch(url, options) {
    const res = await fetch(url, options);
    if (res.status === 401) {
        handleSessionExpired('Sessao expirada.');
    }
    return res;
}
```

#### Módulo: `index.js`

```javascript
// static/js/index.js

import { handleLogin, handleLogout, changePassword } from './auth.js';
import { startModel, stopModel, renameModel, deleteModel } from './models.js';
import { updateMetrics, updateStatus, startLogs } from './metrics.js';
import { updateTotal, resetToDefaults } from './gpu.js';

// Estado global compartilhado
export let statusPollTimer = null;
export let metricsTimer = null;
export let downloadsTimer = null;
export let modelsTimer = null;
export let logStream = null;
export let startTime = null;

export function initDashboard() {
    // ... original logic
}

export function startDashboardPolling() {
    // ... original logic
}

export function stopDashboardPolling() {
    // ... original logic
}

// Inicialização quando DOM estiver pronto
document.getElementById('chat-link').href = `http://${window.fixedIp}:8085/`;
document.getElementById('api-link').innerText = `http://${window.fixedIp}:8085/v1`;

if (document.getElementById('dashboard').style.display !== 'none') {
    initDashboard();
    startDashboardPolling();
}
```

### Mock Server para E2E

```python
# tests/e2e/mock_server.py
"""Servidor FastAPI mock para testes E2E Playwright."""

import json
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from llama_manager import app, auth_manager, _build_html, config_manager
from llama_manager import DEFAULT_CONTEXT_SIZE, DEFAULT_PARALLEL_SLOTS, DEFAULT_BATCH_SIZE

mock_server_app = FastAPI()

# Estado do mock
_mock_state = {
    "running": False,
    "model": None,
    "start_time": None,
}

# Mock de dados fake
_FAKE_MODELS = [
    {"id": "m1", "name": "llama-3.1-8b.gguf", "path": "/models/llama-3.1-8b.gguf", "dir": "/models"},
    {"id": "m2", "name": "mistral-7b.gguf", "path": "/models/mistral-7b.gguf", "dir": "/models"},
]

@mock_server_app.post("/api/auth/login")
async def mock_login(request: Request):
    """Aceita qualquer credencial e retorna cookie de sessão."""
    body = await request.json()
    # Retorna cookie de sessão válido
    response = JSONResponse({"detail": "logged in"})
    response.set_cookie(
        key="session",
        value=auth_manager.create_session(body.get("username", "admin"), "admin"),
        httponly=True,
        max_age=3600,
    )
    return response

@mock_server_app.get("/status")
async def mock_status():
    return _mock_state

@mock_server_app.post("/stop")
async def mock_stop():
    _mock_state.update({"running": False, "model": None})
    return {"status": "ok"}

@mock_server_app.post("/start")
async def mock_start(request: Request):
    body = await request.json()
    _mock_state.update({"running": True, "model": body.get("path"), "start_time": 1000000})
    return {"probing": False, "status": "ok"}

@mock_server_app.get("/metrics")
async def mock_metrics():
    return {
        "cpu": 42.5, "ram": 67.2,
        "gpus": [
            {"index": 0, "util": 85, "temp": 72, "power": 240,
             "mem_used": 12000, "mem_total": 24564, "vram_pct": 48}
        ]
    }

@mock_server_app.get("/models")
async def mock_models():
    return {"models": _FAKE_MODELS, "projectors": []}

@mock_server_app.get("/logs")
async def mock_logs():
    """SSE stream fake com linhas controladas."""
    from fastapi.responses import StreamingResponse
    async def event_stream():
        for line in ["[INFO] llama server started", "[INFO] model loaded"]:
            yield f"data: {line}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Mount mock routes on the main app
for route in mock_server_app.routes:
    if hasattr(route, 'path') and route.path:
        # Replace original route with mock
        app.router.routes = [r for r in app.router.routes if r.path != route.path]
        app.routes.append(route)

# Dependency overrides para autenticação
def fake_auth():
    return {"sub": "admin", "role": "admin"}

app.dependency_overrides[auth_manager.get_current_user] = fake_auth
```

### Testes Jest — Exemplo de Estrutura

```javascript
// static/js/auth.test.js
import { handleLogin, showAlert, showConfirm, sessionExpiredHandled } from './auth.js';

// Mock do DOM e fetch
beforeEach(() => {
    document.body.innerHTML = `
        <div id="login-overlay"><form id="login-form"><input id="login-username"/><input id="login-password"/></form></div>
        <div id="dashboard" style="display:none;"></div>
        <p id="login-error" class="hidden"></p>
    `;
    global.fetch = jest.fn();
    window.initDashboard = jest.fn();
    window.startDashboardPolling = jest.fn();
});

test('handleLogin mostra erro ao falhar autenticação', async () => {
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({ detail: 'Credenciais invalidas' }),
    });
    
    const form = document.getElementById('login-form');
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    
    await new Promise(r => setTimeout(r, 0));
    const errEl = document.getElementById('login-error');
    expect(errEl.textContent).toBe('Credenciais invalidas');
});

test('handleLogin esconde overlay e mostra dashboard ao sucesso', async () => {
    fetch.mockResolvedValue({ ok: true });
    
    const form = document.getElementById('login-form');
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    
    await new Promise(r => setTimeout(r, 0));
    expect(document.getElementById('login-overlay').style.display).toBe('none');
    expect(document.getElementById('dashboard').style.display).toBe('block');
    expect(window.initDashboard).toHaveBeenCalled();
});
```

### Testes de Contrato HTML — Exemplo de Estrutura

```python
# tests/unit/test_html_contract.py
from fastapi.testclient import TestClient
from llama_manager import app, _build_html, config_manager

client = TestClient(app)

def test_html_contains_login_overlay():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'id="login-overlay"' in html
    assert 'id="login-form"' in html
    assert 'id="login-username"' in html
    assert 'id="login-password"' in html
    assert 'onsubmit="handleLogin(event)"' in html

def test_html_contains_status_badge():
    assert 'id="status-badge"' in response.text
    assert 'OFFLINE' in response.text

def test_html_contains_metrics_panel():
    assert 'id="metrics-panel"' in response.text
    assert 'id="cpu-val"' in response.text
    assert 'id="cpu-bar"' in response.text
    assert 'id="ram-val"' in response.text
    assert 'id="ram-bar"' in response.text

def test_html_contains_gpu_table():
    assert 'id="gpu-table-body"' in response.text
    assert 'gpu-row' in response.text
    assert 'gpu-weight' in response.text
    assert 'gpu-checkbox' in response.text
    assert 'gpu-pin' in response.text

def test_html_contains_model_list():
    assert 'id="model-list-container"' in response.text
    assert 'model-item-container' in response.text

def test_html_contains_log_terminal():
    assert 'id="log-box"' in response.text
    assert 'Limpar' in response.text

def test_html_contains_pacman_canvas():
    assert 'id="pacman-background"' in response.text
    assert 'aria-hidden="true"' in response.text

def test_html_contains_active_model_card():
    assert 'id="active-card"' in response.text
    assert 'id="active-model-name"' in response.text
    assert 'id="uptime-val"' in response.text

def test_html_serves_external_js_scripts():
    assert 'src="/static/js/auth.js"' in response.text
    assert 'src="/static/js/models.js"' in response.text
    assert 'src="/static/js/metrics.js"' in response.text
    assert 'src="/static/js/gpu.js"' in response.text
    assert 'src="/static/js/index.js"' in response.text
```

### Testes E2E Playwright — Exemplo de Estrutura

```typescript
// tests/e2e/login.spec.ts
import { test, expect } from '@playwright/test';

test('login com credenciais validas mostra dashboard', async ({ page }) => {
    await page.goto('/');
    
    // Verifica overlay de login
    await expect(page.locator('#login-overlay')).toBeVisible();
    await expect(page.locator('#login-username')).toBeVisible();
    await expect(page.locator('#login-password')).toBeVisible();
    
    // Faz login
    await page.locator('#login-username').fill('admin');
    await page.locator('#login-password').fill('password');
    await page.locator('#login-form').submit();
    
    // Verifica dashboard carregado
    await expect(page.locator('#dashboard')).toBeVisible();
    await expect(page.locator('#login-overlay')).not.toBeVisible();
    await expect(page.locator('#status-badge')).toBeVisible();
    await expect(page.locator('#cpu-val')).toBeVisible();
    await expect(page.locator('#model-list-container')).toBeVisible();
});

test('login com credenciais invalidas mostra erro', async ({ page }) => {
    // Mock server retorna 401 para credenciais invalidas
    // ... (depende da implementacao do mock)
});
```

### Endpoints de API — Cobertura de Testes

| Endpoint | Método | Teste Unitário Jest | Teste HTML Contract | Teste E2E Playwright |
|----------|--------|-------------------|-------------------|-------------------|
| `/api/auth/login` | POST | ✓ (mock fetch) | — | ✓ (fluxo real) |
| `/api/auth/logout` | POST | ✓ (mock fetch) | — | ✓ |
| `/api/auth/change-password` | POST | ✓ (mock fetch) | — | — |
| `/status` | GET | — | ✓ (presença de #status-badge) | ✓ (verifica ONLINE/OFFLINE) |
| `/metrics` | GET | — | ✓ (presença de #cpu-val, #ram-val) | ✓ (atualizacao de valores) |
| `/models` | GET | — | ✓ (presenca de #model-list-container) | ✓ (lista de modelos) |
| `/logs` | GET | — | ✓ (presenca de #log-box) | ✓ (SSE streaming) |
| `/start` | POST | — | — | ✓ (fluxo completo) |
| `/stop` | POST | — | — | ✓ (fluxo completo) |
| `/rename` | POST | — | — | ✓ (renomear modelo) |
| `/delete` | POST | — | — | ✓ (deletar modelo) |
| `/downloads` | POST/GET | — | ✓ (presenca de #download-status) | ✓ (progresso download) |
| `/api/key` | GET | — | ✓ (presenca de #api-token) | — |
| `/api/key/renew` | POST | — | — | — |
| `/set_default` | POST | — | ✓ (presenca de .model-default-checkbox) | — |
| `/` (HTML) | GET | — | ✓ (8+ contratos de elementos) | ✓ (dashboard carregado) |

## Análise de Impacto

| Componente | Tipo de Impacto | Descrição e Risco | Ação Necessária |
|------------|-----------------|-------------------|-----------------|
| `llama_manager.py` | Modificado | `_build_html()` alterado para referenciar scripts externos ao invés de JS embutido. Risco médio: se a ordem de scripts for errada, dashboard não inicializa. | Refatorar `_build_html()` (linhas 969-2074): extrair `<script>` embutido para `<script type="module">` com `src` externos |
| `static/` (diretório) | Novo | Criar subpasta `js/` para módulos JS e testes Jest. Risco baixo: novo diretório sem conflitos. | Criar `static/js/{auth,models,metrics,gpu,index}.js` + `*.test.js` |
| `tests/` (diretório) | Modificado/Novo | Criar `tests/e2e/` para Playwright. Manter estrutura pytest existente. Risco baixo: sem sobreposição com testes existentes. | Criar `tests/e2e/conftest.py`, `mock_server.py`, `*.spec.ts` |
| `requirements.txt` | Modificado | Adicionar `playwright` para Python e `pytest-playwright` (opcional). Risco baixo: dependência isolada. | Adicionar `playwright>=1.40.0` |
| `package.json` | Novo | Raiz do projeto com Jest e Playwright via npm. Risco baixo: `node_modules/` ignorado pelo git. | Criar na raiz |
| `jest.config.js` | Novo | Configuração Jest. Risco baixo: sem impacto em runtime. | Criar na raiz |
| `playwright.config.ts` | Novo | Configuração Playwright. Risco baixo: sem impacto em runtime. | Criar na raiz |
| `.gitignore` | Modificado | Adicionar `node_modules/`, `coverage/`, `*.d.ts`, `.playwright/`. Risco baixo. | Adicionar entradas |

## Abordagem de Testes

### Testes Unitários (Jest)

**Estratégia:** Cada módulo de `static/js/` terá seu arquivo `.test.js` correspondente. Jest roda em JSDOM — o DOM é simulado, não há navegador real.

**Cobertura alvo:** 90% de statement, branch, function e line coverage por módulo.

**Módulos e funções a testar:**

| Módulo | Funções | Cenários Críticos |
|--------|---------|-------------------|
| `auth.js` | `handleLogin`, `handleLogout`, `changePassword`, `apiFetch`, `handleSessionExpired` | Login sucesso/falha, sessão expirada, rede indisponível, validação de senha |
| `models.js` | `startModel`, `stopModel`, `renameModel`, `deleteModel`, `setDefaultModel`, `selectModel`, `downloadModel` | Start com GPU válida, start com hardware incapaz, rename com mesmo nome, delete com confirmação |
| `metrics.js` | `updateMetrics`, `updateStatus`, `updateUptime`, `startLogs`, `stopDashboardPolling` | Atualização de barras de CPU/RAM, parsing de dados GPU, polling interval, SSE abort |
| `gpu.js` | `getContextSize`, `setContextSize`, `updateTotal`, `redistributeUnpinnedWeights`, `balanceWeights`, `applyGpuWeightsToUI`, `resetToDefaults` | Contexto custom vs preset, total = 100%, redistribuição com pinned/unpinned, balanceWeights delega |
| `index.js` | `initDashboard`, `startDashboardPolling`, `stopDashboardPolling` | Inicialização correta, timers criados e destruídos |

**Mocks:**

- `fetch` → Jest mock que retorna responses controladas
- `alert()`/`confirm()` → Jest mock via `global.alert = jest.fn()`, `global.confirm = jest.fn()`
- `document.getElementById()` → JSDOM DOM (configurado nos `beforeEach`)
- `setInterval`/`clearInterval` → Jest mock para evitar timers reais

### Testes de Contrato HTML (pytest)

**Estratégia:** httpx TestClient faz `GET /` e valida que o HTML retornado contém os IDs, classes e atributos esperados.

**Cenários:**
- HTML contém todos os overlays, painéis e controles (8+ elementos por contrato)
- Scripts externos são referenciados corretamente (5 módulos)
- IP dinâmico é injetado no HTML
- Token de API aparece no HTML
- Login overlay respeita estado de autenticação

### Testes E2E (Playwright)

**Estratégia:** Playwright conecta ao mock_server.py (FastAPI com dependency_overrides). O navegador executa JavaScript real contra um backend controlado.

**Fluxos E2E por prioridade:**

| Fluxo | Prioridade | Passos |
|-------|------------|--------|
| Login sucesso | P0 | Preencher form → submit → verificar dashboard visível → verificar componentes |
| Login falha | P0 | Preencher credenciais inválidas → submit → verificar mensagem de erro |
| Start modelo | P1 | Selecionar modelo → clicar load → verificar status ONLINE → verificar card ativo |
| Stop modelo | P1 | Clicar encerrar → confirmar → verificar status OFFLINE |
| Métricas atualizam | P2 | Verificar que CPU/RAM aparecem com valores não-zero |
| Logs SSE | P2 | Verificar que terminal exibe linhas de log |
| Download | P3 | Preencher URL → clicar download → verificar progresso |
| Rename modelo | P3 | Clicar rename → confirmar novo nome → verificar nome atualizado |
| Delete modelo | P3 | Clicar delete → confirmar → verificar sumário atualizado |

### Testes de Integração Existentes

**Nenhum teste existente será alterado.** Os testes em `tests/unit/` e `tests/integration/` continuam funcionando com `pytest` normal. A nova suíte de testes de UI é complementar, não substituta.

## Sequenciamento de Desenvolvimento

### Ordem de Construção

1. **Configuração de ferramentas** — Criar `package.json`, `jest.config.js`, `playwright.config.ts`, instalar dependências (npm + pip). **Sem dependências.**
2. **Refatorar `_build_html()`** — Substituir `<script>` embutido por 5 `<script type="module">` com `src` externos. **Depende do passo 1.**
3. **Extrair `auth.js` + `auth.test.js`** — Primeiro módulo extraído e testado, serve como referência. **Depende do passo 2.**
4. **Extrair `metrics.js` + `metrics.test.js`** — Segundo módulo, usa `auth.js` (apiFetch). **Depende do passo 3.**
5. **Extrair `models.js` + `models.test.js`** — Terceiro módulo, usa `auth.js` e `metrics.js`. **Depende do passo 4.**
6. **Extrair `gpu.js` + `gpu.test.js`** — Quarto módulo, usa funções auxiliares de contexto. **Depende do passo 4.**
7. **Extrair `index.js`** — Arquivo principal que importa todos os módulos. **Depende dos passos 3-6.**
8. **Criar `tests/e2e/mock_server.py`** — Servidor FastAPI mock com dependency_overrides. **Depende do passo 2 (app importável).**
9. **Criar `tests/e2e/login.spec.ts`** — Primeiro teste E2E (fluxo crítico). **Depende do passo 8.**
10. **Criar `tests/unit/test_html_contract.py`** — Testes de contrato HTML com pytest. **Depende do passo 2.**
11. **Adicionar testes E2E restantes** — start/stop, métricas, modelos. **Depende do passo 9.**

### Dependências Técnicas

- **Node.js 18+** — Necessário para Jest e Playwright via npm
- **Python 3.11+** — Já disponível, para pytest e mock_server
- **Browsers (Playwright)** — `npx playwright install chromium` após npm install
- **GPU não necessária** — Mock server não requer llama-server ou hardware GPU

## Monitoramento e Observabilidade

### Métricas de Teste

| Métrica | Valor-Alvo | Como Medir |
|---------|------------|------------|
| Tempo suite completa | < 5 min | `npm test -- --json` + `pytest --tb=short` + `playwright test --reporter=json` |
| Cobertura JS | ≥ 90% | `jest --coverage` |
| Taxa de flakiness | < 2% | Contar falhas intermitentes em 50 execuções |
| Tempo Jest (unit) | < 30s | `npm test -- --json` |
| Tempo E2E | < 3 min | `playwright test --reporter=json` |

### Logs e Debug

- **Jest:** `npm test -- --verbose` para logs detalhados de cada teste
- **Playwright:** `PLAYWRIGHT_HTML_OPEN=never npx playwright test --headed` para debug visual
- **Playwright traces:** configurados com `trace: 'on-first-retry'` para debug automático em retry
- **Screenshots:** configurados com `screenshot: 'only-on-failure'` para captura automática em falha

## Considerações Técnicas

### Decisões-Chave

| Decisão | Escolhida | Justificativa | Alternativas Rejeitadas |
|---------|-----------|---------------|------------------------|
| JS extraído para módulos ES6 | 5 arquivos separados | Separação clara de responsabilidades, fácil testar isoladamente | Arquivo único (não organizável), classes (overhead desnecessário) |
| Mock total no E2E | Mock server FastAPI | Isolamento total, rodar sem GPU, determinístico | Backend real (depende de hardware), mock seletivo (estado inconsistente) |
| Jest via npm | Jest + JSDOM | Padrão da indústria, testes ultra-rápidos, sem navegador | Vitest (menos maduro), pytest (não suporta JS unit) |
| Playwright via npm | Playwright + Chromium | Navegador real, interações completas | pytest-playwright (apenas E2E, sem unit), Cypress (ecossistema menor para Python) |
| Threshold de cobertura | 90% | Qualidade alta sem ser impraticável | Sem threshold (qualidade não medida), 80% (limiar baixo para projeto crítico) |
| Navegadores E2E | Chromium apenas | Cobertura suficiente para SPA vanilla, menor overhead | Chromium+Firefox (mais overhead), 3 navegadores (muito lento para execução frequente) |

### Riscos Conhecidos

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| Refatoração do JS quebra funcionalidade existente | Média | Refatoração incremental — extrair módulo por módulo, rodar suite após cada etapa |
| Mock server desatualizado não cobre novo endpoint | Média | Checklist de desenvolvimento: "adicionar mock para novo endpoint" |
| Jest não consegue testar funções que acessam DOM diretamente | Baixa | JSDOM suporta a maioria das APIs DOM; para as raras que não suporta, usar wrappers |
| Desenvolvedores não têm Node.js instalado | Média | Documentar instalação no README.md, incluir como requisito de dev |
| Testes E2E ficam lentos com muitos fluxos | Baixa | Playwright é rápido para SPA simples; limitar E2E aos fluxos mais críticos |

## Registros de Decisão de Arquitetura

- [ADR-001: Extração de JavaScript para testes unitários com Jest + Playwright E2E](adrs/adr-001.md) — Decisão por extrair todo o JS embutido e adotar Jest para unit tests + Playwright para E2E
- [ADR-002: Estrutura modular dos arquivos JavaScript](adrs/adr-002.md) — Decisão por dividir o JS em 5 módulos (auth, models, metrics, gpu, index) mantendo funções puras
- [ADR-003: Estratégia de mock para testes E2E com Playwright](adrs/adr-003.md) — Decisão de mockar todas as respostas da API via FastAPI dependency_overrides
- [ADR-004: Configuração Jest e Playwright para testes de UI](adrs/adr-004.md) — Decisão por Jest via npm (JSDOM) + Playwright via npm (Chromium), com threshold de 90% de cobertura
