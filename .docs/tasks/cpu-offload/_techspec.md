# TechSpec — CPU Offload no Automanager Llama.cpp

## 1. Resumo Executivo

Adiciona suporte a CPU offload ao Automanager Llama.cpp, permitindo que modelos excedam o VRAM total das GPUs usando o processador como recurso complementar. A CPU aparece como uma linha adicional na tabela "Recursos de GPU & Configuração", seguindo o mesmo padrão visual das GPUs, com controle de peso compartilhado (CPU + GPUs = 100%). O Auto Balance prioriza GPUs como dispositivos de compute principal, usando a CPU apenas como último recurso.

**Trade-off principal:** Estender o schema `GPUWeight` existente com um campo `device` ("gpu"|"cpu") vs. criar um schema separado `CPUWeight`. A escolha é estender o schema existente para minimizar mudanças — um único campo booleano/enum adicionado — mantendo a mesma UI e lógica de distribuição de pesos.

## 2. Contexto

O Automanager Llama.cpp é uma aplicação FastAPI com painel HTML/JS embutido que orquestra instâncias do `llama-server`. Atualmente:

- GPUs são detectadas via `llama-server --help` e `nvidia-smi` (gpu_manager.py)
- Distribuição de peso é feita exclusivamente entre GPUs (schemas.GPUWeight, auto_balance.py)
- `--n-gpu-layers` está hardcoded para `99` em process_manager.py (todas as camadas na GPU)
- psutil já é dependência: usado em gpu_manager.py para `cpu_percent()` e `virtual_memory()`
- Métricas são polidas via `/metrics` a cada 2s
- Frontend modular em `static/js/` (gpu.js, metrics.js, models.js, state.js)
- Linha da CPU/RAM já existe no painel superior como card de métricas de host (não na tabela de dispositivos)

## 3. Objetivos Técnicos

| Objetivo | Detalhamento |
|----------|-------------|
| Detectar nome do processador e RAM do sistema | Usar `platform.processor()` e `psutil` para obter informações do CPU |
| Exibir CPU na tabela de dispositivos | Adicionar linha com mesmo padrão visual das GPUs (checkbox, nome, métricas, peso) |
| Monitorar CPU usage e RAM em tempo real | Usar psutil via endpoint `/metrics` existente |
| Permitir distribuição de peso unificada | CPU + GPUs compartilham 100%, com pinagem e validação |
| Calcular `--n-gpu-layers` dinamicamente | Substituir hardcoded `99` por cálculo baseado nos pesos das GPUs |
| Priorizar GPUs no Auto Balance | Algoritmo tenta maximizar GPUs primeiro, CPU só como último recurso (max 70%) |

## 4. Design Arquitetural

### 4.1. Visão Geral das Mudanças

A implementação é incremental e segue o padrão de extensibilidade da codebase existente. Nenhuma nova pasta ou módulo é criado. As mudanças são localizadas em:

- **Backend Python:** `schemas.py`, `gpu_manager.py`, `process_manager.py`, `auto_balance.py`, `llama_manager.py`
- **Frontend JS:** `static/js/gpu.js`, `static/js/metrics.js`, `static/js/models.js`
- **Frontend HTML:** Template embutido em `llama_manager.py` (função `index()`)

### 4.2. Modelo de Dados

#### 4.2.1. Extensão do Schema GPUWeight

`schemas.py` — campo `device` adicionado ao modelo `GPUWeight`:

```python
class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str
    active: bool = True
    is_main: bool = False
    pinned: bool = False
    device: str = "gpu"  # "gpu" | "cpu"
```

- Dispositivos GPU mantêm `device="gpu"` (padrão) e `index >= 0`
- CPU usa `device="cpu"`, `index=-1` (convenção para não conflitar com índices GPU)
- `is_main` é sempre `False` para CPU (CPU não pode ser principal)
- backward-compatible: campo tem default, requests antigos funcionam

#### 4.2.2. Estrutura de Métricas

O response do endpoint `/metrics` (gpu_manager.py `get_metrics()`) será estendido:

```python
{
    "cpu": float,       # usage % via psutil.cpu_percent()
    "ram": float,       # usage % via psutil.virtual_memory().percent
    "ram_used_mb": int, # RAM usada em MB
    "ram_total_mb": int,# RAM total em MB
    "gpus": [ ... ],    # existente, sem mudança
    "cpu_name": str,    # Nome completo do processador
}
```

### 4.3. Detecção de Hardware

**gpu_manager.py** — método `detect_cpu_info()`:

```python
def detect_cpu_info(self) -> dict:
    """Detecta nome do processador e informações de RAM."""
    try:
        cpu_name = platform.processor() or platform.machine()
    except Exception:
        cpu_name = "CPU Host"
    
    try:
        mem = psutil.virtual_memory()
        return {
            "name": cpu_name or "CPU Host",
            "ram_used_mb": mem.used // (1024 * 1024),
            "ram_total_mb": mem.total // (1024 * 1024),
            "ram_percent": mem.percent,
        }
    except Exception:
        return {"name": "CPU Host", "ram_used_mb": 0, "ram_total_mb": 0, "ram_percent": 0}
```

- `platform.processor()` retorna o nome no Windows (ex: "Intel64 Family 6 Model 79 Stepping 0, Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz")
- Fallback: `platform.machine()` ou string genérica "CPU Host"
- psutil para RAM: já é dependência

**gpu_manager.py** — método `detect_gpus()` retorna lista sem mudança; a CPU é adicionada no HTML no lado do servidor (`llama_manager.py`).

### 4.4. Distribuição de Peso

#### 4.4.1. Validação

`process_manager.py` — nova função de validação:

```python
def validate_weights(gpu_weights: List[GPUWeight]) -> bool:
    """Valida que soma dos pesos = 100% e CPU <= 70%."""
    total = sum(w.weight for w in gpu_weights)
    if abs(total - 100.0) > 0.01:
        return False
    cpu_weights = [w for w in gpu_weights if w.device == "cpu"]
    for cw in cpu_weights:
        if cw.weight > 70.0:
            return False
    return True
```

#### 4.4.2. Cálculo de `--n-gpu-layers`

`process_manager.py` — substitui hardcoded `-ngl 99`:

```python
def compute_n_gpu_layers(gpu_weights: List[GPUWeight], total_model_layers: int) -> int:
    """Calcula camadas para GPU baseado no peso total das GPUs."""
    gpu_weight_sum = sum(w.weight for w in gpu_weights if w.device == "gpu")
    gpu_ratio = gpu_weight_sum / 100.0
    return max(0, round(total_model_layers * gpu_ratio))
```

- O total de layers do modelo é obtido via `llama-server --model <path> --list-layers` ou inferido como um valor alto (999) que o llama-server clampa internamente
- Se todas as GPUs têm peso 0: `--n-gpu-layers 0` (tudo na CPU)
- Se GPU weight soma 100%: `--n-gpu-layers` = total layers (tudo na GPU)
- `--tensor-split` continua sendo GPU-only (CPU não participa de tensor split)

### 4.5. Auto Balance com CPU

`auto_balance.py` — modificações no `AutoBalanceProber.discover()`:

**Fase 1 — Find Feasible Split (modificada):**
1. Mesma lógica atual: tenta 1 GPU, depois 2, etc.
2. Após esgotar todas as GPUs, se ainda há OOM: adiciona CPU com peso = `100 - sum(gpu_weights)`
3. Peso da CPU é limitado a 70%
4. Se modelo ainda não cabe com CPU a 70%: retorna erro "Modelo além da capacidade"

```python
def _should_add_cpu(self, all_gpus: list, weight_map: dict) -> bool:
    """CPU só entra se todas as GPUs já estão ativas e ainda há OOM."""
    return len(all_gpus) > 0 and all(g in weight_map for g in all_gpus)
```

**Fase 2 — Maximize VRAM (modificada):**
1. Para cada GPU na spill order: binary search do peso máximo
2. Após otimizar GPUs: CPU recebe `100 - sum(gpu_weights)` (se > 0)
3. Se CPU weight > 70%: limita a 70%, redistribui o excedente para GPUs

**Priorização GPU:** A spill order (spills da GPU principal para as demais) garante que GPUs sempre recebem carga antes da CPU. A CPU é o último recurso.

### 4.6. Interface com llama-server

`process_manager.py` `ProcessManager.start()`:

```python
# Substitui hardcoded "-ngl", "99"
n_gpu_layers = compute_n_gpu_layers(gpu_weights, total_layers=999)
cmd = [
    LLAMA_SERVER_BIN,
    "-m", model_path,
    "-ngl", str(n_gpu_layers),  # DYNAMIC
    "--flash-attn", "on",
    "--host", "0.0.0.0",
    "--port", "8085",
    "--tools", "all",
    "--parallel", str(parallel_slots),
    "--ctx-size", str(server_ctx_size),
    "--batch-size", str(batch_size),
    "--main-gpu", main_gpu,
    "--split-mode", split_mode,
    "--tensor-split", ",".join(split),
    "--api-key", api_token,
]
```

- `--tensor-split` é calculado apenas entre GPUs ativas (lógica existente de `gpu_manager.compute_tensor_split()`)
- A CPU não recebe tensor split nem entra em `CUDA_VISIBLE_DEVICES`

## 5. Design da Interface

### 5.1. HTML — Linha da CPU na Tabela de Dispositivos

`llama_manager.py` — a função `index()` injeta uma linha de CPU após as linhas de GPU:

```python
# Detecta CPU
cpu_info = gpu_manager.detect_cpu_info()
cpu_row = f"""
<tr id="cpu-row" class="cpu-row group" data-device="cpu">
    <td>
        <div class="device-util-val">0</div>
        <div class="device-util-bar">...</div>
    </td>
    <td>
        <input type="checkbox" class="cpu-checkbox" checked>
    </td>
    <td>
        <span class="device-name">{cpu_info["name"]}</span>
    </td>
    <td>
        <div class="device-vram-text">{used_mb} / {total_mb} MB</div>
        <div class="device-vram-bar">...</div>
    </td>
    <td>
        <input type="number" class="cpu-weight" min="0" max="70" value="0">
        <input type="checkbox" class="cpu-pin">
    </td>
</tr>
"""
```

**Estrutura DOM:**
- `#cpu-row` — linha da CPU, com `data-device="cpu"`
- `.cpu-checkbox` — checkbox de ativação
- `.cpu-weight` — input numérico (0-70)
- `.cpu-pin` — checkbox de fixar peso
- `.device-util-val`, `.device-util-bar` — CPU usage %
- `.device-vram-text`, `.device-vram-bar` — RAM usada/total

**Observações visuais:**
- Mesma paleta, tipografia, animações das linhas GPU
- Sem radio button (CPU não tem "principal")
- Checkbox de ativação visível e checked por padrão
- Max weight no input HTML = 70

### 5.2. Frontend — gpu.js

#### 5.2.1. Bind de Event Listeners

`static/js/gpu.js` — função `bindGpuManualListeners()` estendida para CPU:

```javascript
function bindCpuListeners() {
    const cpuCheckbox = document.querySelector('.cpu-checkbox');
    const cpuWeight = document.querySelector('.cpu-weight');
    const cpuPin = document.querySelector('.cpu-pin');
    
    if (cpuWeight) {
        cpuWeight.addEventListener('change', () => {
            let val = parseFloat(cpuWeight.value) || 0;
            val = Math.min(70, Math.max(0, val));
            cpuWeight.value = val;
            redistributeWeights('cpu');
        });
    }
}
```

#### 5.2.2. Redistribuição de Pesos

Função `redistributeUnpinnedWeights()` modificada para incluir CPU:

```javascript
function redistributeWeights(changedDevice) {
    const gpuRows = document.querySelectorAll('.gpu-row[data-active="true"]');
    const cpuRow = document.querySelector('#cpu-row');
    const cpuEnabled = cpuRow.querySelector('.cpu-checkbox').checked;
    const cpuWeight = cpuEnabled ? (parseFloat(cpuRow.querySelector('.cpu-weight').value) || 0) : 0;
    
    // CPU é pinned por padrão
    const cpuPinned = { weight: cpuWeight };
    const gpuUnpinned = [];
    
    // Coleta GPUs unpinned
    gpuRows.forEach(row => {
        if (!row.querySelector('.gpu-pin').checked) {
            gpuUnpinned.push({ row, currentWeight: ... });
        }
    });
    
    // Distribui (100 - cpuWeight) entre GPUs unpinned
    const remaining = 100 - cpuWeight;
    // ... lógica proporcional existente
}
```

#### 5.2.3. Total de Pesos

`updateTotal()` estendida para incluir CPU no cálculo:

```javascript
function updateTotal() {
    let total = 0;
    document.querySelectorAll('.gpu-row[data-active="true"]').forEach(row => {
        total += parseFloat(row.querySelector('.gpu-weight').value) || 0;
    });
    const cpuRow = document.querySelector('#cpu-row');
    if (cpuRow && cpuRow.querySelector('.cpu-checkbox').checked) {
        total += parseFloat(cpuRow.querySelector('.cpu-weight').value) || 0;
    }
    // ... display do total
}
```

### 5.3. Frontend — models.js

`static/js/models.js` — função `startModel()` estendida para coletar peso da CPU:

```javascript
function startModel(path, elementId) {
    const gpuWeights = [];
    
    // Coleta GPUs
    document.querySelectorAll('.gpu-row[data-active="true"]').forEach(row => {
        gpuWeights.push({
            index: parseInt(row.dataset.index),
            weight: parseFloat(row.querySelector('.gpu-weight').value) || 0,
            name: row.querySelector('.gpu-name').textContent,
            active: true,
            is_main: row.querySelector('.gpu-main-radio').checked,
            pinned: row.querySelector('.gpu-pin').checked,
            device: "gpu"
        });
    });
    
    // Coleta CPU
    const cpuRow = document.querySelector('#cpu-row');
    if (cpuRow && cpuRow.querySelector('.cpu-checkbox').checked) {
        gpuWeights.push({
            index: -1,
            weight: parseFloat(cpuRow.querySelector('.cpu-weight').value) || 0,
            name: cpuRow.querySelector('.device-name').textContent,
            active: true,
            is_main: false,
            pinned: cpuRow.querySelector('.cpu-pin').checked,
            device: "cpu"
        });
    }
    
    // ... envia StartRequest com gpuWeights estendido
}
```

### 5.4. Frontend — metrics.js

`static/js/metrics.js` — função `updateMetrics()` estendida para CPU/RAM:

```javascript
function updateMetrics(data) {
    // CPU usage (linha da CPU na tabela de dispositivos)
    const cpuVal = document.querySelector('.cpu-row .device-util-val');
    const cpuBar = document.querySelector('.cpu-row .device-util-bar');
    if (cpuVal) {
        cpuVal.textContent = Math.round(data.cpu) + '%';
        cpuBar.style.width = data.cpu + '%';
    }
    
    // RAM na linha da CPU
    const ramText = document.querySelector('.cpu-row .device-vram-text');
    const ramBar = document.querySelector('.cpu-row .device-vram-bar');
    if (ramText && data.ram_used_mb !== undefined) {
        ramText.textContent = `${data.ram_used_mb} / ${data.ram_total_mb} MB`;
        ramBar.style.width = data.ram + '%';
    }
    
    // GPU metrics (inalteradas)
    // ...
}
```

## 6. API Changes

### 6.1. Endpoint `/status`

**Response atual:**
```json
{
    "running": true,
    "model": "/path/to/model.gguf",
    "gpu_weights": [{"index": 0, "weight": 70, ...}]
}
```

**Nova resposta:** `gpu_weights` inclui itens com `device: "cpu"` quando a CPU está ativada.

### 6.2. Endpoint `/metrics`

**Response atual:**
```json
{
    "cpu": 25.3,
    "ram": 45.2,
    "gpus": [...]
}
```

**Nova resposta:**
```json
{
    "cpu": 25.3,
    "ram": 45.2,
    "ram_used_mb": 14592,
    "ram_total_mb": 32768,
    "cpu_name": "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz",
    "gpus": [...]
}
```

### 6.3. Endpoint `/start`

**Request:** `gpu_weights` agora aceita objetos com `device: "cpu"`.

### 6.4. Endpoint `/set-default-config`

**Request:** Salva CPU weight no config JSON junto com GPU weights.

## 7. Integração com Auto Balance

### 7.1. Fluxo Modificado

```
AutoBalanceProber.discover()
├── Fase 1: _find_feasible_split()
│   ├── Tenta GPU0 (100%) → probe
│   ├── Tenta GPU0+GPU1 → probe
│   ├── ...
│   └── Todas as GPUs ativas + OOM? → Adiciona CPU (100 - sum_gpu_weights)
│       └── CPU > 70%? → Limita a 70%
└── Fase 2: _maximize_vram_per_gpu()
    ├── Para cada GPU na spill order: binary search max weight
    └── Recalcula CPU weight = 100 - sum(gpu_weights)
        └── CPU > 70%? → Limita, redistribui para GPUs
```

### 7.2. Constantes

| Constante | Valor | Descrição |
|-----------|-------|-----------|
| `MAX_CPU_WEIGHT` | 70 | Peso máximo da CPU em % |
| `CPU_DEVICE_INDEX` | -1 | Índice do dispositivo CPU no schema |
| `CPU_DEVICE` | "cpu" | Tipo do dispositivo CPU |

## 8. Config Storage

`config_manager.py` — o JSON de config para cada modelo passa a incluir `device` nos pesos:

```json
{
    "model_configs": {
        "/path/to/model.gguf": {
            "gpu_weights": [
                {"index": 0, "weight": 55, "name": "NVIDIA A100", "active": true, "is_main": true, "pinned": false, "device": "gpu"},
                {"index": 1, "weight": 25, "name": "NVIDIA A100", "active": true, "is_main": false, "pinned": false, "device": "gpu"},
                {"index": -1, "weight": 20, "name": "Intel Xeon E5-2680 v4", "active": true, "is_main": false, "pinned": false, "device": "cpu"}
            ]
        }
    }
}
```

## 9. Plano de Implementação

### 9.1. Backend (Python)

| Ordem | Arquivo | Mudança |
|-------|---------|---------|
| 1 | `schemas.py` | Adiciona campo `device: str = "gpu"` ao `GPUWeight` |
| 2 | `gpu_manager.py` | Adiciona `detect_cpu_info()` (retorna nome do CPU, RAM usada/total); estende `get_metrics()` com `ram_used_mb`, `ram_total_mb`, `cpu_name` |
| 3 | `auto_balance.py` | Estende `AutoBalanceProber` para considerar CPU: `_should_add_cpu()`, limita CPU a 70%, redistribui excedente |
| 4 | `process_manager.py` | Implementa `compute_n_gpu_layers()`, substitui hardcoded `-ngl 99`, adiciona validação de peso da CPU |
| 5 | `llama_manager.py` | Backend: injeta linha da CPU no HTML template, estende response de `/metrics` |

### 9.2. Frontend (JavaScript)

| Ordem | Arquivo | Mudança |
|-------|---------|---------|
| 6 | `static/js/gpu.js` | Adiciona `bindCpuListeners()`, estende `redistributeWeights()`, estende `updateTotal()` |
| 7 | `static/js/metrics.js` | Estende `updateMetrics()` para renderizar CPU usage + RAM na linha da CPU |
| 8 | `static/js/models.js` | Estende `startModel()` para coletar peso da CPU no request `/start` |

### 9.3. Testes e Validação

| Ordem | Arquivo | Mudança |
|-------|---------|---------|
| 9 | `tests/` | Adiciona testes unitários para `detect_cpu_info()`, `compute_n_gpu_layers()`, `validate_weights()` com CPU |

## 10. Sequência de Build

1. **schemas.py** — campo `device` (sem dependências externas)
2. **gpu_manager.py** — `detect_cpu_info()` e extensão de `get_metrics()` (depende: 1)
3. **auto_balance.py** — integração CPU no prober (depende: 1, 2)
4. **process_manager.py** — `compute_n_gpu_layers()` e validação (depende: 1, 3)
5. **llama_manager.py** — HTML da linha CPU + response `/metrics` (depende: 2)
6. **gpu.js** — bindings e redistribuição (depende: 5, backend stable)
7. **metrics.js** — renderização de CPU/RAM na tabela (depende: 5)
8. **models.js** — coleta de peso da CPU no start (depende: 6, 7)
9. **tests/** — cobertura unitária (depende: 1-8)

## 11. Registros de Decisão de Arquitetura

### ADRs Criados

| ADR | Título | Resumo |
|-----|--------|---------|
| [ADR-001](adrs/adr-001.md) | CPU como Dispositivo Unificado na Tabela de Recursos | CPU aparece na mesma tabela das GPUs, com mesmo padrão visual e controles |
| [ADR-002](adrs/adr-002.md) | Extensão do Schema GPUWeight com Campo Device | Extensão do schema existente vs. schema separado para CPU |
| [ADR-003](adrs/adr-003.md) | Cálculo Dinâmico de --n-gpu-layers | Substituição do hardcoded `99` por cálculo baseado em pesos |
| [ADR-004](adrs/adr-004.md) | Priorização GPU no Auto Balance com Limite de 70% | GPU sempre priorizada, CPU limitada a 70% máximo |

## 12. Testes

### 12.1. Testes Unitários (Backend)

| Teste | Cobertura |
|-------|-----------|
| `detect_cpu_info()` — nome do processador retornado corretamente | gpu_manager.py |
| `detect_cpu_info()` — fallback para "CPU Host" quando detecção falha | gpu_manager.py |
| `compute_n_gpu_layers()` — 100% GPU = 999 layers, 0% = 0, 50% = 499 | process_manager.py |
| `validate_weights()` — soma = 100%, CPU <= 70% | process_manager.py |
| `validate_weights()` — rejeita soma != 100% | process_manager.py |
| `validate_weights()` — rejeita CPU > 70% | process_manager.py |
| Auto Balance — todas GPUs cabem, CPU = 0% | auto_balance.py |
| Auto Balance — VRAM insuficiente, CPU recebe carga | auto_balance.py |
| Auto Balance — CPU limit a 70%, redistribui excedente | auto_balance.py |
| Schema — `GPUWeight` com `device="cpu"` serializa corretamente | schemas.py |

### 12.2. Testes de Integração

| Teste | Cobertura |
|-------|-----------|
| Linha da CPU aparece no HTML do dashboard | llama_manager.py |
| `/metrics` retorna `ram_used_mb`, `ram_total_mb`, `cpu_name` | llama_manager.py |
| Peso da CPU coletado corretamente no `startModel()` | models.js |
| Redistribuição inclui CPU na soma de 100% | gpu.js |
| Inicia llama-server com `--n-gpu-layers` calculado dinamicamente | process_manager.py |
| OOM recalcula pesos incluindo CPU | process_manager.py |

### 12.3. Testes E2E

| Teste | Cobertura |
|-------|-----------|
| Ativar/desativar CPU na UI e iniciar modelo | Fluxo completo |
| Ajustar peso da CPU (0-70) e validar soma = 100% | Fluxo completo |
| Pin weight da CPU e executar Auto Balance | Fluxo completo |
| Modelo > VRAM: Auto Balance ativa CPU automaticamente | Fluxo completo |
| Métricas de CPU usage e RAM atualizam em tempo real | Fluxo completo |

## 13. Riscos Técnicos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| `platform.processor()` retorna string vazia ou genérica em algumas builds do Python | Médio | Fallback para `platform.machine()` e depois "CPU Host" |
| llama-server retorna erro se `--n-gpu-layers` é muito alto para o VRAM | Alto | Auto Balance e OOM Watchdog já tratam OOM e recalculam |
| Auto Balance leva muito tempo com CPU + GPUs combinados | Médio | Mesmo timeout de 180s por probe; CPU só é adicionada após GPUs esgotadas |
| Frontend JS não encontra elementos da CPU se HTML não foi injetado | Baixo | Todos os seletores JS verificam `null` antes de acessar propriedades |
| Config antigo (sem campo `device`) quebra ao ler pesos salvos | Baixo | Campo tem default `"gpu"`; leitura usa `.get("device", "gpu")` |

## 14. Fora do Escopo

- Detecção de número de threads físicos/lógicos da CPU (apenas nome e RAM)
- Temperatura da CPU (apenas usage %)
- Limitação de RAM utilizável
- Suporte a swap disk
- Suporte a NPUs ou outros aceleradores
- Interface web de configuração de parâmetros avançados da CPU
- Múltiplas CPUs físicas (o sistema usa a CPU detectada como um único dispositivo)
