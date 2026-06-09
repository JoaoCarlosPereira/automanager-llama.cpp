# TechSpec: Correção do Offload Automático por CPU e Auto-Balance

**Data:** 2026-06-08
**Tarefa:** correcao-offload-cpu-auto-balance
**Status:** Rascunho
**PRD:** `[_prd.md](._prd.md)`

---

## 1. Resumo Executivo

Corrigir as inconsistências na distribuição de carga GPU/CPU do AutoManager, unificando o motor de distribuição em um único módulo stateless `LoadDistributor`. O checkbox de CPU passa a funcionar como válvula on/off real, a política GPU-first/CPU-mínimo é aplicada consistentemente nos modos manual e automático, e o limite rígido de 70% no CPU é removido.

**Trade-off principal:** Unificar em vez de corrigir isoladamente — aceita o custo de um novo módulo (`load_distributor.py`) e migração dos consumers (`gpu_manager.py`, `auto_balance.py`, `process_manager.py`, `gpu.js`) em troca de fonte única de verdade, testabilidade e manutenibilidade.

## 2. Arquitetura Atual (As-Is)

### Diagrama de Componentes

```
                    ┌─────────────────┐
                    │  llama_manager.py │  (FastAPI endpoints)
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ process_mgr  │ │ auto_balance │ │ gpu_manager  │
     │              │ │              │ │              │
     │ start()      │ │ discover()   │ │ compute_off- │
     │   ├─ validate│ │   ├─ Phase 1 │ │   load_plan()│
     │   ├─ plan    │ │   └─ Phase 2 │ │   ├─ GPU→ngl │
     │   └─ launch  │ │              │ │   └─ split   │
     └──────────────┘ └──────────────┘ └──────────────┘
```

### Problemas Identificados

| Arquivo | Problema | Linha(s) |
|---------|----------|----------|
| `auto_balance.py` | `MAX_CPU_WEIGHT_PCT = 70` — cap rígido | 33 |
| `auto_balance.py` | `_escalate_cpu_offload()` escala CPU em 10% até 70% | 552-575 |
| `auto_balance.py` | `_finalize_cpu_split()` aplica cap de 70% | 605-611 |
| `auto_balance.py` | `compute_cpu_offload_weights()` aplica cap de 70% | 359, 362 |
| `gpu_manager.py` | `validate_weights()` rejeita CPU > 70% | 575-580 |
| `gpu.js` | `validateDeviceWeights()` rejeita CPU > 70% | 293-299 |
| `gpu_manager.py` | `compute_offload_plan()` trata CPU como peso normal, sem política GPU-first | 415-450 |

## 3. Arquitetura Alvo (To-Be)

### Diagrama de Componentes

```
                    ┌─────────────────┐
                    │  llama_manager.py │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ process_mgr  │ │ auto_balance │ │ gpu_manager  │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │  load_distributor  │  ← NOVO
                  │  LoadDistributor   │
                  │  (stateless)       │
                  └────────────────────┘
```

### Fluxo de Dados

```
[UI] → checkbox CPU + pesos GPU → [StartRequest]
                                      │
                                      ▼
                            ┌─────────────────┐
                            │  LoadDistributor │
                            │  .distribute()   │
                            └────────┬────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
       [Modo Manual]         [Auto-Balance]         [compute_offload_plan]
         process_manager          prober.discover        gpu_manager
           .start()                  .discover()          .compute_offload_plan
              │                        │                       │
              └────────────────────────┼───────────────────────┘
                                       │
                                       ▼
                              OffloadPlan (ngl, cpu_layers,
                              gpu_pct, cpu_pct, tensor_split)
                                       │
                                       ▼
                              llama-server --ngl N --tensor-split X
```

## 4. Design Detalhado

### 4.1 Novo Módulo: `load_distributor.py`

**Caminho:** `d:\dsv-git\automanager-llama.cpp\load_distributor.py`

```python
"""Unified GPU/CPU load distribution engine — GPU-first, CPU-minimum policy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DistributionResult:
    """Result of a GPU/CPU load distribution calculation."""
    gpu_weights: Dict[int, int]       # {gpu_index: weight_pct}
    cpu_weight: int                   # 0-100
    total_gpu_pct: int                # sum(gpu_weights.values())
    is_feasible: bool                 # False if model exceeds all hardware


class LoadDistributor:
    """
    Stateless engine that calculates the optimal distribution of model layers
    across GPUs and CPU following the GPU-first, CPU-minimum policy.

    The CPU checkbox acts as an on/off valve:
      - OFF (cpu_enabled=False)  → CPU_weight = 0, model must fit in GPUs
      - ON  (cpu_enabled=True)   → CPU absorbs spill-over only
    """

    @staticmethod
    def distribute(
        gpu_vram: Dict[int, int],        # {index: vram_mb}
        gpu_weights: Dict[int, int],     # {index: user_weight_pct} (manual mode)
        total_layers: int,
        estimated_model_vram_mb: int,
        cpu_enabled: bool = True,
    ) -> DistributionResult:
        """
        Calculate the GPU/CPU distribution.

        Policy:
        1. If cpu_enabled=False: distribute 100% across GPUs, CPU = 0.
        2. If total_gpu_vram >= model_vram: distribute across GPUs, CPU = 0.
        3. If total_gpu_vram < model_vram AND cpu_enabled:
           GPU receives proportional share of model_vram, CPU = remainder.
        4. If total_gpu_vram < model_vram AND NOT cpu_enabled:
           is_feasible = False, GPU distributes anyway for error reporting.
        """
        ...

    @staticmethod
    def is_feasible(
        gpu_vram: Dict[int, int],
        estimated_model_vram_mb: int,
        cpu_enabled: bool = True,
    ) -> bool:
        """Check if the model can fit in the given hardware."""
        ...

    @staticmethod
    def compute_n_gpu_layers(
        total_layers: int,
        gpu_weight_pct: float,
    ) -> int:
        """Convert GPU weight percentage to --ngl value."""
        return max(0, min(total_layers, int(round(gpu_weight_pct / 100.0 * total_layers))))
```

#### Regras de Negócio Implementadas

| Regra | Implementação |
|-------|---------------|
| GPU-first, CPU mínimo | `distribute()` verifica `total_gpu_vram >= model_vram` antes de atribuir a CPU |
| CPU = válvula on/off | Parâmetro `cpu_enabled`; se `False`, `cpu_weight` sempre 0 |
| Sem cap de 70% | Nenhuma constante `MAX_CPU_WEIGHT_PCT` no novo módulo |
| Manual respeita pesos | `gpu_weights` é passado como input e preservado (escalonado proporcionalmente) |
| Auto-balance agressivo | CPU só entra após todas as GPUs no máximo (Phase 2 já faz isso) |

### 4.2 Atualização: `gpu_manager.py`

#### 4.2.1 `compute_offload_plan()` — Linha 415

**Mudança:** Delegar o cálculo de distribuição ao `LoadDistributor`.

```python
# ANTES (simplificado):
def compute_offload_plan(self, gpu_weights, total_layers=32):
    gpu_pct = self.sum_active_weight(gpu_weights, "gpu")
    cpu_pct = self.sum_active_weight(gpu_weights, "cpu")
    if self.cpu_offload_active(gpu_weights):
        n_gpu_layers = int(round(gpu_pct / 100.0 * total_layers))
    ...

# DEPOIS:
def compute_offload_plan(self, gpu_weights, total_layers=32, cpu_enabled=None):
    # cpu_enabled=None → inferir de gpu_weights
    # cpu_enabled=True/False → forçar estado do checkbox
    result = LoadDistributor.distribute(
        gpu_vram={g.index: g.vram for g in self._cached_gpus},
        gpu_weights={w.index: int(w.weight) for w in gpu_weights if w.device == "gpu"},
        total_layers=total_layers,
        estimated_model_vram_mb=self._cached_model_vram_mb or 0,
        cpu_enabled=cpu_enabled if cpu_enabled is not None else self.cpu_offload_active(gpu_weights),
    )
    ...
```

**Integração com `llama_manager.py` start endpoint (linha 226-304):**
O endpoint `start_model` deve passar `cpu_enabled` baseado no estado do checkbox de CPU do `StartRequest`. Adicionar campo `cpu_enabled: bool` ao `StartRequest` (schemas.py).

#### 4.2.2 `validate_weights()` — Linha 539

**Mudança:** Remover verificação de cap de 70%.

```python
# ANTES (linha 575):
if cpu_weight > 70.0:
    return False, "O peso da CPU excede o limite máximo de 70%..."

# DEPOIS:
# Remover completamente esta verificação
```

### 4.3 Atualização: `auto_balance.py`

#### 4.3.1 Remoção de `MAX_CPU_WEIGHT_PCT` — Linha 33

```python
# REMOVER:
MAX_CPU_WEIGHT_PCT = 70  # Hard cap: CPU weight cannot exceed 70%
CPU_OFFLOAD_STEP = 10
```

#### 4.3.2 `_escalate_cpu_offload()` — Linha 552

**Mudança:** Escalonar CPU sem limite superior (apenas até 100%).

```python
# ANTES:
def _escalate_cpu_offload(self, ...):
    new_cpu = min(cpu_weight + CPU_OFFLOAD_STEP, MAX_CPU_WEIGHT_PCT)

# DEPOIS:
def _escalate_cpu_offload(self, ...):
    new_cpu = min(cpu_weight + CPU_OFFLOAD_STEP, 100 - min_gpu_budget)
    # min_gpu_budget = 10 (mínimo para tentar uma GPU)
```

#### 4.3.3 `_finalize_cpu_split()` — Linha 588

**Mudança:** Remover cap de 70%.

```python
# ANTES (linha 605):
if raw_cpu <= MAX_CPU_WEIGHT_PCT:
    return gpu_map, raw_cpu
cpu_weight = MAX_CPU_WEIGHT_PCT

# DEPOIS:
return gpu_map, raw_cpu  # Sem cap
```

#### 4.3.4 `compute_cpu_offload_weights()` — Linha 330

**Mudança:** Delegar ao `LoadDistributor` ou remover cap.

```python
# ANTES (linha 359):
gpu_fraction = min(gpu_fraction, MAX_CPU_WEIGHT_PCT / 100.0)
cpu_weight = min(cpu_weight, MAX_CPU_WEIGHT_PCT)

# DEPOIS: Remover ambas as linhas de cap
gpu_fraction = gpu_fraction  # Sem cap
cpu_weight = 100 - int(round(gpu_fraction * 100))  # Sem cap
```

#### 4.3.5 `_cpu_config_from_request()` — Linha 491

**Mudança:** O checkbox de CPU no `StartRequest` controla se CPU é permitido, não um peso fixo.

```python
@staticmethod
def _cpu_config_from_request(request) -> Dict[str, Any]:
    """CPU checkbox = valve; use LoadDistributor for spill-over calc."""
    cpu_w = next((w for w in request.gpu_weights if w.device == "cpu"), None)
    if not cpu_w or not cpu_w.active:
        return {"enabled": False, "pinned": False, "weight": 0}
    # Apenas habilita; peso é calculado pelo LoadDistributor
    return {"enabled": True, "pinned": False, "weight": 0}
```

### 4.4 Atualização: `process_manager.py`

#### 4.4.1 `start()` — Linha 364

**Mudança:** Passar `cpu_enabled` para `compute_offload_plan`.

```python
# ANTES:
plan = self.gpu_manager.compute_offload_plan(gpu_weights, total_layers)

# DEPOIS:
cpu_enabled = any(w.active and w.device == "cpu" for w in gpu_weights)
plan = self.gpu_manager.compute_offload_plan(
    gpu_weights, total_layers, cpu_enabled=cpu_enabled
)
```

#### 4.4.2 `validate_weights()` call — Linha 386-390

**Mudança:** Remover a chamada a `validate_weights()` (que contém o cap de 70%).

```python
# ANTES:
if has_active_cpu:
    ok, err = self.gpu_manager.validate_weights(gpu_weights)
else:
    ok, err = self.gpu_manager.validate_gpu_weights(gpu_weights)

# DEPOIS:
ok, err = self.gpu_manager.validate_gpu_weights(gpu_weights)
# validate_weights cap removido; validate_gpu_weights mantém a validação
# de soma dos pesos das GPUs (≈100%)
```

### 4.5 Atualização: `schemas.py`

**Mudança:** Adicionar campo `cpu_enabled` ao `StartRequest`.

```python
class StartRequest(BaseModel):
    ...
    cpu_enabled: bool = False  # Checkbox de CPU — valve on/off
    ...
```

### 4.6 Atualização: `static/js/gpu.js`

#### 4.6.1 `validateDeviceWeights()` — Linha 275

**Mudança:** Remover verificação de cap de 70% no frontend.

```javascript
// REMOVER (linhas 290-299):
const cpuWeight = active
    .filter(w => w.device === 'cpu')
    .reduce((sum, w) => sum + (w.weight || 0), 0);
if (cpuWeight > 70) {
    return {
        ok: false,
        message: `O peso da CPU (${cpuWeight}%) excede o limite máximo de 70%...`,
    };
}
```

#### 4.6.2 `collectDeviceWeightsFromUI()` — Linha 236

**Mudança:** O checkbox de CPU (`cpu-checkbox`) já define `active`; o peso da CPU é calculado automaticamente pelo `LoadDistributor`, não pelo usuário. O campo `cpu-weight` permanece para display, mas o backend ignora o valor.

```javascript
// MANTER a coleta atual; o backend usa cpu_enabled (checkbox) e calcula peso
```

## 5. Interação entre Módulos

### 5.1 Fluxo: Modo Manual

```
1. User: seleciona GPUs com pesos → marca/desmarca checkbox CPU → START
2. frontend: collectDeviceWeightsFromUI() → StartRequest(gpu_weights, cpu_enabled=checkbox)
3. llama_manager.start_model(): recebe StartRequest
4. process_manager.start():
   a. gpu_manager.validate_gpu_weights() — valida soma ≈100% entre GPUs
   b. gpu_manager.compute_offload_plan(gpu_weights, total_layers, cpu_enabled)
      → LoadDistributor.distribute(...)
   c. OffloadPlan(n_gpu_layers, n_cpu_layers, gpu_pct, cpu_pct, tensor_split)
   d. llama-server --ngl N --tensor-split X
```

### 5.2 Fluxo: Modo Auto-Balance

```
1. User: seleciona GPUs → marca checkbox Auto-Balance → START
2. llama_manager.start_model(): req.auto_balance=True
3. process_manager.start_auto_balance():
   a. Thread → _run_auto_balance()
   b. AutoBalanceProber.discover()
   c. Phase 1: _find_feasible_split() — tenta progressivamente mais GPUs
   d. Phase 2: _maximize_vram_per_gpu() — binary search de peso máximo
   e. Se OOM em TODAS as GPUs: _escalate_cpu_offload() (sem cap 70%)
   f. _finalize_cpu_split() (sem cap 70%)
   g. Resultado → gpu_weights → process_manager.start()
      → gpu_manager.compute_offload_plan() → llama-server
```

## 6. Testes

### 6.1 Testes Unitários: `tests/test_load_distributor.py`

| Caso de Teste | Input | Esperado |
|---------------|-------|----------|
| GPU-only, modelo cabe | vram=12GB, model=8GB, cpu_enabled=False | gpu={0:100}, cpu=0, feasible=True |
| CPU valve OFF, modelo não cabe | vram=4GB, model=8GB, cpu_enabled=False | gpu={0:100}, cpu=0, feasible=False |
| CPU valve ON, spill-over | vram=4GB, model=8GB, cpu_enabled=True | gpu={0:50}, cpu=50, feasible=True |
| Multi-GPU, CPU valve ON | vram={0:4GB, 1:4GB}, model=10GB, cpu_enabled=True | gpu={0:50, 1:50}, cpu=20, feasible=True |
| Manual pesos preservados | weights={0:70, 1:30}, vram=12GB, model=8GB | gpu={0:70, 1:30}, cpu=0, feasible=True |
| n_gpu_layers calculation | total_layers=32, gpu_pct=62.5 | n_gpu_layers=20 |

### 6.2 Testes de Integração: `tests/test_offload_integration.py`

| Cenário | Descrição |
|---------|-----------|
| Manual, CPU desligado, modelo cabe | Valida que layers vão apenas para GPUs |
| Manual, CPU desligado, modelo não cabe | Valida erro claro de hardware incapaz |
| Manual, CPU ligado, spill-over | Valida que CPU recebe o que sobra |
| Auto-balance, GPUs suficientes | Valida que CPU=0 |
| Auto-balance, OOM em todas, CPU usado | Valida CPU escalona sem cap 70% |
| Auto-balance, CPU desligado, OOM | Valida falha com mensagem clara |

## 7. Plano de Implementação

### Fase 1: Módulo Unificado e Remoção de Caps (MVP)

**Ordem de Build:**

1. **Criar `load_distributor.py`** — Módulo stateless com `LoadDistributor.distribute()`.
   - Arquivo novo: `d:\dsv-git\automanager-llama.cpp\load_distributor.py`
   - Sem dependências externas (apenas dataclasses, typing)

2. **Atualizar `schemas.py`** — Adicionar `cpu_enabled: bool = False` ao `StartRequest`.
   - Linha ~33

3. **Atualizar `gpu_manager.py`**
   - `compute_offload_plan()`: Delegar cálculo ao `LoadDistributor` (linha 415)
   - `validate_weights()`: Remover verificação de cap de 70% (linha 575)
   - Dependência: depende do passo 1

4. **Atualizar `auto_balance.py`**
   - Remover `MAX_CPU_WEIGHT_PCT = 70` (linha 33)
   - Remover `CPU_OFFLOAD_STEP = 10` (linha 34)
   - `_escalate_cpu_offload()`: Remover cap de 70% (linha 562)
   - `_finalize_cpu_split()`: Remover cap de 70% (linha 605-611)
   - `compute_cpu_offload_weights()`: Remover caps (linha 359, 362)
   - `_cpu_config_from_request()`: Simplificar para boolean valve (linha 491)
   - Dependência: depende do passo 1

5. **Atualizar `process_manager.py`**
   - `start()`: Passar `cpu_enabled` para `compute_offload_plan` (linha 402)
   - Remover chamada a `validate_weights()` (linha 386)
   - Dependência: depende dos passos 3 e 4

6. **Atualizar `static/js/gpu.js`**
   - `validateDeviceWeights()`: Remover verificação de cap de 70% (linha 293)
   - Dependência: independe (frontend)

7. **Criar testes unitários**: `tests/test_load_distributor.py`
   - Dependência: depende do passo 1

### Fase 2: Mensagem de Erro e UI

**Ordem de Build:**

8. **Atualizar `auto_balance.py` build_hardware_capacity_failure()**
   - Adicionar informação sobre checkbox de CPU na mensagem de erro
   - Dependência: depende do passo 4

9. **Atualizar `gpu.js` showAutoBalanceCapacityAlert()**
   - Exibir peso de CPU de forma destacada quando > 0%
   - Adicionar tooltip ao checkbox de CPU
   - Dependência: independe (frontend)

10. **Testes de integração**
    - Dependência: depende de todos os passos anteriores

## 8. Arquivos Modificados

| Arquivo | Ação | Linhas Afetadas |
|---------|------|-----------------|
| `load_distributor.py` | **NOVO** | — |
| `schemas.py` | Modificar | ~33 |
| `gpu_manager.py` | Modificar | 415-450, 575-580 |
| `auto_balance.py` | Modificar | 33-34, 359-362, 491-507, 552-575, 588-611 |
| `process_manager.py` | Modificar | 386-390, 402 |
| `static/js/gpu.js` | Modificar | 290-299 |
| `tests/test_load_distributor.py` | **NOVO** | — |
| `tests/test_offload_integration.py` | **NOVO** | — |

## 9. Registros de Decisão de Arquitetura

| ADR | Título | Resumo |
|-----|--------|--------|
| [ADR-001](./adrs/adr-001.md) | Unificação do Motor de Distribuição de Carga (GPU/CPU) | Criar módulo `LoadDistributor` stateless como fonte única de verdade para distribuição GPU/CPU |
