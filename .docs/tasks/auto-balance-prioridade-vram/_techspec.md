# TechSpec: Maximização de VRAM por Prioridade no Auto-Balance

## Resumo Executivo

Consolidamos toda a decisão de alocação em uma **cascata estrita por prioridade, calculada analiticamente**, residente em `LoadDistributor.distribute` (`load_distributor.py`) como fonte única da verdade. Para cada GPU em ordem de prioridade (principal → demais por índice → CPU), a função aloca `min(VRAM_total × 0,98, restante_do_modelo)` em MB e só usa a próxima GPU quando a atual atinge 98%; a CPU/RAM recebe carga apenas após todas as GPUs habilitadas. Os MB por dispositivo são convertidos em percentual, **preservando o contrato `DistributionResult`** e todo o mapeamento atual para `--tensor-split`/`--ngl`/`--main-gpu`. O probing empírico por OOM (`AutoBalanceProber`) é aposentado do caminho de decisão.

**Trade-off técnico principal:** trocamos a robustez empírica do probing por OOM por determinismo e simplicidade. A margem de 98% e o overhead já embutido em `estimate_model_vram_mb` cobrem o risco de OOM; casos extremos de subestimação não terão recuo automático (rede de segurança adiada — ADR-002).

## Arquitetura do Sistema

### Visão dos Componentes

- **`LoadDistributor` (load_distributor.py)** — *motor único*. Recebe VRAM por GPU, ordem de prioridade, MB estimados do modelo e a válvula `cpu_enabled`; devolve `DistributionResult` (pesos % por GPU, % CPU, viabilidade). Contém a nova cascata por MB.
- **`GPUManager.compute_offload_plan` (gpu_manager.py)** — adaptador. Monta os dicionários de VRAM/ordem, chama `LoadDistributor` e converte `DistributionResult` em `OffloadPlan` (`n_gpu_layers`, `tensor_split`, etc.). Perde o caminho proporcional/legado.
- **`AutoBalancePlanner.estimate_model_vram_mb` (auto_balance.py)** — reutilizado para a estimativa de VRAM (disco + overhead + KV-cache). `AutoBalanceProber` e os métodos de probing/escala proporcional são aposentados.
- **`ProcessManager.start` (process_manager.py)** — inalterado no mapeamento de args; passa a ignorar `pinned` quando `auto_balance=true`.
- **Frontend (`static/js/gpu.js`, `models.js`)** — limpa os checkboxes `.gpu-pin` ao ligar o `auto-balance-toggle`; mantém o alerta de capacidade (`showAutoBalanceCapacityAlert`) para `is_feasible=false`.

**Fluxo:** `models.js (/start)` → `ProcessManager.start` → `GPUManager.compute_offload_plan` → `LoadDistributor.distribute` (cascata) → `OffloadPlan` → args do `llama-server`.

## Design de Implementação

### Interfaces Principais

A cascata reescrita, preservando a assinatura e o tipo de retorno atuais:

```python
@dataclass(frozen=True)
class DistributionResult:
    gpu_weights: Dict[int, int]   # índice GPU -> peso %
    cpu_weight: int               # peso % da CPU (0 quando cabe nas GPUs)
    total_gpu_pct: int            # soma dos pesos de GPU
    is_feasible: bool             # False: não cabe e cpu_enabled=False

class LoadDistributor:
    @staticmethod
    def distribute(
        gpu_vram: Dict[int, int],          # índice -> VRAM total (MB)
        priority_order: List[int],         # principal primeiro, depois por índice
        estimated_model_vram_mb: int,
        cpu_enabled: bool = True,
        vram_limit_pct: float = 98.0,      # constante fixa (ADR-001)
    ) -> "DistributionResult": ...
```

Algoritmo (pseudo-código, MB):

```python
restante = estimated_model_vram_mb
mb_por_gpu = {}
for idx in priority_order:                 # ordem de prioridade
    util = int(gpu_vram[idx] * vram_limit_pct / 100)
    aloc = min(util, restante)
    mb_por_gpu[idx] = aloc
    restante -= aloc
    if restante <= 0:
        break
if restante > 0 and not cpu_enabled:
    return DistributionResult({}, 0, 0, is_feasible=False)
cpu_mb = max(0, restante)
# converte MB -> %, reconciliando a soma em 100 no último dispositivo
```

> Observação: a assinatura troca `gpu_weights`/`total_layers` por `priority_order`/`vram_limit_pct`. O `compute_offload_plan` deriva `priority_order` de `is_main` + índice e da lista de GPUs ativas; `total_layers` continua sendo usado fora da cascata, na conversão %→`n_gpu_layers`.

### Modelos de Dados

Sem novas entidades persistidas. Schemas reutilizados (`schemas.py`):
- `GPUWeight { index, weight, name, active, is_main, pinned, device }`
- `StartRequest { ..., gpu_weights, auto_balance, cpu_enabled, total_layers }`

Config por modelo (`config_manager.py`) inalterada: `gpu_weights`, `auto_balance`, `hardware_incapable`, `hardware_incapable_message`.

### Endpoints de API

`POST /start` — superfície inalterada. Mudança de comportamento:
- Quando `auto_balance=true`: o backend ignora `pinned` em cada `GPUWeight` e calcula a distribuição via cascata.
- Quando `is_feasible=false` (modelo não cabe e `cpu_enabled=false`): não inicia; retorna o payload de capacidade excedida que o frontend já renderiza, e persiste `hardware_incapable=true` + mensagem.

## Análise de Impacto

| Componente | Tipo de Impacto | Descrição e Risco | Ação Necessária |
|------------|-----------------|-------------------|-----------------|
| `load_distributor.py` | modificado | Reescrita de `distribute` para cascata por MB. Risco médio (núcleo). | Implementar cascata + conversão MB→% + `is_feasible`. |
| `gpu_manager.py` | modificado | `compute_offload_plan` sempre delega à cascata; remove caminho proporcional/legado e `_compute_offload_plan_with_lu`. Risco médio. | Refatorar; derivar `priority_order`. |
| `auto_balance.py` | depreciado (parcial) | Aposentar `AutoBalanceProber` (probing/escala) e `_split_pool_by_vram`/`compute_cpu_offload_weights`. Mantém `estimate_model_vram_mb`. Risco médio. | Remover lógica proporcional/probing; preservar estimativa. |
| `process_manager.py` | modificado | Ignorar `pinned` quando `auto_balance=true`. Risco baixo. | Filtrar pins antes do cálculo. |
| `static/js/gpu.js` / `models.js` | modificado | Limpar `.gpu-pin` ao ligar o toggle. Risco baixo. | Handler no `auto-balance-toggle`. |
| `tests/unit/test_auto_balance.py` | modificado | Testes do probing/proporcional ficam obsoletos. Risco baixo. | Remover/atualizar casos. |
| `tests/unit/test_load_distributor_cascade.py` | novo | Cenários A–D e bordas. | Criar arquivo. |

## Abordagem de Testes

### Testes Unitários

Novo arquivo **`tests/unit/test_load_distributor_cascade.py`**:
- **Cenários A–D parametrizados** (VRAM 3090=24GB→23,5 útil; P100=16GB→15,7 útil), validando MB/% por dispositivo:
  - A: modelo 20GB → 3090=100%, P100s=0, CPU=0.
  - B: 30GB → 3090 cheia, P100#1 com a sobra, P100#2=0, CPU=0.
  - C: 50GB → 3090 + P100#1 cheias, P100#2 com sobra, CPU=0.
  - D: 70GB → todas as GPUs a 98%, CPU com a sobra.
- **Bordas:** modelo cabe 100% na principal (sem ativar secundárias); `cpu_enabled=false` e não cabe → `is_feasible=false`; GPU única; soma de % reconciliando em 100; arredondamento com VRAM desigual.
- **Regra zero-offload:** nenhum `cpu_weight>0` enquanto qualquer GPU ainda tem folga abaixo do limite.

Mocks: nenhum I/O; função pura. Estimativa de VRAM testada com valores diretos.

### Testes de Integração

Ajustar `tests/unit/test_gpu_manager_core.py` para confirmar que `compute_offload_plan` delega à cascata e produz `tensor_split`/`n_gpu_layers` coerentes nos cenários A–D. (Integração ampla no `/start` fica fora do MVP — ver Fora de Escopo do PRD.)

## Sequenciamento de Desenvolvimento

### Ordem de Construção

1. **Cascata em `LoadDistributor.distribute`** — sem dependências. Implementa o algoritmo MB + conversão %→ e `is_feasible`.
2. **Testes unitários da cascata** (`test_load_distributor_cascade.py`) — depende do passo 1; trava os cenários A–D.
3. **Adaptação de `compute_offload_plan`** — depende do passo 1; deriva `priority_order`, remove caminho proporcional/legado.
4. **Aposentadoria do probing/proporcional em `auto_balance.py`** — depende dos passos 1 e 3; preserva `estimate_model_vram_mb`.
5. **Ignorar `pinned` no backend quando `auto_balance=true`** — depende do passo 3.
6. **Frontend: limpar pins no toggle** — depende do passo 5 (alinhamento de contrato).
7. **Limpeza de testes obsoletos** — depende dos passos 3 e 4.

### Dependências Técnicas

- Nenhuma dependência externa nova. Reaproveita detecção de GPU/VRAM e estimativa de modelo já existentes.

## Monitoramento e Observabilidade

- Log estruturado no cálculo da cascata: por dispositivo, `vram_total_mb`, `vram_limit_mb`, `alocado_mb`, `peso_pct`; e `model_vram_mb`, `is_feasible`.
- Log de decisão de bloqueio (`hardware_incapable`) com o motivo.
- Métrica observável no painel: ocupação por GPU (já existente) confirma "98% antes de CPU".

## Considerações Técnicas

### Decisões-Chave

- **Decisão:** cascata analítica por MB em `LoadDistributor`. **Justificativa:** determinismo e testabilidade (cenários A–D). **Trade-off:** sem recuo automático por OOM. **Alternativas rejeitadas:** probing empírico (lento/não-determinístico), módulo novo (YAGNI), centralizar em `auto_balance.py` (acoplado ao probing).
- **Decisão:** contrato em percentual (converter MB→%). **Justificativa:** menor impacto a jusante. **Trade-off:** arredondamento. **Alternativa rejeitada:** estender contrato com MB.
- **Decisão:** ignorar `pinned` no backend quando `auto_balance=true` + limpar no frontend. **Justificativa:** previsibilidade e robustez via API.

### Riscos Conhecidos

- **Subestimação de VRAM → OOM** (probabilidade baixa/média): mitigado por margem de 98% + overhead na estimativa; reintroduzir rede de segurança em fase futura se recorrente.
- **Arredondamento MB→%** (baixa): reconciliação no último dispositivo + tolerância existente.
- **Regressão ao remover probing** (média): cobertura via novos testes e ajuste dos testes legados antes de remover código.

## Registros de Decisão de Arquitetura

- [ADR-001: Cascata estrita por prioridade como contrato único](adrs/adr-001.md) — Regra determinística de preenchimento por prioridade, 98%, bloqueio com CPU off, Auto-Balance limpa pins.
- [ADR-002: Cálculo analítico determinístico da cascata](adrs/adr-002.md) — Aposentar o probing empírico por OOM em favor de cálculo fechado.
- [ADR-003: LoadDistributor como motor único de cascata por MB](adrs/adr-003.md) — Reescrever `distribute`, remover trechos proporcionais, manter contrato em %.
