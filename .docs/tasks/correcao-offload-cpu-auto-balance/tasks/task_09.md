---
status: pending
title: Atualizar frontend - exibir peso de CPU e tooltip do checkbox
type: frontend
complexity: low
dependencies: []
---

# Task 09: Atualizar frontend - exibir peso de CPU e tooltip do checkbox

## Visao Geral

Atualizar a interface do usuario para exibir de forma clara o peso de CPU no resultado do auto-balance e adicionar um tooltip ao checkbox de CPU explicando seu comportamento como válvula on/off. Tambem deve destacar quando o CPU esta sendo usado (peso > 0%).

<critical>
- Ler o PRD secao "Experiencia do Usuario" e TechSpec secoes 5.1/5.2
- Nao adicionar novos campos, botoes ou layouts (fora de escopo do PRD)
- Apenas exibir informacoes existentes - nao alterar logica de backend
- Manter consistencia visual com o design existente (Tailwind CSS)
</critical>

<requirements>
1. No resultado do auto-balance, o peso de CPU DEVE ser exibido de forma destacada quando > 0%
2. Ao desmarcar o checkbox de CPU, DEVE exibir um tooltip: "CPU desativado - se o modelo nao couber nas GPUs, o carregamento falhará"
3. O badge de status do auto-balance DEVE indicar quando o CPU foi usado
4. A mensagem de erro de hardware incapaz DEVE exibir `cpu_checkbox_hint` de `build_hardware_capacity_failure()`
5. Nenhuma nova configuracao, botao ou layout DEVE ser adicionado (fora de escopo)
</requirements>

## Subtarefas

- [ ] Adicionar tooltip ao checkbox de CPU (`cpu-checkbox`) com texto explicativo
- [ ] Atualizar `showAutoBalanceCapacityAlert()` para exibir `cpu_checkbox_hint` quando presente
- [ ] Adicionar destaque visual ao peso de CPU no resultado do auto-balance (ex.: badge, cor diferente)
- [ ] Adicionar tooltip ao checkbox de CPU: "CPU desativado - se o modelo nao couber nas GPUs, o carregamento falhará"
- [ ] Confirmar que a UI exibe `cpu_weight` quando > 0% no resultado do auto-balance

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `static/js/gpu.js` | Adicionar tooltip e exibicao de CPU weight |
| `llama_manager.py` | Se necessario, ajustar HTML template para tooltip do checkbox |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `static/js/index.js` | Pode precisar de ajustes no HTML template |
| `auto_balance.py` (task_08) | Fornece `cpu_checkbox_hint` no dict de failure |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- Tooltip ao checkbox de CPU com explicacao clara
- Destaque visual do peso de CPU no resultado do auto-balance
- `showAutoBalanceCapacityAlert()` exibe `cpu_checkbox_hint`
- Nenhuma nova configuracao, botao ou layout adicionado

## Testes

- [ ] `test_cpu_checkbox_tooltip_exists` - checkbox de CPU possui tooltip explicativo
- [ ] `test_cpu_weight_displayed_when_active` - CPU weight > 0% é exibido no resultado
- [ ] `test_capacity_alert_shows_cpu_hint` - `cpu_checkbox_hint` é exibido na mensagem de erro
- [ ] `test_no_new_config_fields` - nenhum novo campo de configuracao foi adicionado

## Critérios de Sucesso

- Tooltip ao checkbox de CPU funciona
- Peso de CPU destacado quando > 0%
- Mensagem de erro exibe hint sobre checkbox
- Nenhuma alteracao de layout ou novos campos adicionados
