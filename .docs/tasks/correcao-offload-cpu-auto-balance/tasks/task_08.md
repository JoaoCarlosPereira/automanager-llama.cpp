---
status: pending
title: Atualizar build_hardware_capacity_failure() - mensagem com info sobre checkbox CPU
type: backend
complexity: low
dependencies: [task_04]
---

# Task 08: Atualizar build_hardware_capacity_failure() - mensagem com checkbox CPU

## Visao Geral

Atualizar o metodo `AutoBalanceProber.build_hardware_capacity_failure()` em `auto_balance.py` para incluir informacoes sobre o checkbox de CPU na mensagem de erro quando o modelo excede a capacidade do hardware. A mensagem deve sugerir ativar o checkbox de CPU como alternativa.

<critical>
- Ler o TechSpec secao 5.1 e PRD secao "Mensagem de Erro Clarificadora" antes de implementar
- A mensagem DEVE sugerir ativar o checkbox de CPU quando CPU esta desativado
- As sugestoes existentes (quantizacao, contexto, modelo menor) DEVEM permanecer
- Nao alterar a estrutura do dict `failure` exceto adicionar campo sobre CPU
</critical>

<requirements>
1. `build_hardware_capacity_failure()` DEVE aceitar um parametro opcional `cpu_enabled: bool = False`
2. Quando `cpu_enabled=False`, a mensagem DEVE incluir: "Marcar o checkbox de CPU permitiria usar a CPU como fallback"
3. A mensagem DEVE incluir sugestoes contextuais baseadas no estado do checkbox de CPU
4. O dict `failure` DEVE incluir um campo `cpu_checkbox_hint` com mensagem para o frontend exibir
5. As sugestoes originais (quantizacao, contexto, modelo menor) DEVEM permanecer
6. A estrutura de retorno `(message: str, failure: dict)` DEVE ser mantida
</requirements>

## Subtarefas

- [ ] Adicionar parametro opcional `cpu_enabled: bool = False` ao metodo `build_hardware_capacity_failure()`
- [ ] Quando `cpu_enabled=False`, adicionar hint: "Marcar o checkbox de CPU permitiria usar a CPU como fallback para offload de layers"
- [ ] Adicionar campo `cpu_checkbox_hint` ao dict `failure` com a mensagem contextual
- [ ] Adicionar "Ativar CPU como fallback" a lista de `suggestions` quando `cpu_enabled=False`
- [ ] Confirmar que as sugestoes originais permanecem (quantizacao, contexto, modelo menor)
- [ ] Verificar que quando `cpu_enabled=True`, a mensagem padrao é mantida

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `auto_balance.py` | Modificar `build_hardware_capacity_failure()` (linhas 432-478) |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `gpu.js` | `showAutoBalanceCapacityAlert()` deve exibir `cpu_checkbox_hint` |
| `llama_manager.py` | Endpoint de auto-balance passa `cpu_enabled` para `build_hardware_capacity_failure()` |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- `build_hardware_capacity_failure()` com hint sobre checkbox de CPU
- Mensagem contextual quando CPU esta desativado
- Sugestoes originais mantidas
- Estrutura de retorno inalterada

## Testes

- [ ] `test_build_hardware_capacity_failure_with_cpu_off` - `cpu_enabled=False` -> mensagem contem hint sobre checkbox
- [ ] `test_build_hardware_capacity_failure_with_cpu_on` - `cpu_enabled=True` -> mensagem padrao sem hint
- [ ] `test_build_hardware_capacity_failure_suggestions_preserved` - sugestoes originais presentes
- [ ] `test_failure_dict_structure` - dict `failure` contem `cpu_checkbox_hint` quando aplicavel

## Critérios de Sucesso

- Mensagem inclui sugestao de checkbox CPU quando `cpu_enabled=False`
- Sugestoes originais (quantizacao, contexto, modelo) mantidas
- Estrutura de retorno `(message, failure)` inalterada
- Campo `cpu_checkbox_hint` presente no dict `failure`
