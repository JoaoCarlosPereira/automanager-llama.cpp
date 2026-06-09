---
status: pending
title: Atualizar gpu.js - remover verificacao de cap de 70% no frontend
type: frontend
complexity: low
dependencies: []
---

# Task 06: Atualizar gpu.js - remover verificacao de cap de 70%

## Visao Geral

Remover a verificacao de cap de 70% no CPU da funcao `validateDeviceWeights()` em `static/js/gpu.js`. O frontend deve parar de rejeitar pesos de CPU acima de 70%, ja que o backend tambem removera esse limite.

<critical>
- Ler o TechSpec secao 4.6 antes de implementar
- Remover APENAS a verificacao de cap de 70% - todas as outras validacoes (soma=100%, pelo menos uma GPU) DEVEM permanecer
- Nao alterar `collectDeviceWeightsFromUI()` - ele ja coleta o checkbox de CPU corretamente
- Nao adicionar novos campos ao payload de start neste arquivo
</critical>

<requirements>
1. A verificacao `if (cpuWeight > 70)` na funcao `validateDeviceWeights()` DEVE ser completamente removida (linhas 290-299)
2. A mensagem de erro associada ao cap de 70% DEVE ser removida
3. A validacao de soma ≈100% DEVE permanecer inalterada (linhas 282-288)
4. A validacao de pelo menos uma GPU ativa DEVE permanecer inalterada (linhas 276-280)
5. `collectDeviceWeightsFromUI()` DEVE permanecer inalterado
6. O frontend DEVE continuar enviando o checkbox de CPU via `active` field do GPUWeight
</requirements>

## Subtarefas

- [ ] Localizar a verificacao `if (cpuWeight > 70)` em `validateDeviceWeights()` (linhas 290-299)
- [ ] Remover todo o bloco `if (cpuWeight > 70) { return { ok: false, message: ... }; }`
- [ ] Verificar que a validacao de soma (linhas 282-288) permanece
- [ ] Verificar que a validacao de GPU ativa (linhas 276-280) permanece
- [ ] Verificar que `collectDeviceWeightsFromUI()` permanece inalterado
- [ ] Confirmar que `validateDeviceWeights()` retorna `{ ok: true, message: '' }` para weights validos com CPU > 70%

## Detalhes de Implementacao

### Arquivos Relevantes

| Arquivo | Motivo |
|---------|--------|
| `static/js/gpu.js` | Remover verificacao de cap em `validateDeviceWeights()` (linhas 290-299) |

### Arquivos Dependentes

| Arquivo | Motivo |
|---------|--------|
| `llama_manager.py` | Endpoint `start_model` consome os weights do frontend |
| `static/js/index.js` | Chama `validateDeviceWeights()` antes de enviar start |

### ADRs Relacionados

- [ADR-001: Unificacao do Motor de Distribuicao de Carga](../adrs/adr-001.md)

## Entregáveis

- `validateDeviceWeights()` sem verificacao de cap de 70%
- Todas as outras validacoes (soma=100%, GPU ativa) intactas
- `collectDeviceWeightsFromUI()` inalterado

## Testes

- [ ] `test_validateDeviceWeights_cpu_80_percent` - weights com CPU=80% e soma=100% devem passar (antes falhavam)
- [ ] `test_validateDeviceWeights_sum_check_still_works` - weights somando != 100% ainda devem falhar
- [ ] `test_validateDeviceWeights_no_gpu_still_works` - sem GPU ativa ainda deve falhar
- [ ] `test_collectDeviceWeightsFromUI_unchanged` - funcao continua coletando todos os campos corretamente

## Critérios de Sucesso

- Capacidade de enviar CPU > 70% pelo frontend sem erro de validacao
- Validacoes de soma e GPU ativa funcionando
- NENHUM lint no codigo JavaScript
- Jest tests existentes continuam passando
