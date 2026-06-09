---
status: completed
title: Implementar detect_model_mtp() no gpu_manager
type: backend
complexity: low
dependencies: []
---

# Implementar detect_model_mtp() no gpu_manager

## Visão Geral

Implementa detecção de compatibilidade MTP via `llama-server --model-info`, parseando `nextn_predict_layers > 0`. Segue o mesmo pattern de `detect_model_layers()` para subprocess, timeout e fallback seguro.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE implementar `GPUManager.detect_model_mtp(model_path: str) -> bool`
- DEVE executar `llama-server --model-info` com `CUDA_VISIBLE_DEVICES=""` e timeout 15s
- DEVE retornar `True` quando regex encontra `nextn_predict_layers = N` com N > 0
- DEVE retornar `False` em falha de subprocess, timeout, metadado ausente ou N = 0
- DEVE logar warning em falha (pattern de `detect_model_layers`)
</requirements>

## Subtarefas

- [ ] 3.1 Implementar `detect_model_mtp()` em `gpu_manager.py`
- [ ] 3.2 Reutilizar env/subprocess pattern de `detect_model_layers()`
- [ ] 3.3 Criar testes com mock de `subprocess.check_output` e fixtures de output
- [ ] 3.4 Cobrir casos de borda: output extra, N=0, exceção, timeout

## Detalhes de Implementação

Ver seção **Interfaces Principais** do TechSpec e [ADR-003](../adrs/adr-003.md). Referência de implementação existente: `gpu_manager.detect_model_layers()` (~linha 395).

### Arquivos Relevantes

- `gpu_manager.py` — novo método `detect_model_mtp()`

### Arquivos Dependentes

- `process_manager.py` — `mtp_cli_args()` invocará detecção (task_04)

### ADRs Relacionados

- [ADR-003: Detecção via model-info](../adrs/adr-003.md) — critério `nextn_predict_layers > 0`
- [ADR-002: Ignorar silenciosamente MTP incompatível](../adrs/adr-002.md) — retorno False habilita skip de flags

## Entregáveis

- Método `detect_model_mtp()` em `gpu_manager.py`
- Testes unitários com mocks de subprocess, cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] Output com `nextn_predict_layers = 1` retorna `True`
  - [ ] Output com `nextn_predict_layers = 0` retorna `False`
  - [ ] Output sem padrão retorna `False`
  - [ ] `subprocess.check_output` lança exceção → retorna `False` + log warning
  - [ ] Output com ruído extra (como testes de `detect_model_layers`) ainda parseia corretamente
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% em `detect_model_mtp()`
- Comportamento alinhado ao ADR-003
