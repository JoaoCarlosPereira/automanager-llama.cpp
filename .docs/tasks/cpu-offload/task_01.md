---
status: pending
title: Extender GPUWeight com campo device
type: backend
complexity: low
dependencies: []
---

# Tarefa 01: Extender GPUWeight com campo `device`

## Visão Geral

Esta tarefa adiciona o campo `device: str = "gpu"` ao modelo `GPUWeight` em `schemas.py`, permitindo que o mesmo schema represente tanto GPUs quanto a CPU como dispositivos de compute. É a base de toda a implementação — sem essa mudança, nenhum outro módulo pode distinguir ou manipular dispositivos CPU.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O schema `GPUWeight` DEVE ter um campo `device: str = "gpu"` com valor padrão `"gpu"`
- A CPU DEVE usar `device="cpu"` e `index=-1` como convenção
- O campo DEVE ser backward-compatible: requests antigos sem `device` funcionam sem erro
- `is_main` DEVE ser sempre `False` para `device="cpu"` (validação mínima no schema)
- A serialização JSON DEVE incluir o campo `device` em todas as respostas da API
</requirements>

## Subtarefas
- [ ] 01.1 Adicionar campo `device: str = "gpu"` ao modelo `GPUWeight`
- [ ] 01.2 Adicionar validação opcional: `is_main=False` quando `device="cpu"`
- [ ] 01.3 Adicionar docstring explicando os valores permitidos de `device`
- [ ] 01.4 Verificar que `StartRequest` e `model_dump()` incluem `device` nas serializações
- [ ] 01.5 Escrever testes unitários para serialização/deserialização com e sem campo `device`

## Detalhes de Implementação

**Arquivo principal:** `schemas.py` (linha 10-16)

O modelo `GPUWeight` está em `d:\dsv-git\automanager-llama.cpp\schemas.py`. O campo `device` é adicionado com valor padrão `"gpu"`.

### Arquivos Relevantes
- `schemas.py` — definição do schema `GPUWeight` e `StartRequest`
- `tests/unit/test_gpu_scanner.py` — padrão de testes existentes para schemas

### Arquivos Dependentes
- `gpu_manager.py` — usa `GPUWeight` para validar pesos; precisará adaptar filtros por `device`
- `process_manager.py` — usa `GPUWeight` no cálculo de `--n-gpu-layers` (tarefa 04)
- `auto_balance.py` — usa `GPUWeight` na conversão weight map → lista (tarefa 05)
- `static/js/models.js` — frontend precisará enviar `device: "cpu"` (tarefa 10)

### ADRs Relacionados
- [ADR-002](adrs/adr-002.md) — Extensão do Schema GPUWeight com Campo Device

## Entregáveis
- Campo `device` adicionado ao `GPUWeight` com default `"gpu"`
- Serialização JSON inclui `device` em todas as respostas
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: `StartRequest` com `device="cpu"` é aceito pela API **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `GPUWeight(device="gpu", ...)` serializa com `"device": "gpu"`
  - [ ] `GPUWeight(device="cpu", index=-1, ...)` serializa corretamente
  - [ ] `GPUWeight(...)` sem campo `device` usa default `"gpu"`
  - [ ] `StartRequest` com lista mista de GPUs e CPU é serializado corretamente
  - [ ] `is_main=True` com `device="cpu"` é rejeitado ou normalizado para `False`
- Testes de integração:
  - [ ] POST `/start` com `device="cpu"` na lista de weights é aceito
  - [ ] GET `/status` retorna `device` nos weights
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Schema backward-compatible: requests antigos sem `device` não quebram
- `model_dump()` inclui `device` em todas as serializações
