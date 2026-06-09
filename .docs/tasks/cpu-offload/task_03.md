---
status: pending
title: Estender /metrics com RAM detalhada e cpu_name
type: backend
complexity: low
dependencies:
  - task_02
---

# Tarefa 03: Estender `/metrics` com RAM detalhada e `cpu_name`

## Visão Geral

Esta tarefa estende o response do endpoint `/metrics` para incluir campos adicionais de RAM (`ram_used_mb`, `ram_total_mb`) e o nome do processador (`cpu_name`). O endpoint já retorna `cpu` (usage %) e `ram` (percentual) — a tarefa é apenas adicionar os novos campos usando os dados coletados pela tarefa 02.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O response JSON do endpoint `/metrics` DEVE incluir `ram_used_mb` (int) e `ram_total_mb` (int)
- O response DEVE incluir `cpu_name` (str) com o nome completo do processador
- Os campos novos DEVE ser adicionados ao dicionário retornado por `get_metrics()`
- Os campos existentes (`cpu`, `ram`, `gpus`) DEVE permanecer inalterados
- Valores padrão DEVE ser `0` para campos de RAM e `"CPU Host"` para `cpu_name` quando indisponíveis

</requirements>

## Subtarefas
- [ ] 03.1 Chamar `detect_cpu_info()` dentro de `get_metrics()` ou no endpoint `/metrics`
- [ ] 03.2 Adicionar `ram_used_mb`, `ram_total_mb`, `cpu_name` ao response dict
- [ ] 03.3 Verificar que o response JSON é válido e inclui todos os campos
- [ ] 03.4 Escrever testes para o endpoint `/metrics` com novos campos

## Detalhes de Implementação

**Arquivo principal:** `gpu_manager.py` — método `get_metrics()` (linhas 74-110) ou `llama_manager.py` — endpoint `/metrics` (linha 275-277)

O método `get_metrics()` retorna um dict com `cpu`, `ram`, `gpus`. O dict estendido será:

```python
{
    "cpu": float,
    "ram": float,
    "ram_used_mb": int,
    "ram_total_mb": int,
    "cpu_name": str,
    "gpus": [...]
}
```

A integração pode ser feita dentro de `get_metrics()` ou diretamente no endpoint `/metrics` em `llama_manager.py`.

### Arquivos Relevantes
- `gpu_manager.py` — método `get_metrics()` e classe `GPUDetector`
- `llama_manager.py` — endpoint `@app.get("/metrics")` (linha 275-277)

### Arquivos Dependentes
- `static/js/metrics.js` — frontend precisará consumir os novos campos (tarefa 09)
- `tests/unit/test_gpu_scanner.py` — padrão de teste existente

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado na Tabela de Recursos

## Entregáveis
- Endpoint `/metrics` retorna `ram_used_mb`, `ram_total_mb`, `cpu_name`
- Campos existentes inalterados
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: GET `/metrics` retorna todos os campos esperados **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `get_metrics()` retorna `ram_used_mb` como int positivo
  - [ ] `get_metrics()` retorna `ram_total_mb` como int positivo
  - [ ] `get_metrics()` retorna `cpu_name` como string não-vazia
  - [ ] `get_metrics()` retorna todos os campos existentes (`cpu`, `ram`, `gpus`)
- Testes de integração:
  - [ ] GET `/metrics` retorna JSON com todos os campos esperados
  - [ ] Valores de `ram_used_mb` e `ram_total_mb` são coerentes com o sistema
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Response `/metrics` inclui todos os novos campos com valores válidos
- Compatibilidade com frontend existente (campos antigos ainda funcionam)
