---
status: pending
title: Implementar detect_cpu_info() no gpu_manager
type: backend
complexity: low
dependencies:
  - task_01
---

# Tarefa 02: Implementar `detect_cpu_info()` no gpu_manager

## Visão Geral

Esta tarefa implementa o método `detect_cpu_info()` na classe `GPUDetector` (`gpu_manager.py`), que retorna o nome completo do processador e as informações de RAM (usada, total em MB, percentual). É o primeiro passo para exibir dados do CPU no dashboard.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O método `detect_cpu_info()` DEVE retornar um dicionário com: `name`, `ram_used_mb`, `ram_total_mb`, `ram_percent`
- O nome do processador DEVE ser obtido via `platform.processor()` com fallback para `platform.machine()` e depois `"CPU Host"`
- A RAM DEVE ser obtida via `psutil.virtual_memory()` (já é dependência do projeto)
- Em caso de qualquer exceção, o método DEVE retornar valores padrão sem falhar: `{"name": "CPU Host", "ram_used_mb": 0, "ram_total_mb": 0, "ram_percent": 0}`
- `ram_used_mb` e `ram_total_mb` DEVE ser inteiros em megabytes
</requirements>

## Subtarefas
- [ ] 02.1 Implementar método `detect_cpu_info()` na classe `GPUDetector`
- [ ] 02.2 Adicionar fallback em cadeia: `platform.processor()` → `platform.machine()` → `"CPU Host"`
- [ ] 02.3 Obter RAM via `psutil.virtual_memory()` e calcular valores em MB
- [ ] 02.4 Adicionar tratamento de exceção com valores padrão
- [ ] 02.5 Escrever testes unitários para caminho feliz e fallback

## Detalhes de Implementação

**Arquivo principal:** `gpu_manager.py` (classe `GPUDetector`, linhas 24-110)

O método `detect_cpu_info()` é adicionado como um novo método público da classe `GPUDetector`. `psutil` já é importado e usado no mesmo arquivo (`cpu_percent()`, `virtual_memory()`).

### Arquivos Relevantes
- `gpu_manager.py` — classe `GPUDetector` com métodos `detect_gpus()` e `get_metrics()`
- `d:\dsv-git\automanager-llama.cpp\gpu_manager.py` — linha 24 (classe), linha 104-105 (psutil usage)

### Arquivos Dependentes
- `llama_manager.py` — `index()` chamará `detect_cpu_info()` para injetar no HTML (tarefa 07)
- `llama_manager.py` — `/metrics` endpoint usará os dados de CPU/RAM (tarefa 03)

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado na Tabela de Recursos

## Entregáveis
- Método `detect_cpu_info()` implementado e testado
- Retorna nome do processador, RAM usada/total em MB, RAM percentual
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: método retorna valores corretos em ambiente de teste **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] `detect_cpu_info()` retorna `name` com valor não-vazio quando `platform.processor()` funciona
  - [ ] `detect_cpu_info()` retorna `"CPU Host"` quando `platform.processor()` lança exceção
  - [ ] `detect_cpu_info()` retorna `"CPU Host"` quando `platform.processor()` retorna string vazia
  - [ ] `ram_used_mb` e `ram_total_mb` são inteiros positivos
  - [ ] `ram_percent` está entre 0 e 100
- Testes de integração:
  - [ ] `detect_cpu_info()` é chamado com sucesso pela classe `GPUManager` (subclasse)
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Método robusto: nunca lança exceção, sempre retorna dicionário com chaves esperadas
- Valores de RAM coerentes com o sistema real
