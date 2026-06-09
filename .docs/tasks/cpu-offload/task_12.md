---
status: completed
title: Validações, tratamento de erros e casos de borda
type: backend
complexity: medium
dependencies:
  - task_04
  - task_11
---

# Tarefa 12: Validações, tratamento de erros e casos de borda

## Visão Geral

Esta tarefa final realiza a polidação da implementação: validações adicionais, tratamento de erros, casos de borda, e alertas de capacidade de hardware. Inclui também a correção de quaisquer problemas identificados durante os testes de integração da tarefa 11.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O sistema DEVE exibir alerta quando CPU weight > 50% (aviso de performance degradada)
- O sistema DEVE manter o alerta existente de "Modelo além da capacidade" e estendê-lo para incluir informações sobre RAM disponível
- O frontend DEVE validar visualmente que soma de pesos = 100% com feedback imediato
- O backend DEVE retornar erro descritivo para pesos inválidos (soma != 100%, CPU > 70% manual)
- O sistema DEVE lidar gracefulmente com configs antigas (sem campo `device`)
- O frontend DEVE mostrar tooltip ou label indicando limite de 70% no input de peso da CPU
- Todos os seletores JS DEVEM incluir null-checks defensivos

</requirements>

## Subtarefas
- [ ] 12.1 Adicionar alerta visual quando CPU weight > 50% (badge ou cor de alerta)
- [ ] 12.2 Estender alerta de "Modelo além da capacidade" para incluir informações de RAM
- [x] 12.3 Adicionar validação visual no frontend: soma = 100% com feedback colorido
- [ ] 12.4 Adicionar tooltip/label indicando limite de 70% no input de peso da CPU
- [x] 12.5 Garantir backward-compatibility: configs sem campo `device` são lidas corretamente
- [x] 12.6 Revisar todos os null-checks em seletores JS
- [x] 12.7 Revisar e corrigir quaisquer problemas encontrados nos testes da tarefa 11
- [x] 12.8 Executar suite completa de testes e garantir passagem
- [ ] 12.9 Teste manual final: fluxo completo em ambiente de desenvolvimento

## Detalhes de Implementação

**Arquivos modificados:** `llama_manager.py`, `static/js/gpu.js`, `static/js/metrics.js`, `static/js/models.js`, `process_manager.py`

Os alertas e validações são adicionados nos pontos existentes de validação do sistema, sem introduzir mudanças arquiteturais.

### Arquivos Relevantes
- `llama_manager.py` — alertas de capacidade no dashboard HTML
- `static/js/gpu.js` — validação visual de soma de pesos
- `static/js/models.js` — null-checks defensivos
- `process_manager.py` — validação de pesos no backend
- `d:\dsv-git\automanager-llama.cpp\llama_manager.py` — linhas 425-679 (index), 682-1062 (_build_html)
- `d:\dsv-git\automanager-llama.cpp\static\js\gpu.js` — todo o arquivo

### Arquivos Dependentes
- Todas as tarefas 01-11 devem estar concluídas
- `tests/unit/test_html_contract.py` — padrão de teste de contrato HTML

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado
- [ADR-004](adrs/adr-004.md) — Priorização GPU no Auto Balance com Limite de 70%

## Entregáveis
- Alerta de CPU weight alto implementado
- Alerta de capacidade estendido com informações de RAM
- Validação visual de soma = 100% com feedback
- Tooltip indicando limite de 70% no input de peso
- Backward-compatibility garantida
- Todos os null-checks revisados
- Suite de testes completa passando
- Teste manual final concluído

## Testes
- Testes unitários:
  - [ ] Alerta é exibido quando CPU weight > 50%
  - [ ] Alerta NÃO é exibido quando CPU weight <= 50%
  - [x] Config sem campo `device` é lida corretamente (default `"gpu"`)
  - [x] Backend retorna erro 400 para soma de pesos != 100%
  - [x] Backend retorna erro 400 para CPU weight > 70%
  - [x] Todos os seletores JS retornam null-safe
- Testes de integração:
  - [ ] Fluxo completo funciona em ambiente de desenvolvimento
  - [ ] Zero regressões em funcionalidades existentes de GPU
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Zero regressões em funcionalidades existentes
- Fluxo completo de CPU offload funciona em ambiente de desenvolvimento
- Usuário recebe feedback claro sobre limites e capacidades
- Configs antigas funcionam sem modificação
