---
status: completed
title: Persistir mtp_enabled/mtp_draft_tokens no config_manager
type: backend
complexity: low
dependencies:
  - task_01
---

# Persistir mtp_enabled/mtp_draft_tokens no config_manager

## Visão Geral

Estende `ConfigManager.update_model_settings()` para persistir `mtp_enabled` e `mtp_draft_tokens` em `automanager_config.json` por caminho de modelo, com merge e defaults seguros para configs legadas.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE incluir `mtp_enabled` e `mtp_draft_tokens` no dict `entry` de `update_model_settings()`
- DEVE usar defaults `False` e `3` quando campos ausentes no merge
- DEVE preservar valores existentes ao fazer merge parcial de settings
- DEVE importar defaults de `schemas.py` (ou duplicar constantes em `config_manager.py` se já houver pattern local)
- DEVE garantir que `get_model_settings()` retorna os campos após save
</requirements>

## Subtarefas

- [ ] 2.1 Adicionar campos MTP ao dict `entry` em `update_model_settings()`
- [ ] 2.2 Alinhar defaults com `schemas.py` (`DEFAULT_MTP_ENABLED`, `DEFAULT_MTP_DRAFT_TOKENS`)
- [ ] 2.3 Criar testes unitários de persistência e merge
- [ ] 2.4 Verificar round-trip save/load via arquivo JSON temporário

## Detalhes de Implementação

Espelhar campos existentes como `thinking_enabled`, `parallel_slots` e `batch_size` no bloco `entry` de `config_manager.py`. Ver seção **Modelos de Dados** do TechSpec.

### Arquivos Relevantes

- `config_manager.py` — método `update_model_settings()` e constantes de default

### Arquivos Dependentes

- `llama_manager.py` — chama `update_model_settings()` no `/start`
- `process_manager.py` — persiste settings após auto-balance
- `model_manager.py` — expõe `last_config` via `/models`

### ADRs Relacionados

- [ADR-001: Campos MTP sempre visíveis](../adrs/adr-001.md) — config salva por modelo

## Entregáveis

- Persistência MTP em `config_manager.py`
- Testes unitários de save/load/merge com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] `update_model_settings()` grava `mtp_enabled=True` e `mtp_draft_tokens=5` no JSON
  - [ ] `get_model_settings()` retorna campos MTP após save
  - [ ] Merge parcial preserva `mtp_draft_tokens` quando só `context_size` é atualizado
  - [ ] Config legada sem campos MTP retorna defaults via merge (`False`, `3`)
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% nos caminhos MTP de `config_manager.py`
- JSON persistido contém `mtp_enabled` e `mtp_draft_tokens` por modelo
