---
status: completed
title: Implementar mtp_cli_args() e integrar no process_manager.start
type: backend
complexity: medium
dependencies:
  - task_01
  - task_03
---

# Implementar mtp_cli_args() e integrar no process_manager.start

## Visão Geral

Cria a função `mtp_cli_args()` para montar flags CLI do llama-server e integra no fluxo de start, status e OOM recovery. Flags MTP só entram no comando quando toggle ligado e modelo compatível.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE implementar `mtp_cli_args(mtp_enabled, mtp_draft_tokens, model_path, gpu_manager) -> List[str]`
- DEVE retornar `[]` quando `mtp_enabled=False` ou `detect_model_mtp()` retorna False
- DEVE retornar `["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]` quando aplicável, com n clamped 1–6
- DEVE estender assinatura de `process_manager.start()` com `mtp_enabled` e `mtp_draft_tokens`
- DEVE chamar `cmd.extend(mtp_cli_args(...))` após `reasoning_cli_args()`
- DEVE incluir campos MTP em `_last_request` e no dict `config` de `get_status()`
- DEVE logar `mtp_enabled`, `mtp_draft_tokens` e se flags foram aplicadas na linha START
- DEVE logar INFO quando MTP solicitado mas modelo incompatível (ADR-002)
</requirements>

## Subtarefas

- [ ] 4.1 Implementar `mtp_cli_args()` em `process_manager.py`
- [ ] 4.2 Estender `start()` com parâmetros MTP e integração no cmd
- [ ] 4.3 Propagar campos em `_last_request` e `get_status()["config"]`
- [ ] 4.4 Atualizar OOM recovery/restart para repassar campos MTP
- [ ] 4.5 Criar `tests/unit/test_mtp_cli_args.py` com casos enabled/disabled/incompatible

## Detalhes de Implementação

Espelhar `reasoning_cli_args()` (~linha 35) e integração em `start()` (~linha 407). Ver seções **Interfaces Principais** e **Monitoramento e Observabilidade** do TechSpec.

### Arquivos Relevantes

- `process_manager.py` — `mtp_cli_args()`, `start()`, `get_status()`, OOM watchdog restart

### Arquivos Dependentes

- `gpu_manager.py` — `detect_model_mtp()` (task_03)
- `schemas.py` — tipos StartRequest (task_01)
- `llama_manager.py` — repassa args ao start (task_05)
- `auto_balance.py` — repassa args ao start (task_10)

### ADRs Relacionados

- [ADR-004: Helpers em gpu_manager + process_manager](../adrs/adr-004.md) — organização dos helpers
- [ADR-002: Ignorar silenciosamente](../adrs/adr-002.md) — skip silencioso de flags
- [ADR-003: Detecção via model-info](../adrs/adr-003.md) — critério de compatibilidade

## Entregáveis

- `mtp_cli_args()` e integração completa em `process_manager.py`
- Testes unitários em `tests/unit/test_mtp_cli_args.py`, cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] `mtp_cli_args(True, 3, path, mgr)` com mock `detect_model_mtp=True` → flags `--spec-type draft-mtp --spec-draft-n-max 3`
  - [ ] `mtp_cli_args(False, 3, path, mgr)` → `[]` sem chamar detecção
  - [ ] `mtp_cli_args(True, 3, path, mgr)` com mock `detect_model_mtp=False` → `[]`
  - [ ] `mtp_cli_args(True, 9, path, mgr)` clamped para max 6
  - [ ] `get_status()` inclui `mtp_enabled` e `mtp_draft_tokens` em `config` quando running
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% em `mtp_cli_args()` e caminhos MTP de `start()`
- Comando llama-server inclui flags MTP apenas quando toggle on + modelo compatível
