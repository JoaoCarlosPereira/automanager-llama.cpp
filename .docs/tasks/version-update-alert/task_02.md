---
status: completed
title: Módulo version_manager com subprocess git
type: backend
complexity: medium
dependencies: []
---

# Módulo version_manager com subprocess git

## Visão Geral

Implementa `version_manager.py` com a função `check_for_updates(install_root)` que executa comandos git via `subprocess` para detectar commits ahead do remoto e extrair notas de atualização. É o núcleo da lógica backend da feature.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE criar `version_manager.py` na raiz do projeto com dataclasses internas e `check_for_updates()`
- DEVE usar `subprocess.run` (não `os.system`) com timeout configurável no `git fetch` (default 30s)
- DEVE comparar branch atual do checkout com `origin/<mesmo-branch>`
- DEVE retornar `status=unavailable` quando não for work-tree git
- DEVE retornar `status=error` quando `git fetch` ou comandos subsequentes falharem
- DEVE retornar commits em ordem cronológica (antigo → recente) via `git log --reverse`
- DEVE usar SHA curto (7 caracteres) em `current_ref` e `remote_ref` na resposta
- DEVE registrar logs INFO/WARNING/ERROR conforme seção Monitoramento do TechSpec
- NÃO DEVE adicionar dependências pip (GitPython proibido)
</requirements>

## Subtarefas

- [x] 2.1 Criar dataclasses `VersionCommit` e `VersionCheckResult` no módulo
- [x] 2.2 Implementar helper `_run_git()` com tratamento de timeout e returncode
- [x] 2.3 Implementar fluxo: validar work-tree → obter branch → fetch → comparar refs → listar commits
- [x] 2.4 Criar `tests/unit/test_version_manager.py` com mocks de `subprocess.run`
- [x] 2.5 Garantir que `paths.INSTALL_ROOT` é o default de `install_root` na rota (tarefa 03)

## Detalhes de Implementação

Seguir padrão subprocess de `gpu_manager.py` e `process_manager.py`. Usar `paths.INSTALL_ROOT` como diretório do repositório. Ver seção **Endpoints de API** (comportamento backend) e **Interfaces Principais** do TechSpec.

### Arquivos Relevantes

- `version_manager.py` — novo módulo (criar)
- `paths.py` — `INSTALL_ROOT` como raiz do repositório git
- `gpu_manager.py` — referência de padrão `subprocess.check_output` / mock em testes
- `llama_manager.py` linhas 443–459 — `_execute_update()` usa `git pull` no mesmo diretório

### Arquivos Dependentes

- `llama_manager.py` — importará `check_for_updates` na rota (tarefa 03)
- `log_manager.py` / logger do manager — padrão de logging existente em `llama_manager.py`

### ADRs Relacionados

- [ADR-002: Módulo version_manager com subprocess git](adrs/adr-002.md) — decisão técnica principal desta tarefa

## Entregáveis

- Módulo `version_manager.py` com `check_for_updates()`
- Arquivo `tests/unit/test_version_manager.py` com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] Work-tree válido com remoto 2 commits ahead → `update_available=True`, lista com 2 commits parseados (sha, message, author, date)
  - [ ] HEAD igual a `origin/<branch>` → `update_available=False`, `commits=[]`
  - [ ] Diretório sem `.git` → `status=unavailable`, sem exceção propagada
  - [ ] `git fetch` retorna código != 0 → `status=error` com `error_message` preenchido
  - [ ] `subprocess.TimeoutExpired` no fetch → `status=error`
  - [ ] Branch com nome contendo `/` (ex.: `feature/foo`) → comandos usam ref correta
  - [ ] Lista longa (10+ commits) retorna todos sem truncamento
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% em `version_manager.py`
- `check_for_updates()` nunca levanta exceção não tratada para o caller da rota
- Parsing de `git log --format` produz commits na ordem cronológica ascendente
