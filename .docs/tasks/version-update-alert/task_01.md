---
status: completed
title: Modelos Pydantic de verificação de versão
type: backend
complexity: low
dependencies: []
---

# Modelos Pydantic de verificação de versão

## Visão Geral

Define os modelos `VersionCommit` e `VersionCheckResponse` em `schemas.py`, estabelecendo o contrato JSON do endpoint de verificação de versão. Esta tarefa é pré-requisito da rota FastAPI e garante validação tipada das respostas.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- DEVE adicionar `VersionCommit` com campos `sha`, `message`, `author` e `date` (string ISO 8601)
- DEVE adicionar `VersionCheckResponse` com `status` restrito a `ok`, `unavailable` ou `error`
- DEVE incluir `update_available: bool` com default `False`
- DEVE incluir campos opcionais `current_ref`, `remote_ref`, `branch` e `error_message`
- DEVE incluir `commits: List[VersionCommit]` com default lista vazia
- DEVE serializar corretamente para JSON compatível com o frontend `version.js`
</requirements>

## Subtarefas

- [x] 1.1 Adicionar `VersionCommit` e `VersionCheckResponse` em `schemas.py` após os modelos existentes
- [x] 1.2 Importar `Literal` e `List` conforme padrão do arquivo
- [x] 1.3 Criar `tests/unit/test_schemas_version.py` com casos de serialização e defaults
- [x] 1.4 Verificar que nenhum import existente de `schemas.py` quebra

## Detalhes de Implementação

Seguir o padrão de modelos Pydantic já usado em `LoginRequest` e `StartRequest`. Ver seção **Modelos de Dados** e **Interfaces Principais** do TechSpec.

### Arquivos Relevantes

- `schemas.py` — definição dos modelos de resposta da verificação de versão
- `tests/unit/test_schemas_mtp.py` — referência de estilo para testes Pydantic no projeto

### Arquivos Dependentes

- `llama_manager.py` — importará `VersionCheckResponse` na rota (tarefa 03)
- `version_manager.py` — resultado interno convertido para o schema na rota (tarefa 02/03)

### ADRs Relacionados

- [ADR-002: Módulo version_manager com subprocess git](adrs/adr-002.md) — define o contrato exposto pela API

## Entregáveis

- Modelos `VersionCommit` e `VersionCheckResponse` em `schemas.py`
- Arquivo `tests/unit/test_schemas_version.py` com cobertura >= 80% **(OBRIGATÓRIO)**

## Testes

- Testes unitários:
  - [ ] `VersionCheckResponse()` com defaults: `status` obrigatório, `update_available=False`, `commits=[]`
  - [ ] `VersionCheckResponse(status="ok", update_available=True, commits=[...])` serializa JSON com todos os campos
  - [ ] `VersionCommit` com `sha`, `message`, `author`, `date` preenchidos aceita payload válido
  - [ ] `status` inválido (ex.: `"pending"`) rejeita com `ValidationError`
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso

- Todos os testes passando
- Cobertura de testes >= 80% nos novos modelos em `schemas.py`
- `VersionCheckResponse.model_dump()` produz estrutura compatível com o contrato do TechSpec
