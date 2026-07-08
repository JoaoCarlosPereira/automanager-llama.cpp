---
status: completed
title: Configuração e schemas do proxy (smart_proxy global + flags por modelo)
type: backend
complexity: medium
dependencies: []
---

# Configuração e schemas do proxy (smart_proxy global + flags por modelo)

## Visão Geral
Cria a base de configuração do Modo Proxy Inteligente: a chave global `smart_proxy` no `automanager_config.json` com métodos dedicados no `ConfigManager`, e as flags por modelo `proxy_eligible` e `max_parallel_requests` na whitelist de `update_model_settings`. Sem esta tarefa nenhum outro componente consegue ler ou persistir o estado do proxy.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. O `ConfigManager` DEVE expor leitura e atualização parcial da chave global `smart_proxy` com defaults `{enabled: false, primary_model_path: null, ttl_minutes: 180, max_wait_seconds: 30}` (ver "Modelos de Dados" do TechSpec).
- 2. A escrita DEVE seguir o padrão atômico existente (`.tmp` + `os.replace`) e o padrão de setter dedicado de `set_default_model` (`config_manager.py:246-266`).
- 3. `update_model_settings` DEVE aceitar e persistir `proxy_eligible` (default `true`) e `max_parallel_requests` (default `1`) na whitelist de campos por modelo, preservando entradas existentes via merge.
- 4. `schemas.py` DEVE ganhar `DEFAULT_PROXY_ELIGIBLE` e `DEFAULT_MAX_PARALLEL_REQUESTS`, e os schemas pydantic `ProxyConfigRequest` e `SetModelProxyRequest`; schemas com campos `model_*` DEVEM usar `model_config = ConfigDict(protected_namespaces=())`.
- 5. `max_parallel_requests` DEVE ser validado como inteiro >= 1; `ttl_minutes` e `max_wait_seconds` como inteiros >= 1.
- 6. A chave `smart_proxy` NÃO DEVE conter segredos (o endpoint `GET /config` retorna o dict sem `admin_password_hash`).
</requirements>

## Subtarefas
- [x] 1.1 Adicionar métodos `get_smart_proxy_settings()` e `update_smart_proxy_settings(partial)` ao `ConfigManager`, com merge sobre defaults.
- [x] 1.2 Incluir `proxy_eligible` e `max_parallel_requests` na whitelist de `update_model_settings` e no entry canônico por modelo.
- [x] 1.3 Criar constantes `DEFAULT_*` e schemas pydantic `ProxyConfigRequest` e `SetModelProxyRequest` em `schemas.py` com validações de faixa.
- [x] 1.4 Garantir que configs antigas (sem `smart_proxy` e sem as novas flags) carregam com defaults sem migração manual.
- [x] 1.5 Escrever testes unitários de persistência, merge parcial e validação.

## Detalhes de Implementação
Ver seções "Modelos de Dados" e "Design de Implementação" do TechSpec. Padrões a seguir: setter global dedicado como `set_default_model` (`config_manager.py:246-266`); whitelist por modelo em `update_model_settings` (`config_manager.py:185-244`, padrão `merged.get("flag", DEFAULT)`); constantes em `schemas.py:4-10`.

### Arquivos Relevantes
- `config_manager.py` — novos métodos globais + whitelist por modelo (linhas ~185-266)
- `schemas.py` — constantes `DEFAULT_*` e novos schemas pydantic

### Arquivos Dependentes
- `llama_manager.py` — consumirá os métodos e schemas nas tarefas 04 e 06
- `proxy_router.py` (novo, tarefas 02/03) — lerá `smart_proxy` e flags por modelo
- `tests/unit/test_config_manager*.py` — suíte existente de config não pode regredir

### ADRs Relacionados
- [ADR-005: Estado sticky e configuração](../adrs/adr-005.md) — define a chave `smart_proxy`, flags por modelo e `model_path` como identificador durável

## Entregáveis
- Métodos de leitura/escrita de `smart_proxy` no `ConfigManager` com escrita atômica
- Flags `proxy_eligible` e `max_parallel_requests` persistidas por modelo
- Schemas e defaults novos em `schemas.py`
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração de carga/salvamento do config com as novas chaves **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [x] `get_smart_proxy_settings()` em config sem a chave retorna os 4 defaults exatos
  - [x] `update_smart_proxy_settings({"enabled": true})` preserva `ttl_minutes`/`max_wait_seconds` existentes (merge parcial)
  - [x] `update_model_settings(path, {"proxy_eligible": false})` persiste e releitura retorna `false`; modelos sem a flag retornam default `true`
  - [x] `max_parallel_requests=0` e `ttl_minutes=0` são rejeitados pela validação pydantic
  - [x] Config legado (arquivo real de exemplo sem `smart_proxy`) carrega sem erro e sem alterar demais chaves
- Testes de integração:
  - [x] Ciclo completo: salvar `smart_proxy` + flags de 2 modelos → reler do disco → valores idênticos, arquivo permanece JSON válido
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Config antigo carrega com defaults sem intervenção manual
- Nenhuma regressão na suíte existente de `config_manager`
