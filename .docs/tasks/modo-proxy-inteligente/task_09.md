---
status: completed
title: Observabilidade (logs [proxy]) e testes de integração ponta a ponta
type: backend
complexity: medium
dependencies:
  - task_04
  - task_05
  - task_06
---

# Observabilidade (logs [proxy]) e testes de integração ponta a ponta

## Visão Geral
Fecha a entrega implementando os logs estruturados de roteamento exigidos pelo PRD F11 e validando o cenário completo do produto em testes de integração: 3 backends, subagentes distribuídos e estáveis, fallback em queda e regressão zero com o modo desligado. É o gate de aceite antes da validação manual no hardware real.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. Cada decisão de roteamento DEVE gerar log `[proxy] route ...` com os pares `chave=valor` definidos em "Monitoramento e Observabilidade" do TechSpec (external_model, internal_model, backend, gpu, affinity_key, sticky_hit, reason, stream, prompt_tokens_estimated).
- 2. Nova sessão e fallback/reassign DEVEM gerar os registros distintos definidos no TechSpec (`new sticky session ...`, `backend <porta> unavailable ... reason=backend_down`) — PRD F11.
- 3. Os logs DEVEM usar o logger `automanager` existente (formato `%(asctime)s - %(levelname)s - %(message)s`, `manager.log`); erros de roteamento em nível `ERROR`, decisões em `INFO`.
- 4. O cenário de integração DEVE reproduzir os critérios de aceite do PRD: 3 subagentes tagueados distribuídos entre 3 backends fake, cada um estável no seu backend por >= 3 requisições consecutivas.
- 5. A queda de um backend fake DEVE produzir reassign único com log correspondente e continuidade da sessão no novo backend.
- 6. Uma execução completa da suíte com `smart_proxy.enabled=false` DEVE provar comportamento idêntico ao atual (regressão zero).
- 7. Nenhum log DEVE vazar conteúdo de mensagens do usuário (apenas metadados e contadores).
</requirements>

## Subtarefas
- [x] 9.1 Implementar os três formatos de log (`route`, `new sticky session`, `reassigned`/`unavailable`) nos pontos de decisão do `ProxyRouter`/handler.
- [x] 9.2 Incluir o nome da GPU no log de decisão a partir dos dados da instância.
- [x] 9.3 Construir a infraestrutura de teste com 3 backends fake SSE (mini-apps FastAPI ou `httpx.MockTransport`).
- [x] 9.4 Escrever os testes de integração do cenário PRD (distribuição, estabilidade sticky, fallback, transparência do model).
- [x] 9.5 Escrever o teste de regressão com modo desligado e validar a suíte completa do projeto.

## Detalhes de Implementação
Ver "Monitoramento e Observabilidade" do TechSpec (formatos exatos) e PRD "Métricas de Sucesso" (comportamentos a provar). Logging central: `log_manager.py:50-73` (logger `automanager`, RotatingFileHandler); exemplos de uso no proxy atual: `llama_manager.py:597,695`. Testes com `caplog` do pytest para asserção de mensagens.

### Arquivos Relevantes
- `proxy_router.py` — pontos de log de decisão/sessão/reassign
- `llama_manager.py` — logs no encaminhamento e erros
- `tests/unit/test_smart_proxy_routes.py` / `tests/integration/` — cenários ponta a ponta

### Arquivos Dependentes
- `log_manager.py` — configuração de logging existente (não alterar formato global)
- `tests/integration/conftest.py` — fixtures de integração existentes

### ADRs Relacionados
- [ADR-001: Least-busy sticky](../adrs/adr-001.md) — `reason` logado por decisão
- [ADR-006: Reescrita SSE](../adrs/adr-006.md) — transparência validada ponta a ponta

## Entregáveis
- Logs `[proxy]` completos nos três eventos (decisão, nova sessão, fallback)
- Suíte de integração cobrindo os critérios de aceite do PRD executável em CI local
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração ponta a ponta do cenário 3 backends **(OBRIGATÓRIO)**

## Testes
- Testes unitários (caplog):
  - [x] Decisão roteada gera 1 linha `[proxy] route` contendo todos os 9 pares chave=valor do TechSpec
  - [x] Primeira requisição de uma tag gera adicionalmente `[proxy] new sticky session ... reason=...`
  - [x] Queda de backend gera `[proxy] backend <porta> unavailable` + `[proxy] reassigned ... old_backend=... new_backend=... reason=backend_down`
  - [x] Nenhuma linha de log contém o conteúdo das mensagens enviadas (asserção negativa sobre payload sentinela)
- Testes de integração (3 backends fake):
  - [x] 3 subagentes (`delphi-auditor`, `sql-reviewer`, `test-writer`) + main → main no principal, subagentes distribuídos; 3 rodadas seguidas mantêm cada sessão no mesmo backend
  - [x] Todas as respostas (stream e não-stream) apresentam `model` do principal e headers `x-automanager-backend*` coerentes com a porta real
  - [x] Derrubar o backend fake de `sql-reviewer` → próxima requisição da tag responde 200 por outro backend com log de reassign
  - [x] Suíte legada completa (`tests/unit/`) verde com `smart_proxy.enabled=false`
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando (incluindo suíte legada completa)
- Cobertura de testes >= 80%
- Logs permitem reconstruir toda decisão de roteamento sem depurador (modelo externo/interno, backend, motivo)
- Critérios de aceite automatizáveis do PRD comprovados por teste
