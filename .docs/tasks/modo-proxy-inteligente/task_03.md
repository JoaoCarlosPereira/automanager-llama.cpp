---
status: completed
title: "ProxyRouter: elegibilidade, seleção least-busy, contadores e reassign"
type: backend
complexity: high
dependencies:
  - task_02
---

# ProxyRouter: elegibilidade, seleção least-busy, contadores e reassign

## Visão Geral
Completa o `proxy_router.py` com a metade "decisão": contadores de requisições em andamento por porta, cálculo de elegibilidade (online, participação, contexto, concorrência), seleção least-busy com preferência do principal para a conversa main, espera por backend ocupado, reassignment único em queda e disable runtime. Implementa o coração do ADR-001.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. `resolve()` DEVE retornar o backend sticky existente quando houver sessão válida (sticky hit), sem trocá-lo por carga — round-robin por mensagem é proibido (PRD F6).
- 2. Backend elegível para NOVA sessão DEVE satisfazer: instância `running`, `proxy_eligible=true`, não desabilitado em runtime, `tokens_estimados*1.1 ≤ context_size // max(1, parallel_slots)` e `in_flight < max_parallel_requests` (seção "Seleção" do TechSpec).
- 3. Conversa main (tag ausente ou `[AGENT:main]`) DEVE preferir o backend principal; se o principal estiver ocupado DEVE aguardar a fila curta; se offline/desabilitado, DEVE cair para least-busy entre secundários.
- 4. Subagente (tag ≠ main) DEVE ir para o backend elegível menos ocupado (empate: menor porta), podendo incluir o principal.
- 5. A estimativa de tokens DEVE ser `len(json.dumps(messages, ensure_ascii=False)) // 4` com margem de 10%; se nenhum secundário comportar o contexto, DEVE usar o principal.
- 6. A espera por backend ocupado DEVE usar polling `asyncio.sleep(0.25)` limitado a `max_wait_seconds`; ao estourar: sessão sticky → `ProxyError` 503 formato OpenAI; sessão nova → tentar próximo elegível antes do erro.
- 7. Sessão cujo backend caiu/foi desabilitado DEVE ser reatribuída UMA vez ao melhor backend disponível com `reason=backend_down` (PRD F7); `reassign()` administrativo DEVE forçar nova seleção.
- 8. `acquire(port)`/`release(port, usage)` DEVEM manter os contadores corretos mesmo com exceções (uso em `finally` pelo chamador); `release` DEVE acumular `tokens_processed` best-effort a partir de `usage`.
- 9. `resolve(dry_run=True)` NÃO DEVE criar sessão nem alterar contadores.
- 10. Sem principal definido/online com o modo ativo, `resolve()` DEVE levantar `ProxyError` com mensagem clara (PRD F2).
</requirements>

## Subtarefas
- [x] 3.1 Implementar contadores in-flight por porta com `acquire`/`release` e acumulação de `usage`.
- [x] 3.2 Implementar cálculo de elegibilidade (estado, participação, contexto por slot, concorrência) sobre `get_status()` + config por modelo.
- [x] 3.3 Implementar a seleção de nova sessão (main→principal, subagente→least-busy, restrições e fallbacks) e o registro da sessão criada.
- [x] 3.4 Implementar espera por backend ocupado com timeout e a semântica sticky vs nova sessão.
- [x] 3.5 Implementar reassignment (automático 1x em queda + administrativo) e disable/enable runtime por porta.
- [x] 3.6 Escrever testes unitários cobrindo todas as regras de seleção, espera e reassign.

## Detalhes de Implementação
Ver seções "Interfaces Principais", "Seleção (nova sessão)" e "Espera por backend ocupado" do TechSpec, e o algoritmo do fluxo de dados na "Visão dos Componentes". Instâncias vêm de `get_status()["instances"]` (estrutura em `process_manager.py:666-673`, com `config` = `StartRequest` completo incluindo `context_size`, `parallel_slots`, `gpu_weights`). Flags por modelo via `config_manager.lookup_model_config`/`get_config()` (tarefa 01).

### Arquivos Relevantes
- `proxy_router.py` — extensão do módulo criado na tarefa 02
- `process_manager.py` — contrato de `get_status()` (`:652-677`) consumido via injeção
- `config_manager.py` — flags por modelo e `smart_proxy` (tarefa 01)

### Arquivos Dependentes
- `llama_manager.py` — chamará `resolve`/`acquire`/`release` (tarefa 04) e `reassign`/`set_backend_enabled` (tarefa 06)
- `tests/unit/test_proxy_router.py` — mesmo arquivo de testes da tarefa 02, estendido

### ADRs Relacionados
- [ADR-001: Least-busy interno com afinidade sticky](../adrs/adr-001.md) — estratégia de seleção e contadores internos
- [ADR-004: Módulo único proxy_router.py](../adrs/adr-004.md) — dependências por injeção
- [ADR-005: Estado sticky](../adrs/adr-005.md) — semântica de reassign e identificador durável

## Entregáveis
- `ProxyRouter.resolve()` completo com sticky hit, seleção, espera, fallback e `dry_run`
- Contadores in-flight e disable runtime funcionais
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração de concorrência (resoluções simultâneas via asyncio) **(OBRIGATÓRIO)**

## Testes
- Testes unitários (com `get_status` e config mockados — 3 instâncias: principal ctx 65536/1 slot, 2 secundárias):
  - [x] Sticky hit: 2ª chamada com mesma affinity_key retorna o mesmo backend mesmo com outro backend menos ocupado
  - [x] Nova sessão main (sem tag) vai ao principal; com principal `in_flight=1` e `max_parallel=1`, espera e obtém o principal quando `release()` ocorre dentro do timeout
  - [x] Nova sessão `[AGENT:sql-reviewer]` com cargas (1,0,1) vai ao backend com 0; empate (0,0) escolhe a menor porta
  - [x] Requisição com estimativa de 60k tokens não elege secundário com `context_size=32768`; cai no principal com `reason` adequado
  - [x] Backend com `proxy_eligible=false` e backend desabilitado via `set_backend_enabled(port, False)` nunca recebem novas sessões
  - [x] Sessão sticky com backend morto (porta ausente do get_status) é reatribuída exatamente 1x com `reason=backend_down`
  - [x] Timeout de espera: sticky em backend ocupado por > `max_wait_seconds` recebe `ProxyError` 503 com corpo formato OpenAI
  - [x] `dry_run=True` não cria sessão nem incrementa contador
  - [x] Modo ativo sem `primary_model_path` online → `ProxyError` com mensagem clara
- Testes de integração:
  - [x] 10 `resolve()` concorrentes (asyncio.gather) de 3 tags distintas distribuem entre os 3 backends respeitando `max_parallel_requests` e nenhuma sessão troca de backend
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Nenhum cenário de teste produz troca de backend para affinity_key existente (exceto reassign explícito/queda)
- Decisão de roteamento sem I/O de rede (apenas memória/config)
