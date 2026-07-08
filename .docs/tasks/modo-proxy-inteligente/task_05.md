---
status: completed
title: Reescrita SSE por linha e headers de telemetria
type: backend
complexity: medium
dependencies:
  - task_04
---

# Reescrita SSE por linha e headers de telemetria

## Visão Geral
Implementa a transparência do streaming (ADR-006): um gerador com buffer incremental de bytes que emite linhas completas, reescrevendo o campo `model` de eventos `data: {json}` para o nome do modelo principal quando o backend é secundário, sem materializar a resposta. Adiciona os headers `x-automanager-backend`/`x-automanager-backend-model` e alimenta `tokens_processed` da sessão a partir do evento final com `usage`.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- 1. O gerador DEVE bufferizar bytes e emitir somente linhas completas (delimitadas por `\n`), garantindo reescrita correta mesmo quando o nome do modelo é cortado entre dois chunks TCP (ADR-006).
- 2. Somente linhas `data: {...}` com JSON válido contendo `"model"` DEVEM ser reescritas; `data: [DONE]`, comentários keep-alive (`:`), linhas vazias e JSON inválido DEVEM passar intocados (fail-open).
- 3. Quando o backend decidido é o próprio principal, o streaming DEVE repassar chunks brutos como hoje (sem custo de parse) — `RouteDecision.rewrite=False`.
- 4. Toda resposta roteada pelo proxy (stream e não-stream) DEVE incluir os headers `x-automanager-backend: <porta>` e `x-automanager-backend-model: <modelo interno>`.
- 5. O evento SSE final contendo `usage`/`timings` DEVE alimentar `tokens_processed` da sessão via `release(port, usage=...)`, best-effort (ausência não é erro).
- 6. O buffer residual (bytes sem `\n` final ao encerrar o stream) DEVE ser emitido no fechamento para não truncar a resposta.
- 7. A reescrita NÃO DEVE reter mais que a linha corrente em memória (streaming ponta a ponta, PRD F8).
</requirements>

## Subtarefas
- [x] 5.1 Implementar o gerador de streaming com buffer incremental por linha e reescrita condicional do campo `model`.
- [x] 5.2 Integrar o gerador ao caminho de streaming do desvio criado na tarefa 04 (substituindo o repasse bruto quando `rewrite=True`).
- [x] 5.3 Adicionar os headers `x-automanager-backend*` às respostas roteadas (stream e não-stream).
- [x] 5.4 Extrair `usage` do evento final e repassar ao `release()` para acumular `tokens_processed`.
- [x] 5.5 Escrever testes unitários com fixtures SSE reais e cortes de chunk adversariais.

## Detalhes de Implementação
Ver ADR-006 e seção "Interfaces Principais" do TechSpec. Streaming atual repassa `aiter_bytes` bruto (`llama_manager.py:670-677`); headers de resposta filtrados por `_filter_proxy_headers` (`llama_manager.py:79-85` — já remove `content-length`, o que acomoda corpos reescritos). O gerador pode viver em `proxy_router.py` ou `llama_manager.py`; preferir função pura testável sem FastAPI (ex.: `rewrite_sse_stream(aiter, external_model) -> AsyncIterator[bytes]`).

### Arquivos Relevantes
- `llama_manager.py` — caminho de streaming do desvio (tarefa 04)
- `proxy_router.py` — `release(usage=...)` e possivelmente a função de reescrita

### Arquivos Dependentes
- `tests/unit/test_smart_proxy_routes.py` — casos de streaming ponta a ponta
- `tests/unit/test_proxy_router.py` — testes da função pura de reescrita (se residir no router)

### ADRs Relacionados
- [ADR-006: Reescrita SSE por linha + headers de telemetria](../adrs/adr-006.md) — decisão implementada integralmente por esta tarefa

## Entregáveis
- Streaming SSE transparente (model = principal) para backends secundários
- Headers de telemetria em todas as respostas roteadas
- `tokens_processed` acumulado por sessão quando `usage` disponível
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração de streaming via TestClient com backend fake **(OBRIGATÓRIO)**

## Testes
- Testes unitários (função de reescrita alimentada por chunks controlados):
  - [x] Evento `data: {"model":"AUX.gguf",...}` inteiro em um chunk → sai com `"model":"principal.gguf"`
  - [x] Nome do modelo cortado entre dois chunks (`data: {"model":"AU` + `X.gguf",...}\n\n`) → reescrita correta após a linha completar
  - [x] `data: [DONE]`, linha keep-alive `: ping` e linha vazia passam byte-a-byte intocados
  - [x] Linha `data:` com JSON inválido passa intocada (fail-open, sem exceção)
  - [x] Chunk final sem `\n` terminal é emitido no fechamento (sem truncamento)
  - [x] Evento final com `usage.total_tokens=123` → `release` chamado com o usage e sessão acumula 123
- Testes de integração:
  - [x] POST `/v1/chat/completions` `stream=true` roteado a secundário fake que emite 5 chunks SSE → cliente recebe SSE válido, todos os eventos com `model` do principal, terminando em `data: [DONE]`, e headers `x-automanager-backend*` presentes
  - [x] Mesmo fluxo com backend = principal → bytes idênticos aos emitidos pelo backend (sem reescrita)
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Nenhum evento SSE corrompido sob qualquer divisão de chunks testada
- Cliente streaming nunca vê nome de modelo interno
