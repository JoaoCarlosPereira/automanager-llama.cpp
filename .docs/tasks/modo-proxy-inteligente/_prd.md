# PRD — Modo Proxy Inteligente

## Visão Geral

O Automanager já gerencia múltiplas instâncias de llama-server, cada uma tipicamente dedicada a uma GPU (ex.: RTX 3090 + Tesla P100 #0 + Tesla P100 #1). Hoje, clientes OpenAI-compatible precisam conhecer e endereçar cada modelo individualmente, e o hardware secundário fica ocioso quando um cliente como o Cursor dispara subagentes paralelos — todas as requisições disputam a mesma instância.

O **Modo Proxy Inteligente** faz o Automanager expor externamente apenas o **modelo principal** definido pelo administrador, enquanto usa automaticamente as demais instâncias online como **backends secundários** para absorver requisições paralelas. O cliente externo (Cursor, Continue, Cline ou qualquer cliente OpenAI-compatible) enxerga um único modelo; internamente, cada conversa/subagente é atribuída a um backend e **permanece presa a ele** (afinidade sticky), preservando KV cache, cache de prompt e consistência.

**Para quem**: administradores de servidores de inferência local com múltiplas GPUs heterogêneas e desenvolvedores que consomem a API via clientes de código com subagentes paralelos.

**Valor**: aproveitamento total do hardware disponível sem nenhuma mudança no cliente — configuração de um único modelo no Cursor, throughput multiplicado quando há subagentes.

## Objetivos

- Permitir que N instâncias online atendam, de forma transparente, o tráfego dirigido ao modelo principal.
- Garantir que 100% das requisições de uma mesma conversa/subagente sejam atendidas pelo mesmo backend enquanto a sessão durar (exceto fallback por indisponibilidade).
- Manter compatibilidade integral com clientes OpenAI-compatible: o nome do modelo na resposta (incluindo streaming) é sempre o do modelo principal.
- Zero configuração no cliente além de Base URL, modelo principal e API key.
- Dar ao administrador visibilidade completa: quais sessões existem, em qual GPU/modelo cada uma está, e por que cada decisão de roteamento foi tomada.

## Histórias de Usuário

**Persona 1 — Administrador do Automanager**

- Como administrador, quero ativar/desativar o Modo Proxy Inteligente por um checkbox no painel, para alternar entre o comportamento atual e o roteamento automático sem editar arquivos.
- Como administrador, quero marcar uma instância como "modelo principal" (apenas uma por vez), para definir qual modelo é exposto externamente.
- Como administrador, quero excluir uma instância específica do proxy mesmo com o modo ligado, para reservá-la a outros usos.
- Como administrador, quero ver no painel as sessões sticky ativas, seus backends/GPUs, tags detectadas, contagem de requisições e último uso, para entender como a carga está distribuída.
- Como administrador, quero limpar ou reatribuir sessões manualmente e habilitar/desabilitar backends, para intervir quando necessário.
- Como administrador, quero testar o roteamento de uma requisição sem executar inferência (`/proxy/resolve`), para depurar a extração de afinidade e a escolha de backend.

**Persona 2 — Desenvolvedor usando Cursor (ou cliente OpenAI-compatible)**

- Como desenvolvedor, quero configurar apenas o modelo principal no Cursor e disparar subagentes paralelos, para que o Automanager distribua a carga entre GPUs sem que eu saiba dos secundários.
- Como desenvolvedor, quero que cada subagente identifique-se por uma tag `[AGENT:nome]` no system prompt, para que cada um mantenha seu próprio backend estável durante toda a execução.
- Como desenvolvedor, quero continuar podendo chamar uma instância específica pelo nome real do modelo, para casos em que preciso de um backend determinado.

## Funcionalidades Principais

### F1. Ativação global do modo (checkbox)

Checkbox "Ativar Modo Proxy Inteligente" no painel. **Desligado** (padrão): a API `/v1` funciona exatamente como hoje — cada instância responde individualmente pelo nome do seu modelo, `/v1/models` agrega todos. **Ligado**: o `/v1` passa a operar como proxy inteligente conforme F2–F12. A alternância tem efeito imediato, sem reiniciar instâncias.

### F2. Definição do modelo principal

Controle por instância "Definir como modelo principal". Somente uma instância pode ser principal por vez (marcar uma desmarca a anterior). A instância principal define o nome de modelo exposto externamente. O modo proxy não pode operar sem um principal definido e online; nesse caso a UI orienta o administrador e a API responde com erro claro.

### F3. Participação por instância no proxy

Checkbox por instância "Usar como backend secundário no proxy" (padrão: ligado). Instâncias com a opção desligada permanecem abertas e endereçáveis pelo nome real, mas nunca recebem tráfego roteado. Estados possíveis de um backend: `online`, `offline`, `busy`, `degraded`, `disabled` (desabilitado pelo admin), `not_eligible` (fora do proxy). Novas sessões nunca são criadas em backends `offline`, `disabled` ou `not_eligible`.

### F4. Exposição externa unificada

- `GET /v1/models` com o modo ligado retorna **somente o modelo principal** (ADR-003).
- Toda resposta ao cliente — incluindo cada evento SSE em streaming — apresenta o campo `model` com o nome do modelo principal, mesmo quando a requisição foi atendida por um secundário.
- Requisições que pedem um modelo secundário pelo nome real continuam atendidas diretamente por aquela instância, sem sticky nem reescrita (ADR-003).

### F5. Afinidade sticky por conversa/subagente

Cada requisição ao modelo principal gera uma `affinity_key`, extraída nesta ordem de precedência:

1. Header `x-automanager-session-id`;
2. Header `x-automanager-agent-id`;
3. `metadata.session_id` / `metadata.agent_id` no corpo JSON;
4. Tag `[AGENT:nome]` no conteúdo das mensagens;
5. Fallback: hash estável de (primeiro system prompt + primeira mensagem user + modelo externo + IP do cliente + User-Agent).

Nunca são usados valores voláteis (timestamp, request id) na chave. A tabela de afinidade registra: `affinity_key`, backend, modelo externo/interno, tag detectada, criação, último uso e contagem de requisições. Sessões expiram por TTL de inatividade (padrão 180 min, configurável) e são persistidas em disco (`data/proxy_sessions.json`), sobrevivendo a restart do manager.

### F6. Seleção automática de backend (ADR-001, ADR-002)

- Sessão existente → **sempre** o mesmo backend (proibido round-robin por mensagem).
- Nova sessão da conversa principal (`[AGENT:main]` ou sem tag com cara de conversa principal) → prefere o backend principal.
- Novo subagente → backend elegível **menos ocupado** (contagem interna de requisições em andamento).
- Restrições respeitadas na escolha: o backend deve ter **capacidade de contexto** suficiente para a requisição (estimativa de tokens; caso nenhum secundário comporte, usar o principal) e estar abaixo do seu **limite de concorrência** (`max_parallel_requests`, configurável por instância; padrão 1).
- A tag `[AGENT:...]` não define backend preferido — serve apenas como chave de afinidade (ADR-002).

### F7. Fallback e resiliência

- Backend de uma sessão caiu ou foi desabilitado → a sessão é reatribuída **uma vez** ao melhor backend disponível, com log do motivo.
- Backend ocupado: sessão já sticky aguarda em fila curta (espera máxima configurável, padrão 30 s); sessão nova escolhe outro backend; sem nenhum disponível, erro controlado no formato OpenAI.
- Fallback entre modelos diferentes é permitido (a resposta segue nomeando o principal).

### F8. Streaming e fidelidade de requisição

- `stream: true` é encaminhado como SSE ponta a ponta, sem materializar a resposta em memória, mantendo formato compatível com OpenAI/Cursor (incluindo `data: [DONE]`).
- Todos os parâmetros da requisição são preservados no encaminhamento (`messages`, `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `stop`, `tools`, `tool_choice`, `response_format`, penalidades, `seed`, campos específicos do llama.cpp e campos desconhecidos). Apenas o campo `model` é reescrito para o nome interno do backend escolhido. Tags `[AGENT:...]` não são removidas do conteúdo por padrão.

### F9. Painel "Proxy Inteligente"

Nova seção no dashboard mostrando: estado do modo (ON/OFF), modelo exposto, lista de backends (papel principal/secundário, modelo, GPU, status, requisições ativas) e sessões sticky ativas (chave/tag, backend/GPU, contagem de requisições, último uso, tokens processados quando disponível). Atualização junto ao polling de status existente.

### F10. Endpoints administrativos

| Endpoint | Função |
|----------|--------|
| `GET /proxy/status` | Estado geral do proxy (modo, principal, backends, contadores) |
| `GET /proxy/sessions` | Lista sessões sticky ativas |
| `DELETE /proxy/sessions` | Limpa todas as sessões |
| `DELETE /proxy/sessions/{affinity_key}` | Remove uma sessão |
| `POST /proxy/sessions/{affinity_key}/reassign` | Força reatribuição de backend |
| `GET /proxy/backends` | Lista backends e estados |
| `POST /proxy/backends/{id}/enable` / `disable` | Habilita/desabilita backend no proxy |
| `POST /proxy/resolve` | Simula o roteamento de uma requisição sem chamar o modelo, retornando tag detectada, `affinity_key`, backend escolhido, modelo interno, motivo e `sticky_hit` |

Endpoints administrativos exigem a autenticação do painel/API existente.

### F11. Logs de roteamento

Cada decisão gera log estruturado com: modelo externo, modelo interno, backend, GPU, `affinity_key`, `sticky_hit`, motivo (`agent_tag`, `least_busy`, `main_preference`, `backend_down`...), streaming e tokens estimados. Eventos de nova sessão, hit sticky e fallback/reassignment têm registros distintos e legíveis.

## Experiência do Usuário

**Fluxo 1 — Ativação (administrador)**: abre o painel → marca a instância desejada como "Principal" → confere quais instâncias participam do proxy (checkbox por instância) → liga "Ativar Modo Proxy Inteligente" → a seção "Proxy Inteligente" passa a exibir modelo exposto e backends ativos.

**Fluxo 2 — Uso no Cursor (desenvolvedor)**: configura Base URL `http://SERVIDOR:8000/v1`, modelo = nome do principal, API key do Automanager → usa normalmente; ao disparar subagentes paralelos com tags `[AGENT:...]` no system prompt, cada um é distribuído a uma GPU e ali permanece. O Cursor exibe sempre o modelo principal.

**Fluxo 3 — Monitoramento e intervenção**: administrador acompanha sessões e distribuição por GPU no painel; pode desabilitar um backend (sessões dele são reatribuídas), remover sessões ou forçar reassignment; usa `POST /proxy/resolve` para depurar por que uma requisição cai em determinado backend.

**Acessibilidade/UI**: os novos controles seguem o padrão visual existente do dashboard (checkboxes por aba de modelo, seção de status com polling), com textos em linguagem clara sobre efeitos de cada opção.

## Restrições Técnicas de Alto Nível

- Compatibilidade estrita com o protocolo OpenAI (incluindo SSE) — Cursor, Continue e Cline devem funcionar sem ajustes.
- O proxy atende no `/v1` existente da porta 8000; nenhuma porta nova (ADR-003).
- Com o modo desligado, o comportamento atual da API permanece byte-a-byte inalterado.
- A decisão de roteamento não pode adicionar latência perceptível (sem health-checks bloqueantes por requisição).
- Sessões sticky persistem em disco e sobrevivem a restart do manager; segredos não são gravados na tabela de sessões.
- Limites por backend (contexto máximo e concorrência) são respeitados em toda decisão.

## Fora de Escopo (Non-Goals)

- Regras configuráveis tag→backend preferido (ADR-002; pode ser reavaliado futuramente).
- Porta/listener dedicado para o proxy (ADR-003).
- Balanceamento por mensagem individual (round-robin) — explicitamente proibido.
- Roteamento cache-aware consultando `/slots` do llama.cpp (ADR-001).
- Auto-início/auto-scaling de instâncias pelo proxy (o proxy usa apenas instâncias já abertas).
- Novo mecanismo de autenticação (usa o existente).
- Múltiplos grupos de proxy / múltiplos modelos principais simultâneos.
- Fila persistente de requisições com prioridades.

## Plano de Entrega por Fases

Por decisão do administrador, a entrega é em **fase única** contendo F1–F11. Marcos internos de validação (não são releases separados):

1. **Roteamento núcleo**: modo liga/desliga, principal definido, sticky + least-busy + reescrita de `model` (com streaming) funcionando via API.
2. **Resiliência e administração**: TTL, persistência, fallback, limites de contexto/concorrência e endpoints `/proxy/*`.
3. **Interface**: controles por instância, seção "Proxy Inteligente" e logs finais.

Critério de conclusão da fase: todos os itens de "Métricas de Sucesso" verificados com o cenário real de 3 instâncias (3090 + 2× P100) e Cursor com subagentes paralelos.

## Métricas de Sucesso

- Cursor configurado apenas com o modelo principal opera normalmente com o modo ligado, incluindo streaming.
- Com 3 subagentes paralelos tagueados, as requisições são distribuídas entre os 3 backends e cada subagente permanece 100% do tempo no mesmo backend (verificável em `/proxy/sessions` e logs).
- `GET /v1/models` retorna apenas o principal com o modo ligado; a resposta de qualquer requisição roteada exibe o nome do principal.
- Nenhuma nova sessão é criada em backend offline/desabilitado/não elegível.
- Sessões expiram pelo TTL configurado e sobrevivem a restart do manager.
- `POST /proxy/resolve` reproduz fielmente a decisão de roteamento sem executar inferência.
- Painel exibe em tempo real backends, sessões e distribuição por GPU.
- Desligar o modo restaura integralmente o comportamento atual.

## Riscos e Mitigações

- **Clientes sem identificador estável de sessão**: se o Cursor não enviar tags/headers, o fallback por hash pode agrupar ou separar conversas incorretamente. Mitigação: ordem de extração com múltiplas fontes, `/proxy/resolve` para diagnóstico e documentação de uso das tags `[AGENT:...]`.
- **Percepção de lentidão heterogênea**: subagentes roteados às P100 respondem mais devagar que o principal na 3090, podendo parecer inconsistência ao usuário. Mitigação: painel mostra qual GPU atende cada sessão; conversa principal sempre prefere a GPU mais forte.
- **Mudanças de comportamento do Cursor** (formato de subagentes/metadata em versões futuras). Mitigação: extração de afinidade em camadas independentes do cliente.
- **Adoção**: administrador esquecer de definir o principal ou deixar instâncias fora do pool sem perceber. Mitigação: UI orienta quando o modo está ligado sem principal e sinaliza instâncias não elegíveis.

## Registros de Decisão de Arquitetura

- [ADR-001: Seleção de backend por least-busy interno com afinidade sticky](adrs/adr-001.md) — novas sessões vão ao backend elegível menos ocupado, contado pelo próprio manager; rejeitados hash consistente e slot-aware.
- [ADR-002: Tag [AGENT:...] apenas como chave de afinidade](adrs/adr-002.md) — sem regras fixas tag→backend; escolha de backend sempre automática.
- [ADR-003: Proxy assume o /v1 existente (porta 8000) expondo somente o principal](adrs/adr-003.md) — sem porta nova; `/v1/models` lista só o principal; secundários seguem acessíveis pelo nome real.

## Perguntas em Aberto

- Valores padrão a confirmar na TechSpec: TTL de 180 min, espera máxima de 30 s em backend ocupado, `max_parallel_requests` = 1 por backend (2 para a 3090?).
- Heurística de "parece conversa principal" para requisições sem tag: critérios exatos ficam para a TechSpec.
- Exposição do modelo real em header auxiliar da resposta (ex.: `x-automanager-backend`) para telemetria — prática comum no mercado; incluir?
- Contabilização de "tokens processados" por sessão no painel depende do que os backends reportam; nível de detalhe a definir.
