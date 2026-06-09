# PRD: Maximização de VRAM por Prioridade no Auto-Balance

## Visão Geral

O Auto-Balance distribui automaticamente um modelo de IA entre as GPUs e, em último caso, a RAM/CPU da máquina. Hoje, mesmo havendo VRAM livre, parte da carga é colocada na CPU/RAM ou distribuída proporcionalmente entre as GPUs — o que reduz a velocidade de inferência e desperdiça o hardware mais rápido disponível.

Esta funcionalidade redefine o comportamento de alocação para um **preenchimento estrito por prioridade**: a GPU principal é preenchida até o limite antes de qualquer outra GPU receber carga, e a RAM/CPU só é usada quando todas as GPUs habilitadas estão cheias. O objetivo é maximizar o uso de VRAM antes de recorrer a RAM/CPU.

- **Problema que resolve:** offload prematuro para CPU/RAM e distribuição proporcional, que degradam o desempenho.
- **Para quem é:** operadores da aplicação que rodam modelos locais em máquinas com uma ou mais GPUs (inclusive heterogêneas, ex.: RTX 3090 + P100).
- **Por que é valioso:** inferência mais rápida, uso previsível do hardware e ausência de surpresas (carga indo para CPU sem necessidade).

## Objetivos

- Garantir que **nenhuma carga vá para RAM/CPU enquanto houver VRAM ociosa** acima do limite em qualquer GPU habilitada.
- Preencher cada GPU até **98%** de ocupação antes de usar a próxima, sempre na ordem de prioridade.
- Eliminar qualquer distribuição **proporcional** entre GPUs ou entre GPU e CPU.
- Reproduzir os cenários A–D (20/30/50/70 GB) com a distribuição exata esperada em testes automatizados.
- Tornar o comportamento de alocação determinístico e protegido contra regressão (fonte única da verdade).

## Histórias de Usuário

- **Como** operador com uma GPU potente (3090) e modelo que cabe nela, **quero** que o modelo fique 100% na 3090, **para que** eu obtenha a maior velocidade possível sem offload desnecessário.
- **Como** operador com GPUs heterogêneas, **quero** que a aplicação encha a GPU principal e depois a próxima por ordem de prioridade, **para que** o desempenho seja previsível e máximo.
- **Como** operador, **quero** que a RAM/CPU só seja usada quando todas as GPUs estiverem cheias, **para que** eu não perca velocidade por offload prematuro.
- **Como** operador com a CPU desligada e um modelo grande demais, **quero** ser avisado claramente de que o hardware é insuficiente, **para que** eu decida ligar a CPU ou reduzir o contexto, em vez de receber um erro confuso.
- **Como** operador, **quero** que ativar o Auto-Balance assuma todo o controle da distribuição, **para que** eu não precise gerenciar pesos fixados manualmente nesse modo.

## Funcionalidades Principais

1. **Cascata estrita por prioridade (núcleo)**
   - Ordena as GPUs: principal selecionada em 1º; demais habilitadas por índice crescente.
   - Preenche a GPU de maior prioridade até o limite de ocupação antes de passar à próxima.
   - Se o restante do modelo cabe inteiramente na GPU atual, carrega tudo e encerra; caso contrário, enche até o limite, subtrai o carregado e segue para a próxima.

2. **Limite de ocupação de VRAM fixo em 98%**
   - Cada GPU é considerada "cheia" ao atingir 98% da sua VRAM total (margem de segurança para overhead/KV-cache).

3. **RAM/CPU como último recurso**
   - Só recebe carga quando todas as GPUs habilitadas estão cheias.
   - Nunca recebe carga havendo VRAM ociosa acima do limite.

4. **Bloqueio com alerta quando CPU desligada e modelo não cabe**
   - Não inicia o servidor; exibe alerta de hardware insuficiente (modelo, VRAM total, contexto, GPUs testadas) com sugestões (ligar CPU, reduzir contexto).

5. **Auto-Balance limpa os pins**
   - Ao ativar o Auto-Balance, os pesos fixados (pin) são limpos/desabilitados na interface, deixando claro que a cascata controla toda a distribuição.

## Experiência do Usuário

1. O operador seleciona o modelo, escolhe a GPU principal e habilita as GPUs/CPU desejadas.
2. Ativa o Auto-Balance — os pins são limpos automaticamente.
3. Ao iniciar, a aplicação calcula a distribuição por prioridade e inicia o servidor.
4. No painel de métricas, o operador vê a GPU principal ocupada até ~98%, depois as secundárias, e a CPU/RAM apenas se necessário.
5. Se a CPU estiver desligada e o modelo não couber, o operador recebe um alerta claro em vez de uma falha silenciosa.

**Resultados esperados (cenários de referência):**

| Cenário | Modelo | 3090 (23,5 GB úteis) | P100 #1 (15,7) | P100 #2 (15,7) | CPU |
|---------|--------|----------------------|----------------|----------------|-----|
| A | 20 GB | 20 | 0 | 0 | 0 |
| B | 30 GB | 23,5 | 6,5 | 0 | 0 |
| C | 50 GB | 23,5 | 15,7 | 10,8 | 0 |
| D | 70 GB | 23,5 | 15,7 | 15,7 | 15,1 |

## Restrições Técnicas de Alto Nível

- Integração obrigatória com o fluxo existente de início do `llama-server` e com a detecção de GPUs/VRAM.
- O comportamento deve respeitar a lista de **GPUs habilitadas** e a **GPU principal** selecionadas pelo usuário.
- Meta de desempenho na perspectiva do usuário: maximizar VRAM antes de RAM/CPU em todos os cenários.

## Fora de Escopo (Non-Goals)

- Expor o limite de ocupação de VRAM na interface (global ou por GPU) — permanece fixo em 98%.
- Painel de simulação/preview da distribuição antes de iniciar.
- Otimizações específicas para modelos MoE (offload seletivo de tensores).
- Ligar a CPU automaticamente quando desligada (decisão: bloquear e avisar).
- Manter pins ativos durante o Auto-Balance.
- Balanceamento dinâmico em tempo de execução (re-distribuição após o servidor iniciar).

## Plano de Entrega por Fases

### MVP (Fase 1)
- Cascata estrita por prioridade como contrato único de alocação.
- Limite fixo de 98%; RAM/CPU como último recurso; zero offload com VRAM ociosa.
- Bloqueio com alerta quando CPU desligada e modelo não cabe.
- Auto-Balance limpa pins.
- Critério para avançar: cenários A–D validados e ausência de offload com VRAM ociosa confirmada no painel.

### Fase 2 (futuro, se necessário)
- Testes de borda adicionais (GPUs com VRAM muito desiguais, múltiplas P100, falhas de detecção).
- Critério: estabilidade em produção sem regressões reportadas.

### Fase 3 (futuro, condicionado a demanda)
- Possível exposição do limite de ocupação na UI (global) — somente se houver demanda real.

## Métricas de Sucesso

- **VRAM cheia antes de CPU:** nos cenários A–D, GPUs ocupadas até ~98% e CPU só com todas as GPUs cheias (verificável no painel).
- **Zero offload com VRAM ociosa:** nenhuma carga em CPU/RAM enquanto qualquer GPU habilitada tiver VRAM livre acima do limite.
- **Cobertura de testes:** cenários A–D (20/30/50/70 GB) reproduzidos em testes automatizados com distribuição exata.
- **Previsibilidade:** mesma entrada produz sempre a mesma distribuição.

## Riscos e Mitigações

- **Adoção:** operadores acostumados a ajustar pesos manualmente podem estranhar a limpeza de pins no Auto-Balance. *Mitigação:* mensagem clara na UI explicando o comportamento.
- **Expectativa de desempenho:** modelos que exigem CPU (cenário D) ainda terão queda de velocidade. *Mitigação:* alerta/transparência sobre uso de CPU.
- **Dependência externa:** a estimativa de tamanho do modelo depende do arquivo em disco e de overheads; estimativas imprecisas podem frustrar o cenário "cabe na GPU". *Mitigação:* margem de segurança e validação por cenário.

## Registros de Decisão de Arquitetura

- [ADR-001: Cascata estrita por prioridade como contrato único de alocação](adrs/adr-001.md) — Consolidar a alocação em uma regra determinística de preenchimento por prioridade, limite fixo de 98%, bloqueio quando CPU desligada e Auto-Balance limpa pins.

## Perguntas em Aberto

- Qual a margem exata de overhead a reservar abaixo dos 98% para evitar OOM em GPUs específicas (a definir na TechSpec/validação)?
- Em GPUs com VRAM já parcialmente ocupada por outros processos, o limite de 98% deve considerar a VRAM total ou a VRAM livre no momento?
