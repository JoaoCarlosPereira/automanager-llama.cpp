# PRD — Automanager Performance Center

**Versão:** 1.0
**Data:** 2026-06-10
**Status:** Rascunho para revisão
**Autor:** Análise Automanager Performance

---

## Visão Geral

O Automanager Llama.cpp é um plano de controle FastAPI que orquestra instâncias do `llama-server` com gerenciamento multi-GPU, recuperação automática de OOM e monitoramento de hardware. Apesar de funcional, a análise profunda revela que o gerenciador **não está operando com as configurações de performance máximas disponíveis no llama-server**.

Múltiplas flags de performance críticas estão ausentes na construção do comando, variáveis de ambiente de otimização CUDA não são definidas, e parâmetros de batching, KV cache, NUMA e sistema operacional não são configurados programaticamente. Este PRD define a especificação para transformar o Automanager em uma **central de desempenho absoluto** dos modelos que gerencia, maximizando throughput (tokens/segundo) em qualquer configuração de hardware multi-GPU PCIe e gama de modelos (1B a 30B+).

O objetivo é identificar TODAS as configurações e ajustes ausentes, pesquisar exaustivamente as melhores práticas de performance do llama.cpp, e especificar implementações concretas no código que permitam ao gerenciador operar no limite máximo de performance de cada modelo e hardware.

---

## Objetivos

1. **Completude técnica:** Mapear 100% dos parâmetros de performance do llama-server e identificar quais o Automanager já utiliza, quais ignora, e quais são críticos mas não implementados.
2. **Maximizar throughput:** Priorizar todas as otimizações pelo impacto em tokens/segundo, com benchmarks de referência para medir o ganho de cada mudança.
3. **Cobertura multi-hardware:** Garantir que as otimizações funcionem para qualquer configuração multi-GPU PCIe (1x a 4+ GPUs), desde modelos pequenos até 30B+.
4. **Adaptabilidade:** O sistema deve detectar automaticamente o hardware disponível e ajustar as configurações de performance sem intervenção do usuário.
5. **Performance auditável:** Implementar métricas e logging que permitam verificar se cada otimização está ativa e medir seu impacto.

---

## Histórias de Usuário

### HU-01 — Detecção Automática de Performance Ótima
Como administrador do Automanager, eu quero que o gerenciador detecte automaticamente o hardware (GPU, VRAM, interconexão, NUMA) e aplique as configurações de performance ideais para aquele setup específico, sem que eu precise pesquisar ou configurar flags manualmente.

### HU-02 — Maximização Transparente de Throughput
Como administrador, eu quero que o gerenciador sempre inicie o llama-server com as melhores configurações de batching, KV cache, CUDA e memory para maximizar tokens/segundo, independentemente do modelo carregado (1B ou 30B+).

### HU-03 — Adaptação Dinâmica por Modelo
Como administrador, eu quero que o gerenciador ajuste automaticamente os parâmetros de performance (batch size, kv cache, tensor split) baseado no tamanho e tipo do modelo carregado, para que cada modelo rode com configurações otimizadas para seu perfil.

### HU-04 — Visibilidade de Otimizações Ativas
Como administrador, eu quero ver no dashboard quais otimizações de performance estão ativas para a instância atual (flash attention, KV quantization, continuous batching, CUDA graph, NUMA, etc.), quais estão desativadas e por quê.

### HU-05 — Benchmark Integrado
Como administrador, eu quero executar benchmarks de performance integrados ao Automanager para comparar o throughput antes e depois das otimizações, e validar que cada mudança está gerando o ganho esperado.

### HU-06 — Recuperação Inteligente com Preservação de Performance
Como administrador, eu quero que quando o gerenciador detecta OOM e redistribui carga, ele também reavalie se as configurações de performance (batch size, kv cache quantization) podem ser ajustadas para evitar OOM sem perder throughput.

---

## Funcionalidades Principais

### FP-01 — Motor de Configuração de Performance (Performance Config Engine)

Criar um novo módulo `performance_config.py` que centraliza a lógica de construção do comando llama-server com TODAS as flags de performance disponíveis. Este motor será responsável por:

- Ler os parâmetros de hardware detectados (GPU count, VRAM, interconexão, NUMA nodes)
- Consultar os metadados do modelo (parâmetros, camadas, tipo de quantização, MTP heads)
- Aplicar uma matriz de otimizações baseada no perfil detectado
- Gerar o comando final com todas as flags de performance otimizadas

Este motor substituirá a construção de comando atual espalhada no `process_manager.py` (linhas 495-524).

### FP-02 — Motor de KV Cache Inteligente

Implementar configuração automática de KV cache quantization (`--cache-type-k` e `--cache-type-v`) baseado em:
- VRAM disponível vs VRAM necessária para o contexto
- Context size solicitado
- Tipo de modelo (7B, 13B, 30B, etc.)

Regras de decisão:
- Se VRAM livre >= 90% da necessidade: `k:f16 / v:f16` (qualidade máxima)
- Se VRAM livre entre 50-90%: `k:q8_0 / v:q8_0` (equilíbrio ideal)
- Se VRAM livre < 50% ou contexto > 32K: `k:q8_0 / v:q8_0` com `--cache-prompt` e `--cache-reuse`
- Emergência (contexto > 64K): `k:q8_0 / v:q8_0` com `--kv-cache-demand-paged`

### FP-03 — Motor de Batching Adaptativo

Substituir o `--batch-size` fixo por um sistema que calcula dinamicamente:
- `--batch-size` (-b) baseado no VRAM livre e parallel slots
- `--ubatch-size` (-ub) baseado no VRAM livre (crítico para prefill speed — 4x impacto)
- `--cont-batching` sempre ativado para multi-usuário
- `--parallel-callbacks` quando parallel slots > 2

Regra base: `--ubatch-size` deve ser maximizado até 92% de pressão de VRAM, pois é o maior fator de velocidade de prefill.

### FP-04 — Variáveis de Ambiente CUDA Avançadas

Adicionar ao ambiente do processo do llama-server:
- `GGML_CUDA_GRAPH_OPT=1` — kernels fusion com streams concorrentes (1.3x-1.5x em RTX 40/50 series)
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — para hardware com memória unificada
- `LLAMA_SCHED_MAX_COPIES` — controle de operações em pipeline multi-GPU
- Otimizações de kernel CUDA (sync reduction, mul_mat_vec_q) automáticas

### FP-05 — Motor NUMA e Sistema Operacional

Implementar detecção e configuração de NUMA para sistemas multi-socket:
- Detectar NUMA nodes via `numastat` ou `/sys/devices/system/node/`
- Aplicar `--numa` flag ao llama-server
- Gerar script de otimização de kernel (vm.swappiness, transparent huge pages, CPU governor)
- `numactl` wrapper quando necessário para bind de CPU-GPU

### FP-06 — Motor de Model Discovery com Metadata de Performance

Estender o `detect_model_layers()` atual para também extrair:
- Número total de parâmetros do modelo (via metadata GGUF)
- Tipo de arquitetura (Mistral, Llama, Qwen, Mixtral MoE, etc.)
- Se o modelo tem MTP (multi-token prediction) heads
- Tamanho estimado em VRAM baseado na quantização detectada
- Recomendação de performance baseada no perfil detectado

### FP-07 — Dashboard de Performance

Adicionar ao dashboard existente:
- Painel "Performance Config" mostrando todas as flags ativas/inativas
- Indicadores de quais otimizações estão aplicadas e quais foram pularadas (com motivo)
- Gráfico de throughput histórico (tokens/segundo ao longo do tempo)
- Botão para benchmark integrado

### FP-08 — Benchmark Integrado

Implementar endpoint `/benchmark` que:
- Carrega o modelo atual com as configurações de performance
- Executa medições de prompt processing (pp512, pp1024, pp2048) e token generation (tg128, tg256)
- Retorna throughput em tokens/segundo para cada métrica
- Compara com baseline do sistema e mostra ganho percentual
- Armazena resultado no log para tracking temporal

### FP-09 — Configuração de Logging de Performance

Adicionar flags de logging ao llama-server para:
- `--log-disable` ou `--log-format` otimizado para reduzir overhead de I/O
- `--threads-http` configurado automaticamente baseado no hardware
- Métricas Prometheus (`--metrics`) sempre habilitadas para monitoramento
- Log de performance por request (tempo de prefill, tempo de generation, throughput)

### FP-10 — Otimização de Interconexão Multi-GPU

Detectar tipo de interconexão entre GPUs (PCIe vs NVLink vs NVSwitch) e ajustar:
- PCIe: usar `--split-mode layer` (pipeline parallelism — melhor para PCIe)
- NVLink/NVSwitch: usar `--split-mode tensor` (tensor parallelism — melhor para alta banda)
- Calcular tensor-split baseado em VRAM real (não proporção abstrata) — ex: 24GB+16GB = `22,14`
- `--main-gpu` sempre configurado para a GPU com maior VRAM ou interconexão direta

---

## Experiência do Usuário

O administrador não precisa conhecer nenhuma flag de performance. O sistema detecta, configura e otimiza automaticamente. No dashboard, o administrador vê:
- Status de performance: "✅ 12 otimizações ativas | 3 não aplicáveis | 1 com fallback"
- Throughput atual em tempo real (tokens/segundo)
- Botão de benchmark para validar configurações
- Alertas se alguma otimização crítica não pôde ser aplicada (com motivo e sugestão)

---

## Fora de Escopo

- Benchmark de modelos concorrentes (vLLM, Ollama, TGI) — comparativo pode ser mencionado no relatório, mas não é implementado
- Modificação do código fonte do llama-server — todas as otimizações são via flags de linha de comando e ambiente
- Otimizações de hardware (BIOS, drivers, firmware) — o sistema pode recomendar, mas não implementa
- Suporte a Apple Silicon — foco em NVIDIA CUDA no escopo inicial (pode ser estendido depois)
- Web UI para ajuste manual de flags de performance — a interface é apenas informativa, não de configuração

---

## Plano de Entrega por Fases

### Fase 1 — Fundação (Alto Impacto, Baixa Complexidade)
- FP-01: Motor de Configuração de Performance (estrutura base)
- FP-04: Variáveis de Ambiente CUDA Avançadas (`GGML_CUDA_GRAPH_OPT=1`)
- FP-09: Configuração de Logging de Performance (`--metrics`, `--threads-http`)
- FP-03 (parcial): `--cont-batching` sempre ativado, `--ubatch-size` adaptativo
- **Impacto estimado: +15-40% throughput**

### Fase 2 — Otimizações de Memória (Alto Impacto, Média Complexidade)
- FP-02: Motor de KV Cache Inteligente (q8_0 automático)
- FP-03 (completo): Batching adaptativo com cálculo de VRAM
- FP-06 (parcial): Extensão do model discovery para metadata de performance
- **Impacto estimado: +20-60% throughput (depende do modelo e VRAM)**

### Fase 3 — Otimizações de Hardware (Médio Impacto, Alta Complexidade)
- FP-05: Motor NUMA e Sistema Operacional
- FP-10: Otimização de Interconexão Multi-GPU (detecção PCIe/NVLink)
- FP-06 (completo): Metadata completa do modelo com recomendações
- **Impacto estimado: +10-30% throughput em setups multi-GPU**

### Fase 4 — Visibilidade e Benchmark (Médio Impacto, Média Complexidade)
- FP-07: Dashboard de Performance
- FP-08: Benchmark Integrado
- **Impacto estimado: Visibilidade e validação, não impacto direto em throughput**

### Fase 5 — Otimizações Avançadas (Baixo-Médio Impacto, Alta Complexidade)
- FP-11 (nova): Prompt Caching (`--cache-prompt`, `--cache-reuse`)
- FP-12 (nova): Otimizações de RoPE/Contexto (`--rope-freq-base`, `--rope-scaling yarn`)
- FP-13 (nova): CPU thread topology (`--cpu-no-hyperthreading`, core pinning)
- **Impacto estimado: +5-15% throughput em workloads específicos**

---

## Métricas de Sucesso

1. **Throughput médio:** +25% ou mais em throughput tokens/segundo comparado com configurações atuais, medido via benchmark integrado.
2. **Cobertura de flags:** 95%+ das flags de performance do llama-server mapeadas e aplicadas quando relevantes.
3. **Adaptação automática:** 100% das otimizações aplicadas sem intervenção do usuário para o hardware detectado.
4. **Estabilidade:** Zero regressões — nenhuma otimização pode causar OOM ou instabilidade.
5. **Visibilidade:** Dashboard mostra status de todas as otimizações com clareza.

---

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| `GGML_CUDA_GRAPH_OPT=1` incompatível com GPUs antigas | Falha no spawn do servidor | Detectar architecture CUDA (sm_80+); habilitar apenas para Ampere+ |
| KV cache q8_0 reduz qualidade do modelo | Respostas degradadas para certos modelos | Benchmark de qualidade antes de habilitar; oferecer modo "full quality" |
| `--ubatch-size` alto causa OOM | Falha no spawn do servidor | Limitar ubatch a 92% de pressão de VRAM; fallback automático para ubatch menor |
| NUMA tuning só funciona em Linux multi-socket | Inútil em setups single-socket ou Windows | Detecção de plataforma; NUMA só ativado quando beneficial |
| Dashboard de performance adiciona carga ao FastAPI | Latência nas requisições da UI | Renderização assíncrona; dados atualizados via SSE, não blocking |
| Múltiplas otimizações simultâneas causam comportamento inesperado | Difícil de debugar | Implementar com feature flags; cada otimização pode ser desativada individualmente via config |
| Modelos MoE (Mixtral, Qwen3 MoE) têm comportamento diferente de performance | Otimizações genéricas subotimizam MoE | Detecção de arquitetura MoE; regras de performance específicas para MoE |

---

## Registros de Decisão de Arquitetura

| ADR | Título | Resumo |
|-----|--------|---------|
| [ADR-001](adrs/adr-001.md) | Abordagem de Análise: Híbrida (Camada + Fluxo) | Combinar mapeamento por fluxo do gerenciador com matriz de otimizações por camada técnica, priorizado por impacto no throughput |

---

## Perguntas em Aberto

1. **Qual é a configuração de hardware real do usuário?** A análise cobre multi-GPU PCIe genérico, mas as otimizações específicas (NUMA, tensor vs layer split, GPU architecture para CUDA graph) dependem do hardware exato.
2. **Existe um modelo "base line" ou conjunto de modelos para validar as otimizações?** Idealmente teríamos 2-3 modelos para benchmark (ex: 7B Q4_K_M, 13B Q4_K_M, 30B Q4_K_M) antes e depois das mudanças.
3. **O Automanager roda em Linux, Windows, ou ambos?** O código atual tem `preexec_fn=os.setsid` (POSIX-only), mas o path `/usr/local/cuda/bin` sugere Linux. NUMA tuning e kernel params são Linux-specific.
4. **Qual versão do llama-server está em uso?** Algumas flags (cont-batching, cache-reuse, turboquant) foram adicionadas em commits específicos. Precisamos saber a versão para saber quais flags estão disponíveis.
5. **Deve haver um "modo seguro" onde nenhuma otimização é aplicada automaticamente?** Para ambientes de produção crítica onde estabilidade é prioritária sobre performance.
6. **As otimizações devem ser persistentes por modelo ou globais?** Modelos diferentes (ex: MoE vs dense) podem se beneficiar de configurações diferentes.
