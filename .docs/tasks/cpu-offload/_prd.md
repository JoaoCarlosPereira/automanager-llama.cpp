# PRD — CPU Offload no Automanager Llama.cpp

## 1. Visão Geral

O Automanager Llama.cpp é uma interface web que permite gerenciar e orquestrar instâncias do `llama-server` em máquinas com hardware variado. Atualmente, o sistema distribui carga exclusivamente entre GPUs via tensor split, mas não oferece fallback para a CPU quando o modelo excede a capacidade de VRAM.

Este projeto adiciona suporte a **CPU offload** ao Automanager, permitindo que o modelo use processamento da CPU como extensão das GPUs. A CPU aparecerá como mais um dispositivo na tabela "Recursos de GPU & Configuração", seguindo o mesmo padrão visual das GPUs, com controle de peso compartilhado (CPU + GPUs = 100%).

Além disso, o painel exibirá informações do processador (nome completo) e da RAM do sistema (usada / total em MB), permitindo ao usuário ter visibilidade completa do hardware disponível.

## 2. Objetivos

| Objetivo | Detalhamento |
|----------|-------------|
| Permitir execução de modelos que excedem o VRAM total | Usar a CPU como recurso complementar para carregar camadas do modelo que não cabem nas GPUs |
| Unificar a experiência de distribuição de carga | CPU e GPUs aparecem na mesma tabela, com o mesmo padrão de controle de peso |
| Fornecer visibilidade completa do hardware | Exibir nome do processador e uso de RAM no painel de recursos |
| Manter consistência visual com o padrão existente | Seguir o mesmo design das linhas de GPU: checkbox, nome, métricas, barra de progresso, input de peso |
| Garantir compatibilidade com Auto Balance | O balanceamento automático deve considerar a CPU junto com as GPUs |

## 3. Histórias de Usuário

### 3.1. Visibilidade do Hardware

Como administrador do sistema, quero visualizar o nome completo do meu processador e o total de RAM disponível, para ter consciência completa do hardware antes de carregar um modelo.

### 3.2. Distribuição de Carga CPU+GPU

Como operador do Automanager, quero distribuir a carga do modelo entre GPUs e CPU usando pesos percentuais, para poder rodar modelos maiores que o VRAM total das minhas GPUs.

### 3.3. Controle de Peso Unificado

Como operador, quero ver CPU e GPUs na mesma tabela com controles idênticos (checkbox, peso %, barra de progresso), para que a experiência seja intuitiva e consistente.

### 3.4. Monitoramento em Tempo Real

Como operador, quero ver o uso de CPU (porcentagem) em tempo real na linha da CPU, para acompanhar o impacto do offload no processador.

### 3.5. Auto Balance com CPU

Como operador, quero que o Auto Balance distribua pesos automaticamente entre GPUs e CPU, para que o sistema encontre a melhor configuração de carga sem ajuste manual.

## 4. Funcionalidades Principais

### 4.1. Detecção e Exibição do Processador

- Exibir o nome completo do processador na linha da CPU na tabela de dispositivos
- Exibir o nome completo do processador na linha da CPU na tabela de dispositivos, idêntico ao padrão das GPUs
- A linha da CPU segue o mesmo layout: checkbox de ativação, nome, métricas de monitoramento, RAM e distribuição

### 4.2. Display de RAM no Padrão VRAM

- Exibir a RAM usada e a RAM total em MB na linha da CPU, seguindo o mesmo padrão de display usado para VRAM nas GPUs
- Incluir barra de progresso que reflete o percentual de RAM usada
- Atualizar o display em tempo real conforme as métricas do sistema

### 4.3. Monitoramento de CPU Usage

- Exibir a porcentagem de uso da CPU na coluna de monitoramento, com barra de progresso colorida
- Atualizar em tempo real via SSE, seguindo o mesmo padrão das GPUs

### 4.4. Controle de Peso Compartilhado

- Permitir que o usuário defina o peso percentual da CPU via input numérico
- CPU e GPUs compartilham o total de 100% — o peso da CPU é subtraído do total disponível para as GPUs
- Manter o mesmo comportamento de validação: soma total dos pesos deve igualar 100%
- Manter o mesmo comportamento de pinagem: o usuário pode travar o peso da CPU para que o Auto Balance não o altere

### 4.5. Checkbox de Ativação da CPU

- Permitir que o usuário ative ou desative a CPU como dispositivo de offload via checkbox
- Quando a CPU estiver desativada, o comportamento é o atual (distribuição exclusiva entre GPUs)
- Quando ativada, a CPU passa a participar da distribuição de carga

### 4.6. Auto Balance com Prioridade GPU

- O Auto Balance deve **priorizar GPUs** como dispositivos de compute principal
- A CPU deve ser usada **apenas como último recurso** — o algoritmo deve minimizar ao máximo a carga atribuída à CPU
- O algoritmo deve primeiro tentar distribuir 100% entre as GPUs marcadas/ativas
- Somente quando as GPUs não suportarem o modelo (VRAM insuficiente) é que a CPU entra com o peso restante
- Manter o mesmo comportamento atual de calibração, mas com a lógica de priorização GPU > CPU

### 4.7. Alerta de Capacidade de Hardware

- Manter o alerta existente de "Modelo além da capacidade do hardware"
- Estender o alerta para incluir informações sobre a RAM disponível como recurso de offload
- Sugestões devem considerar tanto VRAM total quanto RAM disponível

## 5. Experiência do Usuário

### 5.1. Tabela de Dispositivos

A tabela "Recursos de GPU & Configuração" passará a exibir todas as GPUs seguidas de uma linha de CPU ao final (quando detectada). A linha da CPU seguirá exatamente o mesmo padrão visual:

| Coluna | GPU | CPU |
|--------|-----|-----|
| **Uso** | GPU utilization % (nvidia-smi) | CPU usage % (psutil) |
| **Principal** | Radio button para definir GPU principal | Sem radio button (CPU não pode ser "principal") |
| **Dispositivo** | Nome da GPU + checkbox de ativação | Nome do processador + checkbox de ativação |
| **Monitoramento** | Temp (°C) + Power (W) | CPU usage % (já coberto na coluna Uso) |
| **VRAM/RAM Status** | VRAM usada / total em MB com barra | RAM usada / total em MB com barra |
| **Distribuição** | Input de peso % + checkbox de fixar | Input de peso % + checkbox de fixar |

### 5.2. Fluxo de Uso

1. O usuário acessa o painel e vê a tabela com GPUs + CPU
2. O usuário ativa/desativa a CPU conforme necessário
3. O usuário ajusta o peso da CPU (e/ou das GPUs) conforme a necessidade
4. O sistema valida que a soma total é 100%
5. Ao carregar o modelo, o sistema distribui os pesos entre CPU e GPUs
6. O usuário monitora o uso em tempo real (CPU usage, RAM, VRAM)

### 5.3. Padrões Visuais

- Mesma paleta de cores: fundo escuro, destaques em azul/ciano
- Mesma tipografia: fontes Space Grotesk e JetBrains Mono
- Mesmas animações: barras de progresso com transição suave (1000ms)
- Mesma responsividade: layout adapta-se a mobile e desktop
- Mesma estrutura de dados: cada dispositivo (GPU ou CPU) tem id único no DOM

## 6. Fora de Escopo

- Configuração avançada de threads da CPU (o sistema usa o padrão do llama-server)
- Controle granular de quais camadas do modelo vão para a CPU (o llama-server decide automaticamente via `--n-gpu-layers`)
- Suporte a NPUs ou outros aceleradores
- Monitoramento de temperatura da CPU (apenas usage %)
- Limitação de RAM máxima utilizável (o sistema usa toda RAM disponível)
- Suporte a swap disk como extensão de RAM
- Interface web de configuração de parâmetros avançados da CPU

## 7. Plano de Entrega por Fases

### Fase 1 — Detecção e Display
- Detectar nome do processador e total de RAM no backend
- Exibir linha da CPU na tabela "Recursos de GPU & Configuração"
- Mostrar nome do processador, RAM usada/total em MB com barra
- Mostrar CPU usage % com barra de progresso
- Checkbox de ativação/desativação da CPU

### Fase 2 — Controle de Peso
- Input de peso percentual na linha da CPU
- Validação de soma total = 100% (CPU + GPUs)
- Checkbox de fixar peso da CPU
- Auto Balance recalibrado para incluir CPU

### Fase 3 — Integração com llama-server
- Enviar `--n-gpu-layers` correto ao llama-server baseado nos pesos definidos
- Tratar erro de OOM e recalcular pesos automaticamente
- Atualizar alerta de capacidade de hardware para incluir RAM

## 8. Métricas de Sucesso

| Métrica | Meta |
|---------|------|
| **Execução de modelos > VRAM** | Modelos que excedem VRAM total carregam com sucesso via CPU offload, com CPU usada apenas como último recurso |
| **Auto Balance inteligente** | Auto Balance sempre prioriza GPUs — CPU só recebe carga quando VRAM das GPUs é insuficiente |
| **Usabilidade** | Usuário consegue ativar/desativar CPU e ajustar peso em < 10 segundos |
| **Consistência visual** | Linha da CPU é indistinguível em estilo das linhas de GPU |
| **Performance** | Tempo de inferência aceitável com CPU offload (> 5 tok/s para modelos 7B-13B) |
| **Auto Balance** | Auto Balance calcula distribuição válida incluindo CPU em < 30 segundos |

## 9. Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| CPU lenta degrada performance para uso interativo | Alto | Manter alerta de capacidade que informa ao usuário quando o offload resulta em performance abaixo do aceitável |
| RAM insuficiente causa swapping e travamento | Alto | Exibir RAM total claramente; alertar quando RAM disponível é baixa antes de carregar modelo |
| Detecção de nome do processador falha em alguma plataforma | Médio | Fallback para "CPU Host" com detalhes no tooltip |
| Auto Balance calcula pesos impraticáveis (ex: 80% CPU) | Médio | Limitar peso máximo da CPU a 70%; exibir alerta quando CPU recebe peso alto |
| Comportamento do Auto Balance muda sem aviso | Médio | Manter badge "Salvo" ao lado do toggle Auto Balance para indicar que o cálculo foi executado |

## 10. Registros de Decisão de Arquitetura

| ADR | Título | Resumo |
|-----|--------|---------|
| [ADR-001](adrs/adr-001.md) | CPU como Dispositivo Unificado na Tabela de Recursos | CPU aparece na mesma tabela das GPUs, com mesmo padrão visual e controles |

## 11. Perguntas em Aberto

| # | Pergunta | Contexto |
|---|----------|----------|
| 1 | Qual o peso máximo aceitável para CPU? | Auto Balance pode tentar enviar 100% para CPU — faz sentido limitar? |
| 2 | Deve haver um toggle global "Habilitar CPU Offload" no card de métricas superior (onde está "Processador")? | Ou a ativação é apenas via checkbox na tabela? |
| 3 | Quando o usuário desativar a CPU, o Auto Balance existente (que só considera GPUs) deve continuar funcionando como antes? | Ou o Auto Balance sempre considera CPU (mesmo desativada, peso = 0)? |
| 4 | O Auto Balance deve mostrar visualmente quanto da carga foi atribuída à CPU (ex: badge "CPU: 5%") no painel? | Ou apenas exibir o peso na tabela como nas GPUs? |
