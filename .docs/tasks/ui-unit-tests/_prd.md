# PRD: Testes de Interface para AutoManager Llama.cpp

## Visão Geral

Este projeto cria uma suíte completa de testes para a dashboard web do AutoManager Llama.cpp, garantindo que nenhuma alteração futura quebre a interface, a lógica de interação ou os fluxos do usuário. O AutoManager é uma aplicação FastAPI que gerencia instâncias do `llama-server` para inferência de modelos LLM locais, com uma dashboard SPA em JavaScript para monitoramento e controle.

**Problema:** Atualmente não existem testes para a camada de interface — alterações no HTML ou JavaScript podem quebrar funcionalidades sem detecção automática.

**Para quem:** Desenvolvedores que mantêm e evoluem o projeto, garantindo confiança para fazer alterações.

**Por que é valioso:** Elimina o risco de regressões na interface, permite refatorações com segurança e cria uma rede de proteção que detecta automaticamente quando algo quebra.

## Objetivos

- Cobertura de todas as funções JavaScript da dashboard (autenticação, gerenciamento de modelos, métricas, GPU/auto-balance)
- Detecção automática de regressões em fluxos críticos do usuário (login, start/stop, monitoring)
- Testes rápidos (< 5 minutos para suite completa) para encorajar execução frequente
- Testes estáveis (não flaky) que funcionam localmente e podem ser estendidos para CI/CD

## Histórias de Usuário

### Desenvolvedor (persona principal)

- Como desenvolvedor, quero rodar testes antes de fazer deploy para ter certeza de que a dashboard funciona
- Como desenvolvedor, quero testes que detectem quando uma alteração quebra uma função JavaScript existente
- Como desenvolvedor, quero testes de integração que validem fluxos completos do usuário (login → monitoramento → gerenciamento)
- Como desenvolvedor, quero feedback imediato (em segundos) para testes unitários e feedback rápido (em minutos) para testes de interação

### Usuário Final (persona indireta)

- Como usuário final, quero que a dashboard carregue corretamente e mostre todos os componentes esperados
- Como usuário final, quero que as funções de login, start/stop de modelos e monitoramento funcionem de forma confiável

## Funcionalidades Principais

### 1. Testes Unitários de Funções JavaScript

Validar que cada função JavaScript da dashboard se comporta corretamente quando isolada:

- **Autenticação:** login, logout, change password, validação de sessão, expiração de token
- **Gerenciamento de Modelos:** start, stop, rename, delete, set default, download
- **Métricas em Tempo Real:** polling de dados, atualização de UI, parsing de métricas de CPU/RAM/GPU
- **GPU e Auto-Balance:** cálculo de pesos, validação de inputs, toggle de GPUs, cálculo de tensor split
- **SSE Log Streaming:** conexão com stream de logs, parsing de linhas, atualização do terminal
- **Gerenciamento de Sessão:** persistência de token, renovação automática, tratamento de expiração

### 2. Testes de Contrato de HTML

Validar que a resposta HTML da dashboard contém todos os elementos esperados:

- Overlay de login com campos de username e password
- Header com nome do app, status badge e botão de logout
- Painel de métricas com barras de CPU e RAM
- Painel de configuração de GPU com tabela de GPUs e controles de peso
- Card do modelo ativo com nome, uptime e botão stop
- Terminal de logs com botão clear
- Painel de repositório de modelos com lista de modelos e controles
- Canvas do Pac-Man como background

### 3. Testes de Fluxo do Usuário (E2E)

Validar interações completas do usuário no navegador:

- **Fluxo de login:** entrar com credenciais válidas → redirecionamento para dashboard → exibição dos componentes
- **Fluxo de start/stop:** iniciar llama-server → verificar status ONLINE → parar → verificar status OFFLINE
- **Fluxo de monitoramento:** verificar que métricas aparecem e atualizam na tela
- **Fluxo de gerenciamento de modelos:** selecionar modelo → clicar start → verificar card ativo → clicar stop
- **Fluxo de rename/delete:** renomear modelo → verificar nome atualizado → deletar modelo → verificar sumário da lista

## Experiência do Usuário

### Jornada do Desenvolvedor

1. **Primeiro uso:** O desenvolvedor clona o projeto, instala dependências de teste e roda a suite pela primeira vez
2. **Dia a dia:** Antes de fazer commits com alterações na UI ou backend, roda os testes — feedback em segundos (unitários) ou minutos (E2E)
3. **Refatoração:** Ao refatorar uma função JS, os testes garantem que o comportamento não mudou
4. **Novas funcionalidades:** Ao adicionar uma feature nova, escreve o teste primeiro, depois a implementação

### Jornada do Usuário (validada pelos testes)

1. Acessa a dashboard → vê overlay de login
2. Insere credenciais → dashboard carrega com todos os painéis
3. Visualiza métricas em tempo real
4. Gerencia modelos (start, stop, rename, delete, download)
5. Configura GPUs e pesos de tensor
6. Visualiza logs do llama-server em tempo real

## Fora de Escopo (Non-Goals)

- Testes de performance ou load testing da dashboard
- Testes de acessibilidade (WCAG, screen readers)
- Testes de compatibilidade cross-browser (o foco é Chromium)
- Testes de segurança da interface (XSS, CSRF)
- Testes visuais (screenshot comparison, pixel-perfect)
- Cobertura de testes para a pasta `design/` (portfolio template, não parte do app)

## Plano de Entrega por Fases

### MVP (Fase 1)

- Testes unitários das funções de autenticação (login, logout, sessão)
- Testes de contrato HTML (validar presença de elementos na página)
- Fluxo E2E de login com sucesso e falha

**Critério de sucesso:** Suite roda em < 3 minutos, cobre login 100%, dashboard carrega validado.

### Fase 2

- Testes unitários de gerenciamento de modelos (start, stop, rename, delete, set default, download)
- Testes unitários de métricas e SSE log streaming
- Fluxos E2E de start/stop e monitoramento de métricas

**Critério de sucesso:** Todas as funções JS principais testadas, fluxo de gerenciamento de modelos validado no navegador.

### Fase 3

- Testes unitários de GPU e auto-balance
- Fluxos E2E completos de gerenciamento de modelos (rename, delete)
- Testes de cenários de erro (credenciais inválidas, rede instável, modelos ausentes)

**Critério de sucesso:** Cobertura 100% de todas as funções JS, todos os fluxos de usuário validados, suite completa < 5 minutos.

## Métricas de Sucesso

- **Cobertura de funções:** 100% das funções JavaScript da dashboard possuem pelo menos um teste
- **Tempo de execução:** Suite completa roda em menos de 5 minutos localmente
- **Estabilidade:** Taxa de flakiness abaixo de 2% (menos de 1 falha falsa a cada 50 execuções)
- **Confiança:** Desenvolvedores conseguem refatorar código da UI sem medo de regressões
- **Detecção:** 100% dos bugs introduzidos na UI são capturados pelos testes antes do deploy

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Testes E2E flaky por timing de carregamento | Usar condicion-based waits (waitForSelector, waitForResponse) ao invés de sleeps fixos |
| Extração do JS quebrar funcionalidades existentes | Refatoração incremental — extrair módulo por módulo, rodar testes após cada etapa |
| Manutenção dos testes se tornar custosa | Escrever testes focados em comportamento, não em implementação — menos propenso a mudar quando a UI é refatorada |
| Dependência de servidor llama-server rodando para E2E | Mockar o backend para E2E — controlar respostas via HTTP mock para isolar testes de UI |

## Registros de Decisão de Arquitetura

- [ADR-001: Extração de JavaScript para testes unitários com Jest + Playwright E2E](adrs/adr-001.md) — Decisão por extrair todo o JS embutido e adotar Jest para unit tests + Playwright para E2E

## Perguntas em Aberto

- Definição de prioridade de navegadores para E2E — iniciar com Chromium apenas, ou incluir Firefox e WebKit desde o MVP?
- Nível de mock do backend nos testes E2E — mockar todas as respostas da API para isolar testes de UI, ou depender do backend real com dados mockados?
- Definição de threshold mínimo de cobertura de código — existe um percentual-alvo de cobertura JS que o projeto deve buscar?
