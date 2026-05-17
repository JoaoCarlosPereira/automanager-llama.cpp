# 📄 Product Requirements Document (PRD)
**Produto:** Automanager Llama.cpp
**Versão:** 1.0 (Current State)
**Data:** Maio de 2026

## 1. Visão Geral do Produto (Product Overview)
O **Automanager Llama.cpp** é uma solução que atua como orquestrador e interface de controle (Control Plane) para instâncias do `llama-server` (motor do llama.cpp). Ele foi projetado para abstrair a complexidade de rodar modelos de linguagem grandes (LLMs) localmente, fornecendo uma interface web moderna para gestão do ciclo de vida dos modelos, monitoramento de hardware em tempo real e distribuição dinâmica de carga (VRAM) em ambientes com múltiplas GPUs.

## 2. Problema a ser Resolvido (The Problem)
Executar LLMs localmente com `llama.cpp`, especialmente em máquinas com várias placas de vídeo (multi-GPU), exige o cálculo manual de divisões de tensores (`--tensor-split`), configurações complexas de linha de comando, e não oferece feedback visual nativo do consumo de recursos. Além disso, falhas por falta de memória (OOM - Out of Memory) geralmente derrubam o serviço e exigem intervenção manual para reconfiguração.

## 3. Público-Alvo (Target Audience)
*   **Engenheiros e Pesquisadores de IA:** Que precisam testar múltiplos modelos GGUF rapidamente.
*   **Sysadmins / DevOps:** Que gerenciam servidores de inferência locais e precisam de painéis de monitoramento e controle.
*   **Entusiastas de Self-Hosted AI:** Que possuem hardware dedicado (ex: rigs com várias RTX 3090/4090).

## 4. Funcionalidades Core (Requisitos Funcionais)

### 4.1. Monitoramento de Hardware em Tempo Real
*   **Telemetria de Host:** Uso de CPU (%) e Memória RAM (%) do sistema hospedeiro.
*   **Telemetria de GPU:** Integração com `nvidia-smi` para exibir Uso (%), VRAM (Usada/Total), Temperatura (°C) e Consumo de Energia (W) por dispositivo.

### 4.2. Gestão de Repositório de Modelos
*   **Auto-Discovery:** Escaneamento recursivo do diretório de modelos (`/media/docker/models`) para identificar arquivos `.gguf`.
*   **Classificação Inteligente:** Separação automática entre Modelos de Linguagem base e Modelos de Visão (Projectors/mmproj).
*   **Operações CRUD:** Capacidade de Renomear e Excluir modelos diretamente pela UI.
*   **Ingestão via URL:** Ferramenta de download embutida para baixar novos modelos `.gguf` diretamente para o servidor, com barra de progresso assíncrona.

### 4.3. Orquestração e Configuração de Inferência
*   **Balanceamento Visual de Carga (Tensor Split):** Interface com sliders/inputs numéricos para o usuário definir o percentual de carga (0-100%) para cada GPU. O sistema converte isso automaticamente na flag `--tensor-split`.
*   **Configuração Dinâmica:** Seleção via dropdown de Janela de Contexto (2K até 1M) e acoplamento de modelos de visão (mmproj).
*   **Persistência de Estado:** Salvar as configurações específicas (pesos de GPU, contexto) utilizadas na última execução de cada modelo.
*   **Modelo Padrão (Auto-start):** Possibilidade de "favoritar" um modelo para que o servidor o inicie automaticamente na inicialização do sistema ou em caso de crash.

### 4.4. Resiliência e Auto-Cura (Auto-Recovery)
*   **Monitoramento Anti-OOM:** Thread em background (Watchdog) que lê ativamente o `llama_server.log`. Se detectar erros de alocação de memória (Out of Memory), o sistema entra em modo de "Recovery":
    *   Mata o processo atual.
    *   Re-calcula dinamicamente os pesos das GPUs (retirando carga da GPU sobrecarregada).
    *   Reinicia o servidor automaticamente.

### 4.5. Segurança e API
*   **Autenticação Integrada:** Geração automática de Tokens Bearer no padrão OpenAI (`sk-...`).
*   **Gestão de Credenciais:** UI para visualização, cópia para área de transferência e renovação do Token de API.
*   **Injeção de Segurança:** Repasse obrigatório do Token gerado para a flag `--api-key` do `llama-server`.

## 5. Requisitos Não Funcionais (NFRs)

*   **Design de Interface (UX/UI):** Padrão estético *Glassmorphism* (translúcido), paleta Dark Mode com feedback visual imediato via animações TailwindCSS.
*   **Arquitetura Single-Page-ish:** Uso intensivo de `fetch` assíncrono para garantir que a página não recarregue ao operar o servidor, mantendo a sensação de aplicação "viva".
*   **Log Streaming:** O frontend deve ser capaz de fazer stream de logs do terminal bash em tempo real sem estourar o limite de memória do navegador (cap de 500 linhas).
*   **Zero-Database:** A aplicação não deve exigir um SGBD externo. Todo o estado persistente deve viver em um único arquivo de configuração (`automanager_config.json`).

## 6. Arquitetura do Sistema e Stack

*   **Backend:** Python 3.11+
*   **Framework API / Web:** FastAPI + Uvicorn
*   **Gerenciamento de Processos SO:** Módulo `subprocess`, `psutil`, e `signal`.
*   **Frontend:** HTML5 Híbrido retornado via FastAPI, estilizado via CDN (TailwindCSS, FontAwesome). Manipulação de DOM via Vanilla JavaScript.
*   **Motor de IA:** Binário externo `llama-server` no `$PATH` compilado com suporte a CUDA e `--flash-attn`.

## 7. Métricas de Sucesso (KPIs)
*   Redução drástica no tempo de setup/troca de modelos (de minutos no terminal para cliques de segundos).
*   Taxa de sucesso do *Auto-Recovery OOM* (capacidade do sistema se recuperar sozinho de sobrecargas de VRAM sem travar a interface).
*   Utilização balanceada das GPUs através da ferramenta de visualização.