# Automanager Llama.cpp - Guia de Manutenção

Este documento contém todas as informações críticas necessárias para entender, manter e evoluir o projeto Automanager Llama.cpp.

## 📜 Regras de Desenvolvimento
Para diretrizes técnicas detalhadas, padrões de código e fluxos de trabalho obrigatórios, consulte o arquivo [rules.md](rules.md).

## 🚀 Arquitetura Técnica

A aplicação é uma interface de controle baseada em **FastAPI** que gerencia instâncias do **llama-server** (llama.cpp).

- **Backend:** Python 3 + FastAPI + Uvicorn.
- **Frontend:** HTML5 translúcido (Glassmorphism) com Tailwind CSS e Font Awesome.
- **Motor de IA:** `llama-server` (deve estar no PATH do sistema).
- **Monitoramento:** `psutil` para sistema e `nvidia-smi` para métricas de GPU em tempo real.

## 🛠️ Funcionalidades Implementadas

1.  **Monitoramento em Tempo Real:**
    *   Uso de CPU e RAM do sistema.
    *   Uso de GPU (%), Temperatura (°C), Consumo de Energia (W) e VRAM (MB com barra de progresso).
2.  **Gestão de Modelos:**
    *   Escaneamento automático de arquivos `.gguf` em `/media/docker/models`.
    *   Download de novos modelos via URL.
    *   **Modelo Padrão:** Opção de marcar um modelo como padrão para iniciar automaticamente com o serviço.
3.  **Configuração de Hardware Dinâmica:**
    *   Seleção de quais GPUs ativar.
    *   Distribuição de carga (weights) para cálculo automático de `tensor-split`.
    *   Ajuste de Janela de Contexto (Tokens).
4.  **Terminal de Logs:** Visualização em tempo real da saída do `llama-server`.

## 📂 Estrutura de Arquivos e Caminhos

-   **Aplicação Principal:** `/root/automanager-llama.cpp/llama_manager.py`.
-   **Configuração Persistente:** `/root/automanager_config.json` (armazena o `default_model`).
-   **Logs da Aplicação:** `/root/manager.log`.
-   **Logs do Servidor Llama:** `/root/llama_server.log`.
-   **Diretório de Modelos:** `/media/docker/models/`.
-   **Serviço Systemd:** `/etc/systemd/system/llama-manager.service`.

## ⚙️ Configurações e Comandos

### Gerenciar o Serviço
Sempre que o código em `llama_manager.py` for alterado, o serviço deve ser reiniciado:
```bash
systemctl restart llama-manager.service
systemctl status llama-manager.service
```

### Ver Logs do Sistema
```bash
journalctl -u llama-manager.service -f
```

## 🧠 Lógica de Inicialização do Modelo

Quando um modelo é iniciado, o gerenciador realiza os seguintes passos:
1.  **Encerra instâncias anteriores:** Executa `pkill -9 llama-server`.
2.  **Calcula o Tensor Split:** Pega as porcentagens definidas na UI e converte em frações (ex: 0.9500, 0.0500) para o parâmetro `--tensor-split`.
3.  **Define Main GPU:** A GPU com maior peso é definida como `--main-gpu`.
4.  **Parâmetros de Performance:** Utiliza sempre `--flash-attn on`, `--mlock` e `-ngl 99` (offload total para GPU).
5.  **Variáveis de Ambiente:** Configura `PATH` e `LD_LIBRARY_PATH` para incluir o CUDA Toolkit (`/usr/local/cuda`).

## ⚠️ Pontos de Atenção para Manutenção

-   **Persistência:** O arquivo `automanager_config.json` é criado na primeira vez que um modelo padrão é definido.
-   **Segurança:** A API roda em `0.0.0.0:8000`. O servidor Llama roda em `0.0.0.0:8085`.
-   **Mutual Exclusion:** Na lista de modelos, a lógica de JS garante que apenas um checkbox "Padrão" esteja marcado por vez.
-   **nvidia-smi:** O backend depende da presença do comando `nvidia-smi` para coletar métricas detalhadas. Se o comando falhar, as métricas de GPU ficarão zeradas.
