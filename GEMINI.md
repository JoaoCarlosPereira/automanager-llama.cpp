# Automanager Llama.cpp - Contexto do Projeto

Este repositório contém um gerenciador web leve baseado em FastAPI para controlar instâncias do servidor `llama-server` (parte do projeto `llama.cpp`). Ele foi projetado para facilitar a execução de modelos GGUF com otimizações específicas para GPUs NVIDIA (especialmente RTX 3090).

## 🚀 Visão Geral e Arquitetura

O projeto atua como uma camada de interface (Web UI) e API de controle sobre o binário `llama-server`.

- **Tecnologias Principais:** Python 3, FastAPI, Uvicorn, psutil.
- **Dependências Externas:** `llama.cpp` (compilado com suporte CUDA), drivers NVIDIA e `nvidia-smi`.
- **Fluxo de Trabalho:**
    1. O gerenciador escaneia recursivamente o diretório `/media/docker/models` em busca de arquivos `.gguf`.
    2. Através da interface web moderna (porta 8000), o usuário pode selecionar quais GPUs NVIDIA deseja utilizar e definir a porcentagem de carga (VRAM) para cada uma.
    3. O gerenciador calcula o `tensor-split` e o `main-gpu` automaticamente.
    4. O gerenciador inicia o `llama-server` na porta 8085 com configurações de alta performance.

## 🛠️ Comandos Principais

### Executando o Gerenciador
```bash
python gemma_manager.py
```
O painel de controle estará disponível em `http://localhost:8000` com um visual renovado.

## 📝 Funcionalidades da Interface

- **Seleção Dinâmica de GPU:** Checkboxes para ativar/desativar GPUs específicas.
- **Distribuição de Carga:** Campos de porcentagem para definir quanto do modelo cada GPU processará.
- **Auto-Balanceamento:** Lógica inteligente para sugerir divisões equilibradas (ex: 95/5 para setups com RTX 3090).
- **Monitoramento em Tempo Real:** Status visual (Online/Offline) e indicação do modelo ativo.

### Servidor de Modelos (llama-server)
O servidor é iniciado automaticamente pelo gerenciador, mas também pode ser disparado manualmente via script para testes:
```bash
./start_gemma.sh
```
A API compatível com OpenAI será servida em `http://<IP_DA_MAQUINA>:8085/v1`.

## 📂 Estrutura de Arquivos Chave

- `gemma_manager.py`: Aplicação FastAPI principal. Contém a lógica de descoberta de modelos, monitoramento de processos e a interface HTML embutida.
- `start_gemma.sh`: Script auxiliar para iniciar manualmente o modelo Gemma com parâmetros otimizados.
- `README.md`: Documentação básica do projeto.

## 📝 Convenções e Práticas

- **Logs:**
    - Logs do gerenciador: `/root/manager.log`
    - Logs do servidor de modelos: `/root/gemma_server.log`
- **Gestão de Processos:** O projeto utiliza `pkill -9 llama-server` para garantir que instâncias anteriores sejam encerradas antes de iniciar um novo modelo.
- **Otimização de GPU:**
    - Prioridade para RTX 3090 como dispositivo principal.
    - Suporte a `tensor-split` configurável via UI para balancear carga entre múltiplas GPUs (ex: RTX 3090 + Tesla P100).
    - Uso obrigatório de `--flash-attn on` e `--mlock` para performance e estabilidade.

## ⚠️ Requisitos de Ambiente

- `llama.cpp` deve estar instalado e o binário `llama-server` acessível no PATH.
- CUDA Toolkit instalado em `/usr/local/cuda`.
- Bibliotecas Python: `fastapi`, `uvicorn`, `psutil`, `pydantic`.
