# 🚀 Automanager Llama.cpp

Um gerenciador web avançado e leve, baseado em **FastAPI**, projetado para orquestrar instâncias do `llama-server` com foco em máxima performance em hardware **NVIDIA Multi-GPU** (especialmente otimizado para setups como RTX 3090 + Tesla P100).

---

## 🌟 Funcionalidades Principais

### 1. 🔍 Descoberta Automática de Modelos
- Varredura recursiva inteligente no diretório `/media/docker/models`.
- Identifica instantaneamente qualquer arquivo `.gguf` adicionado, organizando-os por subdiretórios na interface.

### 2. 🛠️ Controle Total de Hardware (NVIDIA)
- **Seleção Dinâmica:** Ative ou desative GPUs específicas via checkbox antes de subir o modelo.
- **Distribuição de Carga (Tensor Split):** Ajuste fino da porcentagem de VRAM que cada GPU deve carregar.
- **Mapeamento Anti-Inversão:** Utiliza a ordem de detecção interna do `llama.cpp` (Device IDs) para garantir que a carga configurada vá exatamente para a placa correta, eliminando erros de *Out of Memory*.
- **Priorização Automática:** Sugere automaticamente configurações otimizadas (ex: 95% na RTX 3090).

### 3. 📊 Monitoramento em Tempo Real
- **Dashboard de Recursos:** Acompanhamento vivo do uso de **CPU**, **Memória RAM** e **GPUs** (Utilização % e VRAM ocupada).
- **Console de Execução:** Streaming de logs em tempo real diretamente na interface web, com destaque para erros e avisos do servidor.

### 4. ⚡ Otimização de Performance
- **Contexto Configurável:** Escolha janelas de contexto de **2k até 256k** via menu dropdown.
- **Flash Attention & MLock:** Ativados por padrão para garantir velocidade máxima e evitar que o modelo seja movido para o swap do sistema.
- **GPU Offloading Total:** Configurado para carregar 100% das camadas na VRAM (`-ngl 99`).

### 5. 🔌 Conectividade e Chat
- **API OpenAI-Compatible:** Servidor fixo na porta `8085/v1`, pronto para integração com ferramentas como LangChain, AutoGPT ou apps externos.
- **Acesso Direto ao Chat:** Botão dedicado para abrir a interface nativa de chat do `llama.cpp`.

---

## 📋 Requisitos do Sistema

### Hardware
- Servidor Linux (Ubuntu 20.04+ recomendado).
- GPUs NVIDIA com drivers atualizados (v575+ recomendado).
- Pelo menos 16GB de RAM.

### Software
- **llama.cpp:** Compilado localmente com suporte a CUDA.
- **Python 3.11+**
- **Dependências Python:** `fastapi`, `uvicorn`, `psutil`, `pydantic`.
- **NVIDIA Container Toolkit:** (Opcional, se usar via docker).

---

## 🚀 Instalação e Configuração

### 1. Estrutura de Pastas
O gerenciador espera que seus modelos estejam em:
`/media/docker/models/`

### 2. Caminhos de Binários
O sistema utiliza o `llama-server` instalado em:
`/media/joao/Dados/llama.cpp/build/bin/` (ou disponível no PATH global).

### 3. Configuração do Serviço (Systemd)
O gerenciador é configurado como um serviço de fundo para iniciar com o sistema:
```bash
sudo systemctl enable llama-manager.service
sudo systemctl start llama-manager.service
```

### 4. Atalho de Reinicialização
Sempre que editar o código ou layout, use o comando rápido criado no seu `.bashrc`:
```bash
reiniciar-llama
```

---

## 🖥️ Como Usar

1.  Acesse a interface em seu navegador: `http://localhost:8000`.
2.  Na tabela de **Hardware**, selecione as GPUs desejadas e defina a porcentagem de carga.
3.  Selecione o **Tamanho do Contexto** desejado (ex: 64k).
4.  Clique em **"Iniciar"** ao lado do modelo GGUF escolhido.
5.  Acompanhe o carregamento no **Terminal Output** abaixo.
6.  Assim que o status mudar para **ONLINE**, clique em **"ACESSAR CHAT"** para começar a usar ou conecte sua aplicação na porta `8085`.

---

## 📂 Estrutura do Projeto
- `gemma_manager.py`: O coração do sistema (Backend FastAPI + Frontend HTML/JS).
- `start_gemma.sh`: Script de inicialização legado (mantido para compatibilidade).
- `README.md`: Este guia completo.
- `.gitignore`: Proteção de logs e arquivos temporários.

---

## ⚠️ Solução de Problemas
- **Erro 400 (Contexto):** Verifique se o tamanho do contexto selecionado cabe na VRAM das GPUs escolhidas. Contextos muito grandes (128k+) exigem muita memória extra.
- **Status Offline:** Verifique o "Terminal Output" no rodapé para ver se o processo do `llama-server` falhou ao iniciar devido a bibliotecas faltando ou drivers NVIDIA ocupados.
- **Inversão de GPU:** O Gerenciador agora resolve isso usando o Llama-ID. Certifique-se de que a placa com mais VRAM tenha o maior peso configurado.

---
*Desenvolvido para automação e alta performance de LLMs locais.*
