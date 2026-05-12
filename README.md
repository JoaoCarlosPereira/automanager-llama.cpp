# 🚀 Automanager Llama.cpp

> [!WARNING]
> **ESTADO BETA:** Este projeto está em fase beta de desenvolvimento. Atualmente, é uma solução altamente personalizada para um ambiente específico (RTX 3090 + Tesla P100) e pode não ser compatível com todas as configurações sem intervenções técnicas ou ajustes manuais. Use por sua conta e risco.

Um gerenciador web avançado e leve, baseado em **FastAPI**, projetado para orquestrar instâncias do `llama-server` com foco em máxima performance em hardware **NVIDIA Multi-GPU**.

---

## 🌟 Funcionalidades Principais

### 1. 🔍 Gestão Inteligente de Modelos
- **Descoberta Automática:** Varredura recursiva no diretório `/media/docker/models`. Identifica arquivos `.gguf` e os organiza por subdiretórios.
- **Download via URL:** Interface integrada para baixar novos modelos diretamente para o servidor.
- **Modelo Padrão:** Opção de marcar um modelo como "Padrão" para que ele seja iniciado automaticamente junto com o serviço do sistema.

### 2. 🛠️ Controle Total de Hardware (NVIDIA)
- **Seleção Dinâmica:** Ative ou desative GPUs específicas via checkbox.
- **Distribuição de Carga (Tensor Split):** Ajuste fino da porcentagem de VRAM que cada GPU deve carregar.
- **Mapeamento Anti-Inversão:** Utiliza a ordem de detecção interna do `llama.cpp` para garantir que a carga configurada vá exatamente para a placa correta.
- **Priorização Automática:** Sugere configurações otimizadas baseadas nos pesos definidos.

### 3. 📊 Monitoramento Detalhado em Tempo Real
- **Métricas de Sistema:** Acompanhamento vivo do uso de **CPU** e **Memória RAM**.
- **Métricas de GPU:** Visualização em tempo real de **Utilização (%)**, **Temperatura (°C)**, **Consumo de Energia (W)** e **VRAM (MB)** com barras de progresso.
- **Console de Execução:** Streaming de logs em tempo real diretamente na interface web.

### 4. ⚡ Otimização de Performance
- **Contexto Configurável:** Seleção de janelas de contexto (ex: 2k, 32k, 128k) via interface.
- **Flash Attention & MLock:** Ativados por padrão para garantir velocidade máxima e estabilidade.
- **GPU Offloading Total:** Configurado automaticamente para carregar 100% das camadas na VRAM (`-ngl 99`).

### 5. 🔌 Conectividade e Chat
- **API OpenAI-Compatible:** Servidor fixo na porta `8085/v1`, pronto para integração com ferramentas externas.
- **Acesso Direto ao Chat:** Botão dedicado para abrir a interface nativa de chat do `llama.cpp`.

---

## 📋 Requisitos e Dependências

### Hardware e Sistema
- **SO:** Linux (Ubuntu/Debian recomendado).
- **GPU:** NVIDIA com drivers instalados e `nvidia-smi` acessível.
- **Binário:** `llama-server` deve estar no PATH ou em local configurado no script.

### Software (Python 3.11+)
- `fastapi`, `uvicorn`, `psutil`, `requests`.

---

## 🚀 Instalação e Configuração

### 1. Estrutura de Pastas
O gerenciador busca modelos em: `/media/docker/models/`

### 2. Configuração Persistente
As preferências (como o modelo padrão) são salvas em `/root/automanager_config.json`.

### 3. Gestão do Serviço
O sistema é projetado para rodar como um serviço `systemd`:
```bash
# Reiniciar após alterações no código
sudo systemctl restart llama-manager.service
# Ver logs do serviço
journalctl -u llama-manager.service -f
```

---

## ⚠️ Observações de Uso e Solução de Problemas
- **Personalização:** Muitos caminhos e configurações estão "hardcoded" para o ambiente original. Se for utilizar em outro servidor, verifique as variáveis de caminho dentro de `llama_manager.py`.
- **Memória:** Contextos muito grandes podem causar falha na inicialização se excederem a VRAM disponível.
- **GPU Inconsistente:** Se as métricas de GPU não aparecerem, certifique-se de que o comando `nvidia-smi` está funcionando corretamente.

---
*Desenvolvido para automação e alta performance de LLMs locais.*
