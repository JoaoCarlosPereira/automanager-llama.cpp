# 🚀 Automanager Llama.cpp

> [!WARNING]
> **ESTADO ALFA:** Este projeto está em fase alfa de desenvolvimento. Atualmente, é uma solução altamente personalizada para um ambiente específico (RTX 3090 + Tesla P100) e pode não ser compatível com todas as configurações sem intervenções técnicas ou ajustes manuais. Use por sua conta e risco.

Um gerenciador web avançado e leve, baseado em **FastAPI**, projetado para orquestrar instâncias do `llama-server` com foco em máxima performance em hardware **NVIDIA Multi-GPU**.

---

## 🌟 Funcionalidades Principais

### 1. 🧠 Auto-Recuperação de Memória (Self-Healing)
- **Detecção de OOM:** Monitoramento em tempo real dos logs do servidor para detectar erros de "Out of Memory".
- **Realocação Dinâmica:** Caso um modelo falhe ao carregar, o sistema automaticamente reduz 10% da carga da GPU principal e redistribui para as secundárias, tentando novamente até estabilizar.
- **Sinalização de Falha:** Indica visualmente quando um modelo é incompatível com a capacidade total de VRAM disponível.

### 2. 🔍 Gestão Inteligente de Modelos
- **Interface Expandida:** Layout otimizado para telas largas (1800px) com melhor visibilidade para nomes de modelos longos.
- **Download via URL:** Interface integrada para baixar novos modelos diretamente para o servidor.
- **Modelo Padrão:** Opção de marcar um modelo como "Padrão" para que ele seja iniciado automaticamente junto com o serviço do sistema.
- **Exclusão de Disco:** Botão para remover arquivos `.gguf` diretamente pela interface web.

### 3. 🛠️ Controle Total de Hardware (NVIDIA)
- **Seleção Dinâmica:** Ative ou desative GPUs específicas via checkbox.
- **Distribuição de Carga:** Ajuste fino via interface. Por padrão, inicia com 100% na GPU de maior capacidade.
- **Mapeamento Anti-Inversão:** Utiliza Device IDs internos do `llama.cpp` para garantir que a carga vá exatamente para a placa correta.

### 4. 📊 Monitoramento em Tempo Real
- **Feedback Visual:** Status animados como **"ONLINE"**, **"REALOCANDO..."** e **"FALHA CRÍTICA"**.
- **Métricas de GPU:** Visualização de Utilização (%), Temperatura (°C), Consumo (W) e VRAM (MB) com barras de progresso.
- **Console de Logs:** Streaming de logs com limpeza automática a cada nova tentativa ou troca de modelo.

### 5. ⚡ Otimização de Performance
- **Janela de Contexto:** Suporte nativo até 256k, com **128K** definido como padrão da aplicação.
- **Flash Attention & MLock:** Ativados por padrão para velocidade máxima e estabilidade.
- **GPU Offloading Total:** Força o carregamento de 100% das camadas na VRAM (`-ngl 99`).

---

## 📋 Requisitos e Dependências

### Hardware e Sistema
- **SO:** Linux (Ubuntu/Debian recomendado).
- **GPU:** NVIDIA com drivers instalados e `nvidia-smi` acessível.
- **Binário:** `llama-server` deve estar no PATH ou configurado no código.

### Software (Python 3.11+)
- `fastapi`, `uvicorn`, `psutil`, `requests`, `pydantic`.

---

## 🚀 Instalação e Gestão

### 1. Estrutura de Pastas
O gerenciador busca modelos recursivamente em: `/media/docker/models/`

### 2. Configuração Persistente
As preferências são salvas em `/root/automanager_config.json`.

### 3. Comando de Serviço
O sistema roda como um serviço `systemd`:
```bash
# Reiniciar após alterações no código
sudo systemctl restart llama-manager.service
# Ver logs do serviço
journalctl -u llama-manager.service -f
```

---

## ⚠️ Solução de Problemas
- **FALHA CRÍTICA:** Se esta mensagem aparecer, o modelo+contexto escolhido é grande demais para a soma de VRAM de suas GPUs selecionadas.
- **Logs não aparecem:** Verifique se o caminho `/root/llama_server.log` tem permissões de escrita.
- **GPU Inconsistente:** Certifique-se de que o comando `nvidia-smi` está respondendo corretamente.

---
*Desenvolvido para automação e alta performance de LLMs locais.*
