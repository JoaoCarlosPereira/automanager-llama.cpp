# 🚀 Automanager Llama.cpp

**Idiomas:** [English](README.md) · [Português (BR)](README.pt-BR.md)

**Painel de controle FastAPI para orquestrar o `llama-server` em servidores Linux com múltiplas GPUs NVIDIA.**

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Licença](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

> **Software em estágio alpha.** Voltado para servidores dedicados com GPU NVIDIA. APIs e padrões podem mudar.

---

## Índice

- [Início rápido](#início-rápido)
- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Integrações híbridas de plataforma](#integrações-híbridas-de-plataforma)
- [Referência da API](#referência-da-api)
- [Requisitos de hardware](#requisitos-de-hardware)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Solução de problemas](#solução-de-problemas)
- [Desenvolvimento](#desenvolvimento)
- [Licença](#licença)

---

## Início rápido

1. **Clone** o repositório no seu servidor Linux:

   ```bash
   git clone https://github.com/JoaoCarlosPereira/automanager-llama.cpp.git automanager-llama.cpp
   cd automanager-llama.cpp
   ```

2. **Instale** com o script Quick-Install (requer root, Ubuntu/Debian, drivers NVIDIA e `llama-server` no `PATH`):

   ```bash
   sudo bash installer/setup.sh
   ```

3. **Abra o dashboard** em `http://<ip-do-servidor>:8000/` (credenciais padrão na primeira execução; altere a senha de admin após o login).

---

## Funcionalidades

| Funcionalidade | Descrição | Status |
|----------------|-----------|--------|
| **Orquestração multi-GPU** | Seleção de GPUs, tensor split e `CUDA_VISIBLE_DEVICES` mapeado para o `llama-server` | Estável |
| **Auto-recuperação de OOM** | Monitora logs; em OOM reduz peso da GPU principal e tenta novamente | Estável |
| **Biblioteca de modelos** | Varredura recursiva de `.gguf`, renomear, excluir, modelo padrão, configurações por modelo | Estável |
| **Download por URL** | Baixa modelos na árvore de modelos com acompanhamento de progresso | Estável |
| **Métricas em tempo real** | CPU, RAM e por GPU: utilização, temperatura, potência, VRAM via `nvidia-smi` | Estável |
| **Ciclo de status** | OFFLINE → INICIANDO → ONLINE → (REALOCANDO) → PARANDO; UI esmaece métricas offline | Estável |
| **Stream de logs** | Console SSE com saída do `llama-server` | Estável |
| **Auth por sessão + API key** | Cookie de sessão e Bearer token para clientes API | Estável |
| **Modelos híbridos de plataforma** | Detecta Codex, Claude Code e Google Antigravity e expõe essas ferramentas no mesmo fluxo do dashboard/API | MVP |
| **Quick-Install** | `installer/setup.sh` idempotente (venv, systemd, health check) | Estável |
| **Código modular** | Lógica em módulos; `llama_manager.py` como raiz de composição (rotas + UI) | Estável |

Configurações padrão de inferência: **Flash Attention**, **mlock**, offload total de camadas (`-ngl 99`) e janela de contexto padrão de **65536** tokens (`DEFAULT_CONTEXT_SIZE`).

---

## Arquitetura

O Automanager é um **plano de controle** (FastAPI na porta **8000**) que gerencia um único processo filho **`llama-server`** (porta **8085**). O dashboard é HTML/JS embutido servido em `GET /`.

```
┌─────────────┐     HTTP :8000      ┌──────────────────────────────────────┐
│  Navegador  │ ──────────────────► │  llama_manager.py (FastAPI)          │
│  Dashboard  │ ◄── SSE /status ─── │  Rotas · auth · UI · injeção de deps   │
└─────────────┘                     └───────────┬──────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
           ┌────────────────┐          ┌─────────────────┐          ┌─────────────────┐
           │ config_manager │          │  gpu_manager    │          │ process_manager │
           │ TokenManager   │          │  nvidia-smi     │          │ OOMWatchdog     │
           │ AuthManager    │          │  tensor split   │          │ subprocess      │
           └────────┬───────┘          └────────┬────────┘          └────────┬────────┘
                    │                           │                            │
                    │                           │                            ▼
                    │                           │                   ┌─────────────────┐
                    │                           └──────────────────►│  llama-server   │
                    │                                               │  :8085 / GPUs   │
                    ▼                                               └─────────────────┘
           /root/automanager_config.json
           /media/docker/models/*.gguf
```

**Layout modular:**

| Módulo | Responsabilidade |
|--------|------------------|
| `llama_manager.py` | App FastAPI, rotas, auth, dashboard embutido |
| `config_manager.py` | Config JSON, chaves API, auth admin |
| `gpu_manager.py` | Detecção de GPU, tensor split, `CUDA_VISIBLE_DEVICES` |
| `process_manager.py` | Subprocesso `llama-server`, watchdog OOM |
| `model_manager.py` | Varredura de modelos, renomear/excluir, downloads |
| `platform_manager.py` | Detecção no startup e ciclo de vida do sidecar CLIProxyAPI para integrações de plataforma |
| `proxy_router.py` | Seleção de backends do proxy inteligente, sessões sticky e roteamento local/plataforma |
| `log_manager.py` | Logs rotativos em `logs/`, streaming SSE |
| `schemas.py` | Modelos Pydantic de requisição/resposta |
| `paths.py` | Resolução de caminhos de instalação (`paths.json`) |
| `installer/setup.sh` | Quick-Install: deps, venv, systemd, health check |
| `installer/uninstall.sh` | Remove servico systemd e venv; `--purge` remove config/logs |

---

## Integrações híbridas de plataforma

O AutoManager pode mostrar ferramentas CLI baseadas em assinatura ao lado dos
modelos locais `.gguf`. O MVP suporta Codex, Claude Code e Google Antigravity.

- A detecção roda uma vez quando o AutoManager inicia. Reinicie o AutoManager
  após instalar, remover ou mover uma das ferramentas suportadas.
- O AutoManager não coleta credenciais de provedor, chaves de API ou logins de
  plataforma. Ele usa a autenticação que já existe nas ferramentas CLI
  instaladas.
- Iniciar um card de plataforma inicia um sidecar local compartilhado do
  CLIProxyAPI. Integrações ativas aparecem em `/status`, e os IDs reais de
  modelo retornados pelo sidecar passam por `/v1/models`.
- Backends de plataforma ficam fora do proxy inteligente até você habilitar a
  participação daquele backend. O principal do proxy pode ser um
  `primary_model_path` local ou um `primary_backend_id` de plataforma.
- Se uma CLI suportada for detectada, mas o CLIProxyAPI estiver ausente, o card
  continua visível com o motivo de indisponibilidade.

---

## Referência da API

URL base: `http://<host>:8000`. A maioria dos endpoints exige **cookie de sessão** ou **`Authorization: Bearer <api-key>`** (obtida em `GET /api/key`). `GET /logs` faz stream sem auth para o console embutido.

| Método | Caminho | Descrição |
|--------|---------|-----------|
| `GET` | `/` | UI do dashboard (HTML) |
| `GET` | `/status` | Estado de runtime local e de plataformas |
| `GET` | `/metrics` | CPU, RAM, GPU (uso / temp / potência / VRAM) |
| `GET` | `/models` | Lista modelos `.gguf` e cards de integrações de plataforma |
| `GET` | `/downloads` | Progresso de downloads ativos |
| `POST` | `/downloads` | Inicia download: `{ "url": "...", "filename": "opcional" }` |
| `GET` | `/logs` | Stream SSE do log do servidor |
| `GET` | `/config` | Configuração atual (hash de senha omitido) |
| `POST` | `/start` | Inicia servidor: `{ "path", "gpu_weights", "context_size", "mmproj_path?", "split_mode?" }` |
| `POST` | `/stop` | Para o `llama-server` |
| `POST` | `/platforms/{backend_id}/start` | Inicia uma integração de plataforma e o sidecar compartilhado |
| `POST` | `/platforms/{backend_id}/stop` | Para uma integração; para o sidecar quando nenhuma plataforma fica ativa |
| `POST` | `/models/proxy` | Atualiza elegibilidade de proxy para `model_path` local ou `backend_id` de plataforma |
| `POST` | `/proxy/config` | Atualiza o proxy inteligente, incluindo `primary_model_path` ou `primary_backend_id` |
| `GET` | `/v1/models` | Lista OpenAI-compatible de servidores locais e sidecar de plataforma ativo |
| `*` | `/v1/{path}` | Encaminhamento OpenAI-compatible para servidores locais ou sidecar de plataforma |
| `POST` | `/set_default` | Define modelo padrão: `{ "path": string \| null }` |
| `POST` | `/rename` | Renomeia arquivo: `{ "path", "new_name" }` |
| `POST` | `/delete` | Exclui arquivo: `{ "path" }` |
| `POST` | `/api/auth/login` | Login: `{ "username", "password" }` |
| `POST` | `/api/auth/logout` | Encerra sessão |
| `POST` | `/api/auth/change-password` | Altera senha admin |
| `GET` | `/api/key` | Retorna ou cria chave API |
| `POST` | `/api/key/renew` | Rotaciona chave API |

**Portas:** Manager `0.0.0.0:8000` · `llama-server` `0.0.0.0:8085`

---

## Requisitos de hardware

| Requisito | Detalhes |
|-----------|----------|
| **SO** | Linux — Ubuntu 22.04+ ou Debian 11+ (alvo do Quick-Install) |
| **GPU** | Uma ou mais GPUs NVIDIA com drivers funcionando |
| **`nvidia-smi`** | Deve executar com sucesso e listar ao menos uma GPU |
| **`llama-server`** | Binário pré-compilado no `PATH` (não instalado pelo setup) |
| **RAM / VRAM** | Depende do modelo e `context_size`; multi-GPU usa tensor split |
| **Disco** | Espaço para `.gguf` em `models_dir` do `paths.json` (padrão `data/models/`) |

A instalação do CUDA fica a cargo do ambiente se o seu build do `llama-server` exigir; o instalador não instala drivers nem CUDA.

---

## Instalação

### Pré-requisitos

- Ubuntu 22.04+ ou Debian 11+
- Python 3.11+
- Acesso `sudo` / root
- Drivers NVIDIA + `nvidia-smi`
- `llama-server` no `PATH`
- Clone Git deste repositório

### Quick-Install (recomendado)

Na raiz do repositório:

```bash
sudo bash installer/setup.sh
```

O script irá:

1. Verificar Ubuntu/Debian, privilégios root e Python 3.11+  
2. Instalar `python3`, `python3-venv`, `python3-pip`, `python3-dev`, `curl`, `git`, `lsb-release`  
3. Avisar se `llama-server` estiver ausente  
4. Exigir ao menos uma GPU NVIDIA  
5. Criar `paths.json` a partir de `paths.json.example` quando ausente  
6. Criar ou atualizar `.venv/` e executar `pip install -r requirements.txt`  
7. Criar diretórios configurados (`data/models`, `data/`, `logs/`) via `paths.py`  
8. Instalar e habilitar `llama-manager.service`  
9. Executar `curl http://localhost:8000/` (dashboard público) e exibir a URL do dashboard  

O script é **idempotente**: pode ser reexecutado; atualiza dependências, reescreve o unit do systemd e reinicia o serviço.

### Instalação manual

```bash
cd automanager-llama.cpp
cp paths.json.example paths.json   # pule se paths.json já existir
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from paths import ensure_directories; ensure_directories()"
python llama_manager.py
```

### systemd (manual)

Exemplo de unit (o Quick-Install grava arquivo similar em `/etc/systemd/system/llama-manager.service`):

```ini
[Service]
WorkingDirectory=/caminho/para/automanager-llama.cpp
ExecStart=/caminho/para/automanager-llama.cpp/.venv/bin/python llama_manager.py
User=root
Restart=on-failure
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-manager.service
journalctl -u llama-manager.service -f
```

---

## Configuração

| Item | Padrão / caminho |
|------|------------------|
| **Porta do manager** | `8000` |
| **Porta do servidor** | `8085` |
| **Contexto padrão** | `65536` tokens (`DEFAULT_CONTEXT_SIZE`) |
| **Config de caminhos** | `paths.json` na raiz do install (de `paths.json.example`; gitignored) |
| **Diretório de modelos** | `data/models/` (relativo à raiz; editável no painel ou via `POST /models/dir`) |
| **Config principal** | `data/automanager_config.json` (modelo padrão, GPU/contexto por modelo, auth) |
| **Log do manager** | `logs/manager.log` |
| **Log do servidor** | `logs/server.log` |
| **Rotação de logs** | 10 MB por arquivo, 3 backups (`RotatingFileHandler`) |

Edite `paths.json` para caminhos absolutos ou layout legado. Instalações em `/root` com `/media/docker/models` ou `/root/automanager_config.json` existentes usam defaults legados automaticamente via `paths.py`.

---

## Solução de problemas

| Sintoma | O que verificar |
|---------|-----------------|
| **Health check falhou após instalar** | `systemctl status llama-manager.service` e `journalctl -u llama-manager.service -n 50` |
| **FALHA CRÍTICA / OOM** | Modelo + `context_size` excede VRAM; reduza contexto ou use quantização menor |
| **Nenhum modelo listado** | `models_dir` do `paths.json` existe e contém `.gguf`; permissões do usuário do serviço |
| **`llama-server` não encontrado** | Instale o binário e inclua o `PATH` no unit do systemd |
| **Métricas de GPU vazias** | `nvidia-smi` funciona como usuário do serviço; driver incompatível |
| **401 nas chamadas API** | Faça login no dashboard ou use `Authorization: Bearer <chave>` de `GET /api/key` |
| **Logs não atualizam na UI** | Verifique `logs/server.log` (ou caminho do `paths.json`) e permissões de escrita |

---

## Desenvolvimento

### Executar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python llama_manager.py
```

### Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Testes unitários cobrem config/token, GPU e watchdog OOM; testes de integração exercitam auth e rotas com mocks.

### Estrutura do projeto

```
automanager-llama.cpp/
├── llama_manager.py      # App FastAPI, rotas, UI
├── config_manager.py     # Config, tokens, auth
├── gpu_manager.py        # GPU e tensor split
├── process_manager.py    # Subprocesso + OOM watchdog
├── model_manager.py      # Modelos e downloads
├── log_manager.py        # Logs e SSE
├── schemas.py            # Modelos Pydantic
├── paths.py              # Resolução de caminhos (paths.json)
├── paths.json.example    # Template de caminhos para novas instalações
├── installer/setup.sh      # Quick-Install
├── installer/uninstall.sh  # Remove servico/venv (--purge para config/logs)
├── static/js/            # Assets do dashboard (ex.: fundo Pac-Man)
├── logs/                 # Logs em runtime (gitignored)
├── requirements.txt
├── requirements-dev.txt  # Prod + pytest/httpx para desenvolvimento
├── tests/
└── start_llama.sh        # Exemplo de lançamento manual do llama-server
```

Padrões e fluxos: [rules.md](rules.md). Notas para agentes: [CLAUDE.md](CLAUDE.md).

### Contribuindo

1. Faça fork e branch a partir de `main`.  
2. Mantenha mudanças focadas; siga o estilo existente.  
3. Execute `pytest` antes de abrir um PR.  
4. Atualize README e tabelas de API se alterar endpoints.

---

## Licença

Copyright 2026 Automanager Llama.cpp contributors.

Licenciado sob a **Apache License, Version 2.0**. Veja [LICENSE](LICENSE) para o texto completo.
