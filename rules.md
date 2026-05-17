# 📜 Regras de Desenvolvimento (rules.md)

Este documento define as diretrizes técnicas, padrões de código e fluxos de trabalho obrigatórios para o projeto Automanager Llama.cpp.

## 🎯 Princípios Fundamentais
1. **Segurança de Hardware Primeiro:** Nunca tente iniciar o `llama-server` sem validar se as GPUs selecionadas estão ativas e se a soma dos pesos é exatamente 100%.
2. **Resiliência:** Sempre inclua logs detalhados para operações críticas (download, start, stop).
3. **Persistência Imutável:** Nunca altere a estrutura do `automanager_config.json` sem garantir retrocompatibilidade.

## 🛠️ Padrões de Código (Stack)
- **Backend:** FastAPI (Python 3.11+). Use `Pydantic` para validação de requests.
- **Frontend:** Vanilla JS + Tailwind CSS via CDN. Mantenha toda a lógica de UI dentro do `index()` em `llama_manager.py` para facilitar a manutenção single-file.
- **Processos:** Use `subprocess.Popen` com `os.setsid` para garantir que o `llama-server` possa ser morto de forma limpa via PGID.

## 📋 Workflows Mandatórios

### 1. Alteração de Lógica no Backend
Sempre que o `llama_manager.py` for modificado:
1. Valide a sintaxe: `python3 -m py_compile llama_manager.py`.
2. Reinicie o serviço: `systemctl restart llama-manager.service`.
3. Verifique o status: `systemctl status llama-manager.service`.

### 2. Adição de Novos Parâmetros ao Llama-Server
Ao adicionar novas flags ao comando de inicialização:
1. Atualize a classe `StartRequest`.
2. Atualize a função `execute_start`.
3. Atualize o formulário no HTML (frontend) para refletir a nova opção.
4. Garanta que o parâmetro seja persistido no `update_model_config`.

### 3. Gestão de Tokens de API
- O token deve seguir o formato `sk-...`.
- **Nunca** exiba o token em logs simples; use a interface protegida ou o arquivo de config.
- Ao renovar o token, avise o usuário que a mudança só terá efeito no próximo "Load" do modelo.

## ⚠️ Restrições e Proibições
- **NÃO** use bancos de dados externos (SQLite, Postgres). Use o JSON de configuração.
- **NÃO** remova o `pkill -9 llama-server` antes de iniciar um novo modelo (prevenção de conflito de porta).
- **NÃO** altere caminhos de sistema (`/media/docker/models`, `/root/manager.log`) sem autorização expressa.
