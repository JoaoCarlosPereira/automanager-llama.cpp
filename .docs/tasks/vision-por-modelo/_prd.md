# PRD: Vision por Modelo no Repositório

## Visão Geral

O Automanager Llama.cpp gerencia modelos de linguagem (GGUF) e sua execução via llama-server. Modelos multimodais exigem um projetor de visão (`mmproj`) pareado ao modelo de linguagem. Hoje, a escolha do projetor é feita por um seletor global na barra de controles, o que mistura projetores de modelos diferentes e não oferece um fluxo claro para importar o par correto.

Este produto reposiciona a gestão de visão para dentro de cada item da lista de modelos no repositório: o operador importa projetores exclusivos de um modelo, escolhe qual usar quando houver mais de um, e a preferência é lembrada automaticamente. O campo Vision global deixa de existir.

**Público-alvo:** administradores e operadores que carregam modelos multimodais (LLaVA, Gemma 3 Vision, Qwen2-VL, etc.) no painel do Automanager.

**Valor:** reduz erros de pareamento modelo/projetor, simplifica a barra de controles e torna explícito quais modelos têm capacidade de visão configurada.

## Objetivos

1. Eliminar o seletor global de Vision da barra de controles do painel.
2. Permitir importar um ou mais projetores de visão vinculados exclusivamente a cada modelo de linguagem.
3. Exibir seleção de projetor apenas no item do modelo e somente quando existir ao menos um projetor associado.
4. Restaurar automaticamente o projetor escolhido (ou o primeiro disponível) ao carregar ou selecionar o modelo.
5. Manter a suíte de testes automatizados passando, com cobertura das novas regras de comportamento.

**Marco alvo:** entrega em fase única (MVP completo), sem dependência de funcionalidades futuras.

## Histórias de Usuário

### Operador de modelos multimodais

- Como operador, quero importar um projetor de visão informando um link de download diretamente no item do meu modelo de linguagem, para não precisar copiar arquivos manualmente no servidor.
- Como operador, quero escolher qual projetor usar quando meu modelo tiver mais de um baixado, para testar variantes ou quantizações diferentes.
- Como operador, quero que o sistema selecione automaticamente o primeiro projetor disponível quando eu ainda não escolhi um, para iniciar o modelo com visão sem passos extras.
- Como operador, quero que minha escolha de projetor seja lembrada quando eu voltar a esse modelo, para não reconfigurar a cada sessão.
- Como operador, quero que modelos sem projetor não mostrem controles de visão, para a interface refletir apenas o que está disponível.

### Operador de modelos somente texto

- Como operador de modelos puramente textuais, quero que itens sem projetor permaneçam visualmente simples (sem combobox de visão), para não me distrair com opções irrelevantes.

### Casos de borda

- Como operador, ao excluir o projetor selecionado, quero que o sistema passe a usar o próximo disponível ou oculte o combobox se nenhum restar.
- Como operador com modelo em execução, quero que ações destrutivas (excluir projetor) respeitem as mesmas proteções já existentes para o modelo em si.

## Funcionalidades Principais

### F1 — Remoção do seletor global de Vision

**O que faz:** Remove o campo Vision (`mmproj-path`) da barra de controles superior do painel.

**Por que importa:** Evita associação incorreta entre modelos e projetores de contextos diferentes; reduz poluição visual.

**Comportamento:**
- O painel de controles não exibe mais rótulo "Vision" nem combobox global.
- Configurações de GPU, contexto, batch, split, thinking e MTP permanecem inalteradas.
- Ao iniciar um modelo, o projetor vem da configuração salva daquele modelo (ou do primeiro disponível), não de um campo global.

### F2 — Importação de projetor por modelo

**O que faz:** Adiciona um botão em cada item da lista de modelos do repositório que abre um modal para informar o link de download do projetor.

**Por que importa:** Centraliza o fluxo de obtenção do par multimodal no contexto do modelo correto.

**Comportamento:**
- O botão fica visível em todos os itens de modelo da lista (ícone/ação de importar visão).
- O modal solicita URL de download (mesmo padrão de UX do download de modelos GGUF).
- O arquivo baixado fica associado exclusivamente ao modelo que originou o download (mesmo diretório ou subpasta do modelo).
- O progresso do download aparece na área de downloads já existente no painel.
- Após conclusão, a lista de modelos atualiza e o combobox de visão do item passa a refletir o novo projetor.

### F3 — Seleção de projetor no item do modelo

**O que faz:** Exibe um combobox no próprio item da lista para escolher qual projetor usar, somente quando o modelo tiver ao menos um projetor associado.

**Por que importa:** Permite múltiplos projetores por modelo com escolha explícita e contextual.

**Comportamento:**
- Se zero projetores: combobox não é renderizado.
- Se um ou mais projetores: combobox visível com nomes de arquivo legíveis.
- Ao abrir a lista ou selecionar o modelo, se não houver escolha salva, seleciona automaticamente o primeiro projetor encontrado.
- Ao mudar a seleção, persiste imediatamente (ou ao salvar configuração do modelo, conforme padrão existente de `model_configs`).
- Ao carregar o modelo (`CARREGAR`), usa o projetor selecionado no combobox daquele item.

### F4 — Persistência por modelo

**O que faz:** Salva o caminho do projetor selecionado junto às demais configurações já persistidas por modelo (`mmproj_path` em `model_configs`).

**Por que importa:** Garante continuidade entre sessões e reinícios do manager.

**Comportamento:**
- Cada modelo mantém seu próprio `mmproj_path` independente dos demais.
- Renomear ou mover modelo atualiza referências salvas (comportamento já esperado para `model_configs`).
- Selecionar outro modelo na lista não altera a configuração de visão dos demais.

### F5 — Detecção e listagem de projetores associados

**O que faz:** Identifica arquivos de projetor vinculados a cada modelo (por convenção de nome, pasta compartilhada ou metadado de importação).

**Por que importa:** Alimenta o combobox e a seleção automática sem intervenção manual de caminhos.

**Comportamento:**
- Projetores importados via botão do modelo entram na lista daquele modelo.
- Projetores já presentes no disco na pasta do modelo (ex.: `*-mmproj.gguf`) também aparecem.
- Projetores de outros modelos ou pastas não aparecem no combobox do item.

## Experiência do Usuário

### Fluxo principal — primeiro uso com visão

1. Operador localiza o modelo de linguagem na lista do repositório.
2. Clica no botão de importar visão no item do modelo.
3. Informa URL do arquivo `mmproj` no modal e confirma.
4. Acompanha progresso na seção de downloads.
5. Ao concluir, o combobox de visão aparece no item com o projetor selecionado automaticamente.
6. Operador ajusta GPU/contexto se necessário e clica em CARREGAR.
7. O modelo sobe com visão habilitada via projetor correto.

### Fluxo — retorno a modelo já configurado

1. Operador seleciona modelo na lista.
2. Combobox de visão mostra o projetor previamente salvo (ou o primeiro, se o salvo não existir mais).
3. CARREGAR usa essa seleção sem passos adicionais.

### Fluxo — múltiplos projetores

1. Operador importa segundo projetor pelo mesmo botão.
2. Combobox passa a listar ambos.
3. Operador escolhe o desejado; escolha persiste para próximas cargas.

### Considerações de UI/UX

- Botão de importar visão com ícone de olho ou download secundário, tooltip "Importar projetor de visão".
- Combobox compacto, alinhado à linha de ações do item (renomear, excluir, padrão, carregar).
- Modal consistente com o padrão visual glass/slate do painel existente.
- Em viewports móveis, combobox e botão permanecem acessíveis sem quebrar o layout do item.

### Acessibilidade

- Botão e combobox com `title`/`aria-label` descritivos.
- Modal com foco inicial no campo de URL e fechamento por Escape.

## Restrições Técnicas de Alto Nível

- Deve integrar-se ao fluxo de download e persistência de configuração já existentes no Automanager.
- Deve continuar compatível com o pareamento exigido pelo llama-server (`--mmproj` por sessão).
- Alterações devem ser validadas por testes unitários; a suíte existente não pode regredir.
- Projetores importados permanecem no filesystem do repositório de modelos configurado (`paths.json`).

## Fora de Escopo (Non-Goals)

- Validação automática de compatibilidade arquitetural entre modelo de linguagem e projetor.
- Conversão de formatos (safetensors → GGUF/mmproj) dentro do produto.
- Biblioteca global compartilhada de projetores entre modelos diferentes.
- Configuração de parâmetros avançados de visão (`image-min-tokens`, `image-max-tokens`, offload de mmproj).
- Exclusão individual de projetor via UI (pode ser fase futura; exclusão manual no disco continua possível).
- Suporte a modelos vision que não usem arquivo `mmproj` separado.

## Plano de Entrega por Fases

### MVP (Fase 1) — Entrega única

**Inclui:**
- Remoção do seletor global Vision
- Botão + modal de importação por modelo
- Combobox condicional por item
- Seleção automática do primeiro projetor
- Persistência `mmproj_path` por modelo
- Atualização de testes (HTML contract, scanner, config, JS)

**Critério de sucesso para concluir:**
- Operador importa mmproj pelo item do modelo e carrega com visão sem usar campo global.
- Modelos sem projetor não exibem combobox.
- `npm test` / `pytest` passam sem regressões.

### Fase 2 — Melhorias opcionais (futuro)

- Excluir projetor individual pela UI
- Indicador visual "Vision ativo" no item do modelo
- Validação heurística de compatibilidade (nome/arquitetura)

### Fase 3 — Ecossistema (futuro)

- Sugestão automática de URL de mmproj a partir do modelo selecionado (catálogo Hugging Face)

## Métricas de Sucesso

| Métrica | Meta |
|---------|------|
| Regressão de testes | 100% da suíte existente passando após a mudança |
| Novos testes cobrindo regras de visão por modelo | ≥ 5 casos (UI condicional, persistência, seleção automática, remoção global, download associado) |
| Tempo para configurar visão em modelo novo | ≤ 3 ações (botão → URL → carregar) |
| Erros de pareamento reportados | Redução perceptível (qualitativo nas primeiras semanas) |

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Operador importa projetor incompatível | Mensagem no modal orientando a usar mmproj da mesma família do modelo; link para documentação llama.cpp |
| Confusão após remoção do campo global | Tooltip no botão de importar; release note no painel na primeira visita pós-atualização |
| Projetor salvo deixa de existir no disco | Fallback para primeiro disponível; ocultar combobox se lista vazia |
| Layout congestionado em itens da lista | Combobox só quando necessário; botão compacto com ícone |

## Registros de Decisão de Arquitetura

- [ADR-001: Visão (mmproj) gerenciada por modelo na lista do repositório](adrs/adr-001.md) — Remove seletor global; importação e combobox exclusivos por item de modelo.

## Perguntas em Aberto

1. O botão de importar visão deve permanecer habilitado com o modelo em execução, ou seguir a mesma regra de ocultar ações destrutivas?
2. Projetores detectados automaticamente por nome (fora do fluxo de importação) devem entrar no combobox do modelo correspondente na primeira versão, ou apenas arquivos importados explicitamente pelo botão?
3. A persistência da seleção de projetor deve ocorrer ao mudar o combobox (auto-save) ou apenas ao carregar o modelo (save-on-start)?
