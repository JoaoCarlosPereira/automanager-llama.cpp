# PRD: Alerta de Nova Versão Disponível

**Slug:** version-update-alert  
**Data:** 2026-06-07  
**Status:** Aprovado

---

## Visão Geral

Administradores do Automanager Llama.cpp implantam o software em servidores próprios e o atualizam manualmente via git. Hoje não existe forma in-app de saber se o repositório remoto recebeu commits novos desde a última atualização local.

Esta funcionalidade compara o estado do repositório local com o remoto e, quando houver commits pendentes, informa o administrador por meio de um modal automático com as notas de atualização extraídas dos commits recebidos. O administrador continua responsável por aplicar a atualização no servidor.

**Público-alvo:** administradores e operadores que gerenciam instâncias self-hosted do Automanager.

---

## Objetivos

| # | Objetivo | Indicador |
|---|----------|-----------|
| O1 | Garantir que administradores saibam imediatamente ao abrir o dashboard se há versão mais recente | Modal exibido em 100% das sessões em que o remoto está à frente do local |
| O2 | Tornar transparente o que mudou entre a versão local e a disponível | Lista de commits com mensagem, autor e data visível no modal |
| O3 | Não atrapalhar a operação do dia a dia além do necessário | Opção de dispensar até o fim da sessão; sem bloqueio de outras ações do dashboard |
| O4 | Manter o escopo informativo, sem automação de update | Zero tentativas de pull/restart iniciadas pelo dashboard no MVP |

---

## Histórias de Usuário

- **HU-1:** Como administrador, quero ser avisado automaticamente ao abrir o dashboard quando existirem commits novos no remoto, para decidir se atualizo o servidor.
- **HU-2:** Como administrador, quero ver a lista de commits novos com mensagem, autor e data, para entender o impacto das mudanças antes de atualizar.
- **HU-3:** Como administrador, quero comparar rapidamente minha versão atual com a disponível no remoto, para saber o quanto estou defasado.
- **HU-4:** Como administrador, quero dispensar o alerta durante a sessão atual, para continuar uma operação urgente sem ser interrompido novamente.
- **HU-5:** Como administrador, quero que o alerta volte na próxima vez que abrir o dashboard se ainda não tiver atualizado, para não esquecer a pendência.
- **HU-6:** Como administrador em ambiente sem conectividade ou sem repositório git configurado, quero que o dashboard continue funcionando normalmente, para não depender deste recurso para operar modelos.

---

## Funcionalidades Principais

### F1 — Verificação de versão na abertura do dashboard

- Executada **uma vez por sessão de navegação**, logo após o administrador acessar o dashboard autenticado.
- Compara o commit (ou referência equivalente) do repositório local com o branch de referência no remoto.
- Se o remoto estiver à frente, aciona o fluxo de alerta (F2).
- Se estiver atualizado, em estado indeterminado ou verificação indisponível, não exibe modal automático.

### F2 — Modal automático de update-notes

- Abre automaticamente quando a verificação detecta commits novos.
- Cabeçalho com versão/referência **atual** vs **disponível** (ex.: SHA curto ou tag, conforme disponível no ambiente).
- Corpo com lista **cronológica** dos commits novos, cada item contendo:
  - Mensagem do commit
  - Autor
  - Data/hora
- Rodapé com ação primária de dispensar/fechar e texto orientando que a atualização é manual no servidor.
- Fechamento por botão, tecla Esc e clique fora do modal (padrão não bloqueante).

### F3 — Dispensar por sessão

- Ao dispensar, o modal não reaparece **na mesma sessão de navegação**.
- Em nova abertura do dashboard (nova sessão), se ainda houver commits pendentes, o modal volta a ser exibido automaticamente.
- Dispensar não equivale a "marcar como lido permanentemente" nem a ignorar versões futuras.

### F4 — Estados de indisponibilidade silenciosos

- Falha de rede, ausência de repositório git, remoto não configurado ou branch inexistente: dashboard opera normalmente, sem modal intrusivo.
- Opcionalmente, indicador discreto de "verificação indisponível" pode ser considerado na Fase 2 (fora do MVP).

---

## Experiência do Usuário

**Fluxo principal:**

1. Administrador autentica-se e o dashboard carrega.
2. Sistema verifica em background se há commits novos no remoto.
3. Se houver: modal abre automaticamente com resumo de versão e lista de commits.
4. Administrador lê as novidades e fecha ou dispensa.
5. Continua operando modelos normalmente; atualiza o servidor quando conveniente via procedimento manual existente.
6. Na próxima visita ao dashboard, se ainda defasado, o ciclo se repete.

**Considerações de UX:**

- Modal deve seguir a identidade visual do dashboard (glass/dark theme, tipografia existente).
- Lista de commits com rolagem quando houver muitos itens.
- Mensagens de commit longas truncadas com expansão ou quebra de linha legível.
- Acessibilidade: foco preso no modal enquanto aberto; retorno de foco ao fechar; contraste adequado.

---

## Restrições Técnicas de Alto Nível

- Verificação disponível apenas em instalações que são clones git com remoto configurado.
- Informação de versão exibida somente a usuários autenticados no dashboard.
- Verificação não deve impedir nem atrasar perceptivelmente o carregamento das funções críticas do dashboard (status, modelos, métricas).
- Nenhum dado de versão ou histórico de commits deve ser enviado a terceiros; tudo permanece no servidor do administrador.

---

## Fora de Escopo (Non-Goals)

- Executar `git pull`, reiniciar serviço ou qualquer atualização automática pelo dashboard.
- Exibir diff de arquivos ou detalhes de código dos commits.
- Verificação periódica em background enquanto a página permanece aberta.
- Botão manual "Verificar atualizações" (Fase 2).
- Notificações por e-mail, webhook ou push.
- Suporte a múltiplos remotos ou seleção de branch pelo usuário no MVP.
- Agrupamento de commits por tipo (fix/feature) derivado de convenções de mensagem.
- Integração com GitHub Releases ou tags semver formais.

---

## Plano de Entrega por Fases

### MVP (Fase 1)

- Verificação única na abertura do dashboard (por sessão).
- Modal automático com versão atual vs disponível.
- Lista cronológica de commits (mensagem, autor, data).
- Dispensar por sessão de navegação.
- Tratamento silencioso de falhas/indisponibilidade.

**Critério de sucesso:** administrador autenticado em instalação defasada vê o modal na abertura; após dispensar, não vê novamente na mesma sessão; na próxima abertura, vê novamente se ainda defasado.

### Fase 2

- Botão "Verificar atualizações" no cabeçalho ou configurações.
- Indicador discreto persistente quando há update disponível e o modal foi dispensado na sessão.
- Link para o repositório remoto (GitHub/GitLab).

### Fase 3

- Dispensar por versão específica (não reaparecer até commit remoto mais novo).
- Preferência de administrador para desativar verificação automática.
- Resumo agrupado por tipo de mudança quando mensagens seguirem convenção.

---

## Métricas de Sucesso

| Métrica | Meta |
|---------|------|
| Taxa de exibição do modal quando há commits pendentes | 100% das sessões elegíveis |
| Tempo até o modal aparecer após login | Perceptível como imediato (< 3s na perspectiva do usuário) |
| Taxa de falsos positivos (modal sem commits reais) | 0% |
| Impacto em operações críticas | Nenhum bloqueio de start/stop/monitoramento |
| Adoção informada | Administradores relatam saber o que mudou antes de atualizar (feedback qualitativo) |

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Modal interrompe operação urgente | Dispensar por sessão; fechamento fácil e não bloqueante |
| Mensagens de commit pouco informativas | Exibir metadados; orientar boas práticas de commit no README |
| Ambientes fork/customizados sem remoto upstream | Falha silenciosa; documentar requisito de clone do repositório oficial |
| Administrador ignora update repetidamente | Modal reaparece em cada nova sessão enquanto defasado |
| Muitos commits acumulados geram modal longo | Rolagem no modal; considerar limite de exibição na Fase 2 |

---

## Registros de Decisão de Arquitetura

- [ADR-001: Modal Automático na Abertura do Dashboard](adrs/adr-001.md) — Alerta via modal automático na abertura, verificação por sessão, atualização manual fora do app.

---

## Perguntas em Aberto

1. Qual branch de referência deve ser monitorada por padrão (`main`, `master` ou configurável)?
2. Deve haver limite máximo de commits exibidos no modal (ex.: últimos 20) quando o administrador estiver muito defasado?
3. Instalações que não são clone git (deploy por tarball/cópia) devem exibir mensagem explicativa ou simplesmente omitir o recurso?
