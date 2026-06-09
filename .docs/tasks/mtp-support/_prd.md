# PRD: Suporte MTP no Automanager

## Visão Geral

Operadores do automanager hoje precisam editar manualmente flags do llama-server para ativar Multi-Token Prediction (MTP) — uma técnica que acelera inferência em modelos compatíveis. Esta feature expõe MTP no dashboard com os mesmos padrões dos campos existentes (contexto, parallel, batch, thinking), eliminando edição manual de linha de comando.

**Problema:** Configurar MTP exige conhecimento técnico das flags `--spec-type draft-mtp` e `--spec-draft-n-max`, fora do fluxo visual do dashboard.

**Público:** Administradores e operadores que gerenciam instâncias llama-server via automanager.

**Valor:** Operação simplificada — MTP configurável por modelo, persistido e restaurado automaticamente ao selecionar o modelo.

## Objetivos

1. Eliminar necessidade de editar flags MTP manualmente ao subir modelos.
2. Permitir configuração MTP por modelo, consistente com demais parâmetros.
3. Manter compatibilidade com fluxos existentes (start manual, auto-balance, config salva).

**Critérios de sucesso:** Operador configura MTP inteiramente pelo dashboard; config persiste entre sessões; auto-balance respeita config MTP informada.

## Histórias de Usuário

**Operador de inferência**

- Como operador, quero ligar/desligar MTP por modelo para acelerar inferência sem editar scripts.
- Como operador, quero definir quantos tokens de predição usar para ajustar trade-off velocidade vs. estabilidade.
- Como operador, quero que minha config MTP seja restaurada ao selecionar um modelo previamente configurado.

**Administrador de infraestrutura**

- Como administrador, quero que modelos sem suporte MTP iniciem normalmente mesmo com MTP ligado, sem bloquear operação.

## Funcionalidades Principais

### F1 — Toggle de ativação MTP

- Checkbox/toggle no painel de configuração, padrão visual do `thinking-toggle`.
- Estado persistido por modelo em configuração salva.
- Padrão: desligado se modelo nunca foi configurado; herda última config se já configurado.

### F2 — Campo numérico de tokens de predição

- Input numérico livre, padrão visual do `parallel-slots`.
- Valor padrão sugerido: 3 tokens.
- Sempre visível e editável, independente do estado do toggle.
- Persistido por modelo junto com demais configurações.

### F3 — Persistência e restauração por modelo

- Ao selecionar modelo na lista, campos MTP refletem config salva.
- Ao iniciar modelo (manual ou auto-balance), config MTP é enviada junto com demais parâmetros.
- Auto-balance usa mesma config MTP informada pelo operador durante sondagem.

### F4 — Comportamento em modelos incompatíveis

- Start permitido normalmente.
- Config MTP ignorada silenciosamente quando modelo não suporta MTP.
- Sem alertas ou bloqueios neste MVP.

## Experiência do Usuário

**Fluxo principal:**

1. Operador seleciona modelo na lista.
2. Dashboard restaura toggle MTP e tokens da config salva (ou padrões).
3. Operador ajusta toggle e/ou quantidade de tokens.
4. Operador clica CARREGAR (ou inicia auto-balance).
5. Servidor sobe com MTP aplicado se compatível e toggle ligado.

**Layout:** Toggle MTP + badge ON/OFF e input numérico posicionados no painel de configuração existente, sempre visíveis, com labels e tooltips explicativos no mesmo estilo dos campos atuais.

**Onboarding:** Tooltip no toggle explica que MTP acelera inferência em modelos compatíveis; tooltip no campo numérico indica que valores típicos são 2–3 e que só aplica com MTP ligado.

## Restrições Técnicas de Alto Nível

- Deve integrar-se ao fluxo existente de start/stop e configuração por modelo.
- Deve respeitar o contrato visual e de persistência dos campos atuais.
- Auto-balance deve propagar config MTP durante sondagem.
- Compatível com modelos multimodais (mmproj) já suportados.

## Fora de Escopo (Non-Goals)

- Detecção visual de compatibilidade MTP por modelo.
- Alertas ou bloqueios para modelos incompatíveis.
- Campo de limiar de confiança (`spec-draft-p-min`).
- Presets "Conservador / Balanceado / Agressivo".
- Ajuste adaptativo de tokens durante geração.
- Métricas de acceptance rate MTP no dashboard.
- Configuração MTP global (fora do escopo por modelo).

## Plano de Entrega por Fases

### MVP (Fase 1)

- Toggle MTP + campo numérico sempre visíveis.
- Persistência por modelo e restauração ao selecionar.
- Propagação no start manual e auto-balance.
- Ignorar silenciosamente em modelos incompatíveis.
- **Critério de sucesso:** Operador configura e inicia modelos MTP-compatíveis sem editar flags manualmente.

### Fase 2 (futuro)

- Indicador de compatibilidade MTP por modelo.
- Aviso visual quando MTP está ligado mas não aplicável.

### Fase 3 (futuro)

- Parâmetros avançados (p-min, presets).
- Métricas de throughput/acceptance no dashboard.

## Métricas de Sucesso

- Redução a zero de edições manuais de flags MTP para operadores do dashboard.
- 100% das configs MTP persistidas e restauradas corretamente por modelo.
- Auto-balance completa com config MTP preservada.
- Operadores reportam ganho percebido de throughput em modelos MTP-compatíveis (qualitativo).

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Operador assume MTP ativo em modelo incompatível | Tooltips; Fase 2 com indicador de compatibilidade |
| Valor de tokens inadequado degrada performance | Default 3; tooltip com faixa recomendada 2–3 |
| Confusão com campo visível e toggle off | Label indica que tokens só aplicam com MTP ligado |
| Adoção baixa por falta de awareness | Documentação interna sobre modelos MTP-compatíveis |

## Registros de Decisão de Arquitetura

- [ADR-001: Campos MTP sempre visíveis no painel](adrs/adr-001.md) — Toggle e input numérico permanentes no painel principal.
- [ADR-002: Ignorar silenciosamente MTP em modelos incompatíveis](adrs/adr-002.md) — Start permitido; config MTP não aplicada sem bloqueio ou aviso.

## Perguntas em Aberto

1. Faixa mínima/máxima aceitável para o campo numérico de tokens (validação de input)?
2. Deve haver valor default global (3) ou default diferenciado por família de modelo na Fase 2?
3. Como identificar compatibilidade MTP de forma confiável (delegado à TechSpec)?
