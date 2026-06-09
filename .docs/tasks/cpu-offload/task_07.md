---
status: pending
title: Injetar linha da CPU no HTML do dashboard
type: backend
complexity: low
dependencies:
  - task_02
---

# Tarefa 07: Injetar linha da CPU no HTML do dashboard

## Visão Geral

Esta tarefa adiciona a linha da CPU na tabela "Recursos de GPU & Configuração" no template HTML injetado pela função `index()` em `llama_manager.py`. A linha segue exatamente o mesmo padrão visual das linhas de GPU: checkbox de ativação, nome do processador, métricas de uso, RAM usada/total com barra de progresso, e input de peso com checkbox de fixar.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- A linha da CPU DEVE ser injetada após as linhas de GPU, antes do fechamento do `<tbody>`
- A linha DEVE ter `id="cpu-row"` e `data-device="cpu"`
- A linha DEVE conter: checkbox `.cpu-checkbox`, nome `.device-name`, input `.cpu-weight` (min=0, max=70), checkbox `.cpu-pin`
- A linha DEVE ter colunas para: CPU usage (`.device-util-val` + `.device-util-bar`), RAM usada/total (`.device-vram-text` + `.device-vram-bar`)
- A CPU NÃO DEVE ter radio button "principal" (sem coluna de radio button)
- O checkbox de ativação DEVE estar checked por padrão
- O input de peso DEVE ter `max="70"` no HTML
- Os valores iniciais DEvem ser populados com dados de `detect_cpu_info()`

</requirements>

## Subtarefas
- [ ] 07.1 Chamar `detect_cpu_info()` na função `index()` para obter dados do CPU
- [ ] 07.2 Construir o HTML string da linha CPU no padrão das linhas GPU (f-string)
- [ ] 07.3 Inserir a linha CPU após as linhas GPU, antes do fechamento do `<tbody>`
- [ ] 07.4 Verificar que a estrutura DOM corresponde aos seletores JS esperados (`.cpu-checkbox`, `.cpu-weight`, `.cpu-pin`, `.device-util-val`, `.device-util-bar`, `.device-vram-text`, `.device-vram-bar`)
- [ ] 07.5 Escrever teste de contrato HTML para verificar presença da linha CPU

## Detalhes de Implementação

**Arquivo principal:** `llama_manager.py` — função `index()` (linhas 425-679)

A geração de linhas GPU está nas linhas 443-549. A linha CPU é injetada no mesmo padrão, com as adaptações específicas (sem radio button, max weight 70, data-device="cpu").

O `<tbody id="gpu-table-body">` está na linha 886. A variável `{gpu_rows}` é preenchida com o HTML das linhas GPU. A linha CPU é injetada como `{cpu_rows}`.

### Arquivos Relevantes
- `llama_manager.py` — função `index()` (linha 425-679), geração de linhas GPU (linha 443-549), tbody (linha 886-888)
- `d:\dsv-git\automanager-llama.cpp\llama_manager.py` — linha 425 (index), 682-1062 (_build_html)

### Arquivos Dependentes
- `gpu_manager.py` — `detect_cpu_info()` é chamado para obter dados (tarefa 02)
- `static/js/gpu.js` — JS precisará encontrar elementos por classe (tarefa 08)
- `static/js/metrics.js` — JS precisará atualizar elementos da CPU (tarefa 09)
- `tests/unit/test_html_contract.py` — padrão de teste existente para HTML

### ADRs Relacionados
- [ADR-001](adrs/adr-001.md) — CPU como Dispositivo Unificado na Tabela de Recursos

## Entregáveis
- Linha da CPU injetada no HTML do dashboard
- Estrutura DOM consistente com seletores JS esperados
- Checkbox checked por padrão, input weight max=70
- Testes unitários com cobertura >= 80% **(OBRIGATÓRIO)**
- Testes de integração: HTML contém linha CPU com todos os elementos **(OBRIGATÓRIO)**

## Testes
- Testes unitários:
  - [ ] HTML gerado contém `<tr id="cpu-row">` com `data-device="cpu"`
  - [ ] HTML contém `.cpu-checkbox` checkbox
  - [ ] HTML contém `.cpu-weight` input com `min="0"` e `max="70"`
  - [ ] HTML contém `.cpu-pin` checkbox
  - [ ] HTML contém `.device-util-val` e `.device-util-bar`
  - [ ] HTML contém `.device-vram-text` e `.device-vram-bar`
  - [ ] HTML NÃO contém radio button na linha da CPU
  - [ ] `.cpu-checkbox` está checked por padrão
- Testes de integração:
  - [ ] GET `/` retorna HTML com linha da CPU visível
  - [ ] Nome do processador é exibido corretamente na linha CPU
- Meta de cobertura: >= 80%
- Todos os testes devem passar

## Critérios de Sucesso
- Todos os testes passando
- Cobertura de testes >= 80%
- Linha da CPU visualmente consistente com linhas GPU
- Todos os elementos DOM esperados presentes no HTML
- Compatível com seletores JS das tarefas 08 e 09
