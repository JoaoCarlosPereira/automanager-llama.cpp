---
status: pending
title: Configuração de ferramentas de teste
type: chore
complexity: low
dependencies: []
---

# Tarefa 1: Configuração de ferramentas de teste

## Visão Geral

Configurar o ambiente de teste JavaScript (Jest + Playwright) e Python (playwright) para suportar a nova suíte de testes da dashboard. Esta tarefa é a fundação sobre a qual todas as demais tarefas dependem — sem ela, não é possível escrever ou executar os testes unitários Jest ou E2E Playwright.

<critical>
- SEMPRE LEIA o PRD e o TechSpec antes de começar
- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui
- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como
- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas
- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis
</critical>

<requirements>
- O arquivo `package.json` DEVE existir na raiz do projeto com `"type": "module"` e dependências Jest + Playwright
- O arquivo `jest.config.js` DEVE existir na raiz configurando JSDOM, ES modules, cobertura 90%
- O arquivo `playwright.config.ts` DEVE existir configurando Chromium, webServer mock, trace e screenshot
- O arquivo `requirements.txt` DEVE incluir `playwright>=1.40.0`
- O arquivo `.gitignore` DEVE incluir `node_modules/`, `coverage/`, `.playwright/`
- Os scripts npm `test`, `test:e2e`, `test:e2e:ui`, `test:all` DEVEM existir no package.json
</requirements>

## Subtarefas

- [ ] 1.1 Criar `package.json` com Jest, jest-environment-jsdom, @playwright/test e scripts npm
- [ ] 1.2 Criar `jest.config.js` com JSDOM, moduleFileExtensions, testMatch, coverageThreshold 90%
- [ ] 1.3 Criar `playwright.config.ts` com Chromium, baseURL, webServer, trace on-first-retry
- [ ] 1.4 Adicionar `playwright>=1.40.0` ao `requirements.txt`
- [ ] 1.5 Atualizar `.gitignore` com `node_modules/`, `coverage/`, `*.d.ts`, `.playwright/`
- [ ] 1.6 Instalar dependências npm (`npm install`) e Playwright browsers (`npx playwright install chromium`)
- [ ] 1.7 Validar que `npm test` roda sem erros (mesmo sem testes ainda)

## Detalhes de Implementação

A estrutura de arquivos será:

```
automanager-llama.cpp/
├── package.json              # Dependências e scripts
├── jest.config.js            # Configuração Jest
├── playwright.config.ts      # Configuração Playwright
├── requirements.txt          # Modificado: adicionar playwright
├── .gitignore                # Modificado: adicionar node_modules/, coverage/
```

Referencie a seção "Configuração Jest e Playwright" do TechSpec para detalhes da configuração.

### Arquivos Relevantes

- `requirements.txt` — adicionar dependência Python do Playwright
- `.gitignore` — adicionar entradas para JS tooling
- `package.json` — novo, raiz do projeto
- `jest.config.js` — novo, raiz do projeto
- `playwright.config.ts` — novo, raiz do projeto

### ADRs Relacionados

- [ADR-001: Extração de JavaScript para testes unitários com Jest + Playwright E2E](../adrs/adr-001.md) — Base para toda a estratégia de testes
- [ADR-004: Configuração Jest e Playwright para testes de UI](../adrs/adr-004.md) — Especificação exata da configuração

## Entregáveis

- `package.json` com Jest, jest-environment-jsdom, @playwright/test e scripts npm (test, test:e2e, test:e2e:ui, test:all)
- `jest.config.js` com JSDOM, módulos ES, coverageThreshold 90%
- `playwright.config.ts` com Chromium, webServer, trace on-first-retry
- `requirements.txt` atualizado com `playwright>=1.40.0`
- `.gitignore` atualizado com `node_modules/`, `coverage/`, `.playwright/`
- Dependências instaladas (`node_modules/` existe, browsers instalados)
- `npm test` roda sem erro (sem testes, apenas validação de config)
- `npm run test:e2e` roda sem erro de config (sem testes, apenas validação de config)

## Testes

- Testes de configuração:
  - [ ] `npm test -- --listTests` lista testes (mesmo que vazio, sem erro de parsing)
  - [ ] `npx playwright test --list` lista testes (mesmo que vazio, sem erro de parsing)
  - [ ] `jest --version` retorna versão (Jest instalado corretamente)
  - [ ] `playwright --version` retorna versão (Playwright instalado corretamente)
  - [ ] `pip show playwright` exibe pacote (Playwright Python instalado)

## Critérios de Sucesso

- Todos os scripts npm configurados e executáveis (`npm test`, `npm run test:e2e`, `npm run test:all`)
- Jest e Playwright instalados e configurados sem erros
- `node_modules/` presente e adicionado ao `.gitignore`
- Playwright browsers Chromium instalados (`npx playwright install chromium` passou)
- `npm test` roda sem erro de configuração
