---

status: completed

title: Criar testes de contrato HTML com pytest

type: test

complexity: medium

dependencies: [task_02]

---



# Tarefa 10: Criar testes de contrato HTML com pytest



## Visão Geral



Criar `tests/unit/test_html_contract.py` — testes pytest que validam que a resposta HTML da dashboard (`GET /`) contém todos os elementos esperados. Esta camada de teste protege contra regressões no HTML gerado pelo `_build_html()` (ex: remover um input por engano ao editar a f-string).



<critical>

- SEMPRE LEIA o PRD e o TechSpec antes de começar

- CONSULTE O TECHSPEC para detalhes de implementação — não duplique aqui

- FOQUE NO "O QUÊ" — descreva o que precisa ser feito, não como

- MINIMIZE CÓDIGO — mostre código só para ilustrar estrutura atual ou áreas problemáticas

- TESTES OBRIGATÓRIOS — toda tarefa DEVE incluir testes nos entregáveis

</critical>



<requirements>

- O arquivo `tests/unit/test_html_contract.py` DEVE existir na pasta `tests/unit/`

- O arquivo DEVE conter testes para cada um dos 8+ componentes da dashboard (login overlay, status badge, metrics panel, GPU table, model list, log terminal, pacman canvas, active model card)

- O arquivo DEVE testar que 5 scripts externos são referenciados (auth.js, models.js, metrics.js, gpu.js, index.js)

- O arquivo DEVE testar que IP dinâmico é injetado no HTML

- O arquivo DEVE testar que token de API aparece no HTML

- O arquivo DEVE testar que login overlay respeita estado de autenticação



</requirements>



## Subtarefas



- [x] 10.1 Criar `tests/unit/test_html_contract.py`

- [x] 10.2 Implementar teste para login overlay (id, form, inputs, onsubmit)

- [x] 10.3 Implementar teste para status badge (id, OFFLINE default)

- [x] 10.4 Implementar teste para metrics panel (cpu-val, cpu-bar, ram-val, ram-bar)

- [x] 10.5 Implementar teste para GPU table (gpu-table-body, gpu-row, gpu-weight, gpu-checkbox, gpu-pin)

- [x] 10.6 Implementar teste para model list (model-list-container, model-item-container)

- [x] 10.7 Implementar teste para log terminal (log-box, "Limpar")

- [x] 10.8 Implementar teste para pacman canvas (pacman-background, aria-hidden)

- [x] 10.9 Implementar teste para active model card (active-card, active-model-name, uptime-val)

- [x] 10.10 Implementar teste para scripts externos (5 scripts type="module")

- [x] 10.11 Implementar teste para token de API (api-token, api-link)

- [x] 10.12 Implementar teste para IP dinâmico injetado (display-ip, chat-link href)

- [x] 10.13 Validar que todos os testes passam



## Detalhes de Implementação



Referencie a seção "Testes de Contrato HTML — Exemplo de Estrutura" do TechSpec.



O teste usa httpx TestClient para fazer `GET /` e validar que o HTML retornado contém os IDs esperados.



### Arquivos Relevantes



- `tests/unit/test_html_contract.py` — novo

- `llama_manager.py` — app com `_build_html()` modificado



### Arquivos Dependentes



- `tests/conftest.py` — fixtures existentes (auth_manager, token_manager, mock_http_credentials)



### ADRs Relacionados



- [ADR-001: Extração de JavaScript para testes unitários com Jest + Playwright E2E](../adrs/adr-001.md) — Base para testes de UI



## Entregáveis



- `tests/unit/test_html_contract.py` com 12+ testes de contrato

- Todos os testes passam com pytest

- HTML contém todos os elementos esperados

- 5 scripts externos referenciados no HTML



## Testes



- Testes de contrato HTML:

  - [x] `test_html_contains_login_overlay` — login-overlay, login-form, login-username, login-password, onsubmit

  - [x] `test_html_contains_status_badge` — status-badge, OFFLINE

  - [x] `test_html_contains_metrics_panel` — metrics-panel, cpu-val, cpu-bar, ram-val, ram-bar

  - [x] `test_html_contains_gpu_table` — gpu-table-body, gpu-row, gpu-weight, gpu-checkbox, gpu-pin

  - [x] `test_html_contains_model_list` — model-list-container, model-item-container

  - [x] `test_html_contains_log_terminal` — log-box, "Limpar"

  - [x] `test_html_contains_pacman_canvas` — pacman-background, aria-hidden

  - [x] `test_html_contains_active_model_card` — active-card, active-model-name, uptime-val

  - [x] `test_html_serves_external_js_scripts` — auth.js, models.js, metrics.js, gpu.js, index.js

  - [x] `test_html_contains_api_token` — api-token, api-link

  - [x] `test_html_injects_ip` — display-ip, chat-link href

  - [x] `test_html_contains_default_model_checkbox` — model-default-checkbox

  - [x] `test_html_contains_context_size_select` — context-size, context-size-custom

  - [x] `test_html_contains_parallel_slots_input` — parallel-slots

  - [x] `test_html_contains_batch_size_select` — batch-size

  - [x] `test_html_contains_mmproj_select` — mmproj-path

  - [x] `test_html_contains_split_mode_select` — split-mode

  - [x] `test_html_contains_auto_balance_toggle` — auto-balance-toggle, auto-balance-badge

  - [x] `test_html_contains_auto_balance_cancel_btn` — auto-balance-cancel-btn

  - [x] `test_html_contains_auto_balance_capacity_alert` — auto-balance-capacity-alert

  - [x] `test_html_contains_download_url_input` — download-url

  - [x] `test_html_contains_download_status` — download-status

  - [x] `test_html_contains_password_change_section` — current-password, new-password, password-change-status

  - [x] `test_login_overlay_visible_when_unauthenticated` — overlay flex, dashboard none

  - [x] `test_login_overlay_hidden_when_authenticated` — overlay none, dashboard block

- Meta: todos os testes devem passar — **25 passed** (`pytest tests/unit/test_html_contract.py -v`)



## Critérios de Sucesso



- test_html_contract.py passa todos os 20+ testes

- Nenhum elemento HTML esperado está ausente

- 5 scripts externos referenciados corretamente

- IP dinâmico injetado corretamente

- Token de API exibido corretamente


