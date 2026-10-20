// Modo Proxy Inteligente — controles de configuração (task 07) e painel de
// monitoramento (task 08). Consome /proxy/config, /models/proxy,
// /proxy/status e /proxy/sessions.
import { apiFetch, showToast, showConfirm } from './auth.js?v=4.2.50';

// Quota helpers vivem em models.js; import dinâmico evita ciclo proxy ↔ models.
const QUOTA_PROVIDERS = new Set(['codex', 'claude', 'antigravity']);
let quotaHelpersPromise = null;

function loadQuotaHelpers() {
    if (!quotaHelpersPromise) {
        quotaHelpersPromise = import('./models.js?v=4.2.50').catch((err) => {
            console.debug('quota helpers:', err);
            quotaHelpersPromise = null;
            return null;
        });
    }
    return quotaHelpersPromise;
}

// updateStatus roda a cada 1s; o painel consulta o proxy a cada 3 ticks.
const POLL_EVERY_TICKS = 3;
let tick = 0;
let updating = false;
let priorityDragActive = false;
let prioritySavePending = false;

function esc(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function jsonHeaders() {
    return { 'Content-Type': 'application/json' };
}

export function proxyModeEnabled() {
    const toggle = document.getElementById('proxy-enabled-toggle');
    return !!(toggle && toggle.checked);
}

function syncPrimaryHint(smartProxy) {
    const hint = document.getElementById('proxy-primary-hint');
    if (!hint) return;
    const hasPrimary = smartProxy && (
        smartProxy.primary_model_path || smartProxy.primary_backend_id
    );
    const needsPrimary = proxyModeEnabled() && !hasPrimary;
    hint.classList.toggle('hidden', !needsPrimary);
}

function updateBackendConfigCache(path, backendId, values) {
    if (backendId) {
        window.platformConfigs = window.platformConfigs || {};
        window.platformConfigs[backendId] = {
            ...(window.platformConfigs[backendId] || {}),
            ...values,
        };
        return;
    }
    if (path && window.modelConfigs[path]) {
        window.modelConfigs[path] = {
            ...window.modelConfigs[path],
            ...values,
        };
    }
}

// ---------------------------------------------------------------------------
// Controles de configuração (task 07)
// ---------------------------------------------------------------------------

export async function proxyToggleEnabled(checkbox) {
    try {
        const res = await apiFetch('/proxy/config', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify({ enabled: checkbox.checked }),
        });
        if (!res.ok) throw new Error('config');
        const data = await res.json();
        showToast(
            checkbox.checked
                ? 'Modo Proxy Inteligente ativado'
                : 'Modo Proxy Inteligente desativado',
            'success',
        );
        syncPrimaryHint(data.smart_proxy);
        tick = 0;
        updateProxyPanel(true);
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Falha ao salvar configuração do proxy', 'error');
    }
}

export async function setProxyPrimary(checkbox, path, backendId = null) {
    const payload = backendId
        ? { primary_backend_id: checkbox.checked ? backendId : null }
        : { primary_model_path: checkbox.checked ? path : null };
    try {
        const res = await apiFetch('/proxy/config', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('primary');
        const data = await res.json();
        // Exclusividade: apenas um "Principal" por vez (PRD F2)
        document.querySelectorAll('.proxy-primary-checkbox').forEach((cb) => {
            if (cb !== checkbox) cb.checked = false;
        });
        showToast(
            checkbox.checked
                ? 'Backend definido como principal do proxy'
                : 'Backend principal removido',
            'success',
        );
        syncPrimaryHint(data.smart_proxy);
        updateProxyPanel(true);
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Falha ao definir modelo principal', 'error');
    }
}

export async function setProxyEligible(checkbox, path, backendId = null, cardId = null) {
    const payload = backendId
        ? { backend_id: backendId, proxy_eligible: checkbox.checked }
        : { model_path: path, card_id: cardId, proxy_eligible: checkbox.checked };
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('eligible');
        updateBackendConfigCache(path, backendId, { proxy_eligible: checkbox.checked });
        showToast('Participação no proxy atualizada', 'success');
        updateProxyPanel(true);
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Falha ao atualizar participação no proxy', 'error');
    }
}

export async function setProxyMaxParallel(input, path, backendId = null, cardId = null) {
    const value = parseInt(input.value, 10);
    if (!Number.isFinite(value) || value < 1) {
        input.value = 1;
        return;
    }
    try {
        const payload = backendId
            ? { backend_id: backendId, max_parallel_requests: value }
            : { model_path: path, card_id: cardId, max_parallel_requests: value };
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('parallel');
        updateBackendConfigCache(path, backendId, { max_parallel_requests: value });
        showToast('Capacidade inicial de paralelismo salva', 'success');
    } catch (err) {
        showToast('Falha ao salvar limite de paralelismo', 'error');
    }
}

// ---------------------------------------------------------------------------
// Painel de monitoramento (task 08)
// ---------------------------------------------------------------------------

const STATE_STYLES = {
    online: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10',
    busy: 'text-amber-400 border-amber-500/40 bg-amber-500/10',
    cooldown: 'text-orange-400 border-orange-500/40 bg-orange-500/10',
    disabled: 'text-rose-400 border-rose-500/40 bg-rose-500/10',
    not_eligible: 'text-slate-500 border-slate-700/50 bg-slate-800/40',
    offline: 'text-slate-500 border-slate-700/50 bg-slate-800/40',
    rate_limited: 'text-red-400 border-red-500/40 bg-red-500/10',
};

// Rótulos exibidos em PT-BR; os valores da API permanecem em inglês (contrato)
const STATE_LABELS = {
    online: 'ONLINE',
    busy: 'OCUPADO',
    cooldown: 'EM RECUPERAÇÃO',
    disabled: 'DESATIVADO',
    not_eligible: 'FORA DO PROXY',
    offline: 'OFFLINE',
    rate_limited: 'ESGOTADO',
};

function renderModeBadge(enabled) {
    const badge = document.getElementById('proxy-mode-badge');
    if (!badge) return;
    if (enabled) {
        badge.innerText = 'ATIVO';
        badge.className = 'px-3 py-1 rounded-full text-ui-label font-black tracking-widest uppercase glass border-violet-500/40 text-violet-300 bg-violet-500/10';
    } else {
        badge.innerText = 'INATIVO';
        badge.className = 'px-3 py-1 rounded-full text-ui-label font-black tracking-widest uppercase glass border-slate-700/50 text-slate-500';
    }
}

function renderOff() {
    renderModeBadge(false);
    const body = document.getElementById('proxy-panel-body');
    if (body) body.classList.add('hidden');
}

function providerIcon(provider, isPlatform) {
    const key = String(provider || '').toLowerCase();
    if (!isPlatform) return 'fa-server';
    if (key === 'claude') return 'fa-brain';
    if (key === 'codex') return 'fa-code';
    if (key === 'antigravity') return 'fa-star';
    if (key === 'cursor') return 'fa-i-cursor';
    return 'fa-cloud';
}

function formatLatencyShort(ms) {
    const latency = Number(ms);
    if (!Number.isFinite(latency) || latency <= 0) return '—';
    if (latency >= 1000) return `${(latency / 1000).toFixed(2)} s`;
    return `${Math.round(latency)} ms`;
}

function localMetricTone(pct) {
    if (pct == null || Number.isNaN(Number(pct))) return 'slate';
    const value = Number(pct);
    if (value < 10) return 'rose';
    if (value < 30) return 'amber';
    return 'emerald';
}

function localMetricColor(kind, pct) {
    const tone = localMetricTone(pct);
    if (kind === 'capacity') {
        if (tone === 'rose') return '#ef4444';
        if (tone === 'amber') return '#f59e0b';
        return '#34d399';
    }
    // Startup / prontidão — ciano/teal no acento local
    if (tone === 'rose') return '#f43f5e';
    if (tone === 'amber') return '#22d3ee';
    return '#2dd4bf';
}

/** Capacidade livre e prontidão de startup — substitutos locais de Semanal/Sessão. */
function localGaugesHtml(backend) {
    const maxParallel = Math.max(0, Number(backend.max_parallel) || 0);
    const inFlight = Math.max(0, Number(backend.in_flight) || 0);
    const free = Math.max(0, maxParallel - inFlight);
    const capacityPct = maxParallel > 0
        ? Math.max(0, Math.min(100, Math.round((free / maxParallel) * 100)))
        : null;
    const capacityColor = capacityPct == null ? '#475569' : localMetricColor('capacity', capacityPct);
    const capacityCenter = capacityPct == null ? '—' : `${capacityPct}%`;
    const capacityHint = maxParallel > 0
        ? `${free}/${maxParallel} livres`
        : 'sem slots';

    const latency = Number(backend.startup_latency_ms);
    const measured = Number.isFinite(latency) && latency > 0;
    let startupPct = null;
    if (measured) {
        // Mais rápido → anel mais cheio (escala ~1.5s).
        startupPct = Math.max(5, Math.min(100, Math.round(100 * Math.exp(-latency / 1500))));
    } else if (backend.state === 'offline' || backend.state === 'disabled') {
        startupPct = 0;
    }
    const startupColor = startupPct == null ? '#475569' : localMetricColor('startup', startupPct);
    const startupCenter = measured ? formatLatencyShort(latency) : '—';
    let startupHint = 'medição pendente';
    if (measured) {
        const rank = backend.speed_rank;
        if (rank === 0) startupHint = 'principal';
        else if (Number.isFinite(Number(rank)) && Number(rank) > 0) startupHint = `${rank}º em velocidade`;
        else startupHint = 'startup medido';
    } else if (backend.state === 'offline' || backend.state === 'disabled') {
        startupHint = 'indisponível';
    }

    return `
        <div class="quota-gauges-host">
            <div class="quota-gauges" aria-label="Capacidade e startup local">
                <div class="quota-gauge" data-kind="capacity" title="Slots livres: ${esc(capacityHint)}">
                    <div class="quota-gauge-ring" style="--quota-pct:${capacityPct ?? 0};--quota-color:${capacityColor}">
                        <div class="quota-gauge-center">
                            <span class="quota-gauge-pct">${esc(capacityCenter)}</span>
                            <span class="quota-gauge-sub">livre</span>
                        </div>
                    </div>
                    <p class="quota-gauge-name">Capacidade</p>
                    <p class="quota-gauge-reset"><i class="fas fa-layer-group" aria-hidden="true"></i> <span>${esc(capacityHint)}</span></p>
                </div>
                <div class="quota-gauge" data-kind="startup" title="Startup: ${esc(startupCenter)}">
                    <div class="quota-gauge-ring" style="--quota-pct:${startupPct ?? 0};--quota-color:${startupColor}">
                        <div class="quota-gauge-center">
                            <span class="quota-gauge-pct quota-gauge-pct-sm">${esc(startupCenter)}</span>
                            <span class="quota-gauge-sub">startup</span>
                        </div>
                    </div>
                    <p class="quota-gauge-name">Prontidão</p>
                    <p class="quota-gauge-reset"><i class="fas fa-stopwatch" aria-hidden="true"></i> <span>${esc(startupHint)}</span></p>
                </div>
            </div>
        </div>`;
}

function platformQuotaGaugesSkeleton() {
    return `
        <div class="quota-accounts-host"></div>
        <div class="quota-gauges-host">
            <div class="quota-gauges" aria-label="Cota semanal e de sessao">
                <div class="quota-gauge" data-kind="weekly">
                    <div class="quota-gauge-ring" style="--quota-pct:0;--quota-color:#475569">
                        <div class="quota-gauge-center">
                            <span class="quota-gauge-pct">—</span>
                            <span class="quota-gauge-sub">restante</span>
                        </div>
                    </div>
                    <p class="quota-gauge-name">Semanal</p>
                    <p class="quota-gauge-reset"><i class="fas fa-clock" aria-hidden="true"></i> <span class="quota-gauge-reset-text">—</span></p>
                </div>
                <div class="quota-gauge" data-kind="session">
                    <div class="quota-gauge-ring" style="--quota-pct:0;--quota-color:#475569">
                        <div class="quota-gauge-center">
                            <span class="quota-gauge-pct">—</span>
                            <span class="quota-gauge-sub">restante</span>
                        </div>
                    </div>
                    <p class="quota-gauge-name">Sessão</p>
                    <p class="quota-gauge-reset"><i class="fas fa-clock" aria-hidden="true"></i> <span class="quota-gauge-reset-text">—</span></p>
                </div>
            </div>
        </div>`;
}

function backendCard(backend, index) {
    // Rate-limited platform backends: show "ESGOTADO" regardless of state
    let state = backend.state;
    if (backend.backend_type === 'platform' && backend.is_rate_limited) {
        state = 'rate_limited';
    }
    const style = STATE_STYLES[state] || STATE_STYLES.offline;
    const stateColor = style.split(' ')[0];
    const isPlatform = backend.backend_type === 'platform';
    const provider = String(backend.provider || '').toLowerCase();
    const kind = isPlatform ? `Plataforma · ${esc(backend.provider || 'cloud')}` : 'Local';
    const detail = isPlatform ? kind : `Local · ${esc(backend.gpu)}`;
    const role = backend.role === 'primary' ? 'Principal' : 'Secundário';
    const latency = Number(backend.startup_latency_ms);
    const measured = Number.isFinite(latency) && latency > 0;
    const latencyLabel = formatLatencyShort(latency);
    const priorityRank = index + 1;
    const priorityIcon = priorityRank === 1 ? 'fa-crown' : 'fa-signal';
    const dataBackendIds = backend.grouped_ids ? ` data-backend-ids="${esc(backend.grouped_ids.join(','))}"` : '';
    const dataModelPath = backend.model_path ? ` data-model-path="${esc(backend.model_path)}"` : '';
    const dataCardId = backend.card_id ? ` data-card-id="${esc(backend.card_id)}"` : '';
    const navigationBackendId = backend.config_backend_id || backend.backend_id || '';
    const showQuota = isPlatform && QUOTA_PROVIDERS.has(provider);
    const showLocalGauges = !isPlatform;
    const accent = showQuota ? provider : (isPlatform ? 'platform' : 'local');
    const activeLabel = `${esc(backend.in_flight)}/${esc(backend.effective_parallel ?? backend.max_parallel)}`;
    const workerCount = Array.isArray(backend.grouped_ids) ? backend.grouped_ids.length : 1;
    const gaugesBlock = showQuota
        ? platformQuotaGaugesSkeleton()
        : (showLocalGauges ? localGaugesHtml(backend) : '');

    // Rodapé: plataformas sem cota e locais priorizam infra (gauges já cobrem carga/startup).
    let footerHtml;
    if (showLocalGauges) {
        footerHtml = `
        <div class="proxy-card-footer">
            <div class="proxy-card-stat">
                <i class="fas fa-server" aria-hidden="true"></i>
                <div>
                    <p class="proxy-card-stat-value">${esc(workerCount)} worker${workerCount === 1 ? '' : 's'}</p>
                    <p class="proxy-card-stat-label">Instâncias</p>
                </div>
            </div>
            <div class="proxy-card-stat">
                <i class="fas fa-memory" aria-hidden="true"></i>
                <div>
                    <p class="proxy-card-stat-value">${esc(backend.ctx_per_slot || '—')}</p>
                    <p class="proxy-card-stat-label">Ctx/slot</p>
                </div>
            </div>
            <div class="proxy-card-stat proxy-card-stat-wide">
                <i class="fas fa-microchip" aria-hidden="true"></i>
                <div>
                    <p class="proxy-card-stat-value truncate">${esc(backend.gpu || 'CPU')} · porta ${esc(backend.port)}</p>
                    <p class="proxy-card-stat-label">Infra</p>
                </div>
            </div>
        </div>`;
    } else {
        footerHtml = `
        <div class="proxy-card-footer">
            <div class="proxy-card-stat">
                <i class="fas fa-bolt" aria-hidden="true"></i>
                <div>
                    <p class="proxy-card-stat-value">${esc(measured ? latencyLabel : '—')}</p>
                    <p class="proxy-card-stat-label">${measured ? 'Latência média' : 'Startup pendente'}</p>
                </div>
            </div>
            <div class="proxy-card-stat">
                <i class="fas fa-gauge-high" aria-hidden="true"></i>
                <div>
                    <p class="proxy-card-stat-value">${activeLabel}</p>
                    <p class="proxy-card-stat-label">Req. ativas</p>
                </div>
            </div>
            <div class="proxy-card-stat proxy-card-stat-wide">
                <i class="fas fa-microchip" aria-hidden="true"></i>
                <div>
                    <p class="proxy-card-stat-value truncate">porta ${esc(backend.port)} · ctx ${esc(backend.ctx_per_slot)}</p>
                    <p class="proxy-card-stat-label">Infra</p>
                </div>
            </div>
        </div>`;
    }

    const cardHtml = `
    <div draggable="true" class="proxy-backend-card proxy-card-v2 accent-${esc(accent)} cursor-pointer active:cursor-grabbing select-none" data-proxy-backend="${esc(backend.port)}" data-backend-id="${esc(backend.backend_id || '')}" data-navigation-backend-id="${esc(navigationBackendId)}" data-backend-type="${esc(backend.backend_type || 'local')}" data-provider="${esc(backend.provider || '')}"${dataBackendIds}${dataModelPath}${dataCardId} title="Clique para abrir a aba · Arraste para alterar a prioridade">
        <div class="proxy-card-head">
            <div class="proxy-card-title-row">
                <span class="proxy-card-icon" aria-hidden="true"><i class="fas ${providerIcon(provider, isPlatform)}"></i></span>
                <div class="proxy-card-title-text min-w-0">
                    <div class="proxy-card-name-row">
                        <span class="proxy-card-name truncate">${esc(backend.model)}</span>
                        <span class="proxy-card-status ${stateColor}"><span class="proxy-card-status-dot" aria-hidden="true"></span>${esc(STATE_LABELS[state] || backend.state)}</span>
                    </div>
                    <p class="proxy-card-subtitle truncate">${role} · ${detail}</p>
                </div>
                <i class="fas fa-grip-vertical proxy-card-grip" aria-hidden="true"></i>
            </div>
        </div>
        <div class="proxy-priority-bar">
            <i class="fas ${priorityIcon}" aria-hidden="true"></i>
            <span class="proxy-priority-label">Prioridade de roteamento: ${priorityRank}º</span>
            <span class="proxy-priority-rank" hidden>${priorityRank}º</span>
        </div>
        ${gaugesBlock}
        ${footerHtml}
    </div>`;
    if (!showQuota) return cardHtml;
    // Shell síncrono; percentuais reais entram via applyAllQuotaShells após o render.
    return `
        <div class="platform-quota-shell proxy-quota-shell quota-unknown"
             data-quota-provider="${esc(provider)}"
             title="Cota semanal e sessao 5h">
            ${cardHtml}
        </div>`;
}

function formatLocalDateTime(iso) {
    // Timestamps das sessões são ISO-8601 UTC; exibe no fuso do navegador
    if (!iso) return '';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) {
        return iso.replace('T', ' ').slice(0, 19);
    }
    return date.toLocaleString('pt-BR');
}

function sessionRow(session) {
    const label = session.detected_tag || session.affinity_key;
    const lastUsed = formatLocalDateTime(session.last_used_at);
    const tokens = session.tokens_processed ? ` · ${session.tokens_processed} tokens` : '';
    return `
    <div class="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-slate-900/50 border border-slate-800/60" data-proxy-session="${esc(session.affinity_key)}">
        <div class="min-w-0">
            <p class="text-ui-body-sm font-bold text-slate-200 truncate">${esc(label)} <span class="text-slate-500 font-normal">→ ${esc(session.gpu || '?')} / ${esc(session.internal_model)}</span></p>
            <p class="text-ui-label text-slate-500 font-mono truncate">${esc(session.affinity_key)} · ${esc(session.request_count)} requisição(ões)${esc(tokens)} · ${esc(lastUsed)}</p>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            <button type="button" class="proxy-session-reassign w-7 h-7 flex items-center justify-center rounded bg-slate-800 text-slate-500 hover:text-violet-400 transition-all" title="Reatribuir sessão a outro backend" data-key="${esc(session.affinity_key)}"><i class="fas fa-random text-ui-label"></i></button>
            <button type="button" class="proxy-session-delete w-7 h-7 flex items-center justify-center rounded bg-slate-800 text-slate-500 hover:text-red-400 transition-all" title="Remover sessão sticky" data-key="${esc(session.affinity_key)}"><i class="fas fa-trash-alt text-ui-label"></i></button>
        </div>
    </div>`;
}


function bindDragEvents() {
    const list = document.getElementById('proxy-backends-list');
    if (!list) return;
    const cards = list.querySelectorAll('.proxy-backend-card');
    let draggedItem = null;
    let orderChanged = false;

    const moveUnit = (card) => card.closest('.platform-quota-shell') || card;

    const clearDropStyles = () => {
        list.querySelectorAll('.proxy-backend-card, .platform-quota-shell').forEach((el) => {
            el.classList.remove('opacity-50', 'ring-2', 'ring-violet-500', 'scale-[1.01]');
        });
    };

    const refreshVisibleRanks = () => {
        list.querySelectorAll('.proxy-backend-card').forEach((item, index) => {
            const rank = `${index + 1}º`;
            const badge = item.querySelector('.proxy-priority-rank');
            if (badge) badge.textContent = rank;
            const label = item.querySelector('.proxy-priority-label');
            if (label) label.textContent = `Prioridade de roteamento: ${rank}`;
            const icon = item.querySelector('.proxy-priority-bar i');
            if (icon) {
                icon.className = `fas ${index === 0 ? 'fa-crown' : 'fa-signal'}`;
            }
        });
    };

    cards.forEach(card => {
        let pointerStart = null;
        let pointerMoved = false;
        // Arrasta o shell de cota inteiro (quando existir) para não separar o anel do card.
        card.setAttribute('draggable', 'true');
        const unit = moveUnit(card);
        if (unit !== card) card.removeAttribute('draggable');
        unit.setAttribute('draggable', 'true');

        card.addEventListener('pointerdown', function(e) {
            if (e.button !== 0) return;
            pointerStart = { x: e.clientX, y: e.clientY };
            pointerMoved = false;
        });

        card.addEventListener('pointermove', function(e) {
            if (!pointerStart) return;
            const distance = Math.hypot(
                e.clientX - pointerStart.x,
                e.clientY - pointerStart.y,
            );
            if (distance > 5) {
                pointerMoved = true;
                this.dataset.suppressOpen = 'true';
            }
        });

        card.addEventListener('pointerup', function() {
            pointerStart = null;
            if (pointerMoved) {
                setTimeout(() => { this.dataset.suppressOpen = 'false'; }, 0);
            }
        });

        card.addEventListener('pointercancel', function() {
            pointerStart = null;
            pointerMoved = true;
            this.dataset.suppressOpen = 'true';
            setTimeout(() => { this.dataset.suppressOpen = 'false'; }, 0);
        });

        card.addEventListener('click', function() {
            if (pointerMoved || priorityDragActive || this.dataset.suppressOpen === 'true') return;
            const backendId = this.dataset.navigationBackendId || this.dataset.backendId || '';
            if (this.dataset.backendType === 'platform') {
                if (backendId) window.selectPlatform?.(backendId);
                return;
            }
            const modelPath = this.dataset.modelPath || '';
            const cardId = this.dataset.cardId || '';
            if (modelPath) window.selectModel?.(modelPath, cardId);
        });

        unit.addEventListener('dragstart', function(e) {
            draggedItem = this;
            orderChanged = false;
            priorityDragActive = true;
            card.dataset.suppressOpen = 'true';
            setTimeout(() => this.classList.add('opacity-50'), 0);
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', card.dataset.backendId || 'backend');
        });

        unit.addEventListener('dragend', async function() {
            const shouldSave = orderChanged;
            draggedItem = null;
            priorityDragActive = false;
            clearDropStyles();
            if (shouldSave) await savePriorityOrder();
            setTimeout(() => {
                card.dataset.suppressOpen = 'false';
                pointerMoved = false;
            }, 0);
        });

        unit.addEventListener('dragover', function(e) {
            e.preventDefault();
            if (this !== draggedItem) {
                this.classList.add('ring-2', 'ring-violet-500', 'scale-[1.01]');
                const rect = this.getBoundingClientRect();
                const horizontal = rect.width > rect.height;
                const after = horizontal
                    ? e.clientX > rect.left + rect.width / 2
                    : e.clientY > rect.top + rect.height / 2;
                list.insertBefore(draggedItem, after ? this.nextSibling : this);
                orderChanged = true;
                refreshVisibleRanks();
            }
        });

        unit.addEventListener('dragleave', function() {
            this.classList.remove('ring-2', 'ring-violet-500', 'scale-[1.01]');
        });

        unit.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('ring-2', 'ring-violet-500', 'scale-[1.01]');
        });
    });
}

async function savePriorityOrder() {
    const list = document.getElementById('proxy-backends-list');
    if (!list) return;
    const cards = list.querySelectorAll('.proxy-backend-card');
    const custom_priority = Array.from(cards).flatMap(c => {
        if (c.dataset.backendIds) {
            return c.dataset.backendIds.split(',');
        }
        return [c.dataset.backendId || c.dataset.proxyBackend];
    }).filter(Boolean);
    
    prioritySavePending = true;
    let saved = false;
    try {
        const res = await apiFetch('/proxy/config', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify({ custom_priority })
        });
        if (res.ok) {
            showToast('Prioridade atualizada', 'success');
            tick = 0;
            saved = true;
        } else {
            const detail = await res.json().catch(() => ({}));
            throw new Error(detail.detail || `HTTP ${res.status}`);
        }
    } catch (e) {
        showToast('Erro ao atualizar prioridade', 'error');
    } finally {
        prioritySavePending = false;
    }
    if (!saved) updateProxyPanel(true);
}

function bindSessionActions() {
    document.querySelectorAll('.proxy-session-delete').forEach((btn) => {
        btn.onclick = () => proxyDeleteSession(btn.dataset.key);
    });
    document.querySelectorAll('.proxy-session-reassign').forEach((btn) => {
        btn.onclick = () => proxyReassignSession(btn.dataset.key);
    });
}

function groupBackends(backends) {
    if (!backends) return [];
    const grouped = [];
    const groups = new Map();

    for (const b of backends) {
        if (b.backend_type === 'platform' && b.model) {
            const key = `platform:${b.model}`;
            if (groups.has(key)) {
                const group = groups.get(key);
                group.in_flight += b.in_flight;
                group.max_parallel += b.max_parallel;
                group.effective_parallel += (b.effective_parallel || b.max_parallel);
                group.grouped_ids.push(b.backend_id);
                // Status logic for groups
                if (!b.is_rate_limited) {
                    group.all_rate_limited = false;
                }
                
                // Keep track of lowest startup_latency_ms if present
                if (b.startup_latency_ms != null) {
                    if (group.startup_latency_ms == null) {
                        group.startup_latency_ms = b.startup_latency_ms;
                    } else {
                        group.startup_latency_ms = Math.min(group.startup_latency_ms, b.startup_latency_ms);
                    }
                }
                
                if (b.role === 'primary') {
                    group.role = 'primary';
                }
            } else {
                const newGroup = { ...b, grouped_ids: [b.backend_id], all_rate_limited: !!b.is_rate_limited, port: "múltiplas" };
                groups.set(key, newGroup);
                grouped.push(newGroup);
            }
        } else {
            grouped.push(b);
        }
    }

    // Resolve state for grouped platforms
    for (const b of grouped) {
        if (b.grouped_ids && b.grouped_ids.length > 1) {
            if (b.all_rate_limited) {
                b.state = 'rate_limited';
                b.is_rate_limited = true;
            } else {
                b.is_rate_limited = false;
                if (b.in_flight >= b.effective_parallel) {
                    b.state = 'busy';
                } else {
                    b.state = 'online';
                }
            }
        }
    }

    return grouped;
}

function render(status, sessions) {
    renderModeBadge(status.enabled);
    syncPrimaryHint(status);
    const body = document.getElementById('proxy-panel-body');
    if (body) body.classList.toggle('hidden', !status.enabled);

    const backendsEl = document.getElementById('proxy-backends-list');
    if (backendsEl) {
        const grouped = groupBackends(status.backends);
        backendsEl.innerHTML = grouped.map((backend, index) => backendCard(backend, index)).join('')
            || '<p class="text-ui-label text-slate-600">Nenhuma instância online.</p>';
        const hasPlatformQuota = grouped.some(
            (b) => b.backend_type === 'platform'
                && QUOTA_PROVIDERS.has(String(b.provider || '').toLowerCase()),
        );
        if (hasPlatformQuota) {
            loadQuotaHelpers().then((mod) => {
                if (!mod) return;
                mod.applyAllQuotaShells?.();
                mod.refreshPlatformQuotas?.({ force: false });
            });
        }
    }
    const countEl = document.getElementById('proxy-sessions-count');
    if (countEl) countEl.innerText = `(${sessions.length})`;
    const ttlHint = document.getElementById('proxy-sessions-ttl-hint');
    if (ttlHint) {
        const ttl = Number(status.ttl_minutes) || 0;
        ttlHint.innerText = ttl
            ? `· auto-clean após ${ttl} min ociosas`
            : '';
    }
    const clearBtn = document.getElementById('proxy-sessions-clear-btn');
    if (clearBtn) clearBtn.disabled = sessions.length === 0;
    const sessionsEl = document.getElementById('proxy-sessions-list');
    if (sessionsEl) {
        sessionsEl.innerHTML = sessions.map(sessionRow).join('')
            || '<p class="text-ui-label text-slate-600">Nenhuma sessão ativa.</p>';
        bindSessionActions();
    }
    bindDragEvents();
}


export async function proxyDeleteSession(affinityKey) {
    const ok = await showConfirm(
        'Remover sessão sticky? A próxima requisição desta conversa será roteada novamente.',
    );
    if (!ok) return;
    const res = await apiFetch(`/proxy/sessions/${encodeURIComponent(affinityKey)}`, {
        method: 'DELETE',
    });
    if (res.ok) {
        showToast('Sessão removida', 'success');
        updateProxyPanel(true);
    } else {
        showToast('Falha ao remover sessão', 'error');
    }
}

export async function proxyClearAllSessions() {
    const ok = await showConfirm(
        'Limpar todas as sessões sticky? As próximas requisições serão roteadas do zero.',
    );
    if (!ok) return;
    const res = await apiFetch('/proxy/sessions', { method: 'DELETE' });
    if (res.ok) {
        const data = await res.json().catch(() => ({}));
        const n = typeof data.removed === 'number' ? data.removed : 0;
        showToast(
            n ? `${n} sessão(ões) removida(s)` : 'Nenhuma sessão para limpar',
            'success',
        );
        updateProxyPanel(true);
    } else {
        showToast('Falha ao limpar sessões', 'error');
    }
}

export async function proxyReassignSession(affinityKey) {
    const ok = await showConfirm(
        'Reatribuir sessão para o melhor backend disponível?',
        { danger: false, confirmLabel: 'Reatribuir' },
    );
    if (!ok) return;
    const res = await apiFetch(
        `/proxy/sessions/${encodeURIComponent(affinityKey)}/reassign`,
        { method: 'POST' },
    );
    if (res.ok) {
        showToast('Sessão reatribuída', 'success');
        updateProxyPanel(true);
    } else {
        showToast('Falha ao reatribuir sessão', 'error');
    }
}

export async function updateProxyPanel(force = false) {
    const panel = document.getElementById('proxy-panel');
    if (!panel) return;
    // Modo OFF: card compacto, sem tráfego de polling (task 08)
    if (!proxyModeEnabled()) {
        renderOff();
        return;
    }
    if (priorityDragActive || prioritySavePending) return;
    if (!force && (tick++ % POLL_EVERY_TICKS) !== 0) return;
    if (updating) return;
    updating = true;
    try {
        const res = await apiFetch('/proxy/status');
        if (!res.ok) return;
        const status = await res.json();
        let sessions = [];
        if (status.enabled) {
            const sessionsRes = await apiFetch('/proxy/sessions');
            if (sessionsRes.ok) sessions = await sessionsRes.json();
        }
        render(status, sessions);
    } catch (err) {
        // Painel é best-effort: não interrompe o polling principal
    } finally {
        updating = false;
    }
}

// ---------------------------------------------------------------------------
// Controles do Context Optimizer (task 14)
// ---------------------------------------------------------------------------

export async function toggleOptimizerEnabled(checkbox) {
    try {
        const res = await apiFetch('/proxy/config', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify({
                context_optimizer: { enabled: checkbox.checked },
            }),
        });
        if (!res.ok) throw new Error('optimizer');
        const data = await res.json();
        const co = data.smart_proxy?.context_optimizer || {};
        const enabledEl = document.getElementById('optimizer-enabled-status');
        if (enabledEl) enabledEl.innerText = checkbox.checked ? 'Ativado' : 'Desativado';
        showToast(
            checkbox.checked ? 'Context Optimizer ativado' : 'Context Optimizer desativado',
            'success',
        );
        updateProxyPanel(true);
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Falha ao alternar Context Optimizer', 'error');
    }
}
