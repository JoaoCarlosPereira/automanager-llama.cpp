// Modo Proxy Inteligente — controles de configuração (task 07) e painel de
// monitoramento (task 08). Consome /proxy/config, /models/proxy,
// /proxy/status e /proxy/sessions.
import { apiFetch, showToast, showConfirm } from './auth.js?v=4.2.3';

// updateStatus roda a cada 1s; o painel consulta o proxy a cada 3 ticks.
const POLL_EVERY_TICKS = 3;
let tick = 0;
let updating = false;

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

export async function setProxyEligible(checkbox, path, backendId = null) {
    const payload = backendId
        ? { backend_id: backendId, proxy_eligible: checkbox.checked }
        : { model_path: path, proxy_eligible: checkbox.checked };
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

export async function setProxyMaxParallel(input, path, backendId = null) {
    const value = parseInt(input.value, 10);
    if (!Number.isFinite(value) || value < 1) {
        input.value = 1;
        return;
    }
    try {
        const payload = backendId
            ? { backend_id: backendId, max_parallel_requests: value }
            : { model_path: path, max_parallel_requests: value };
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

function backendCard(backend) {
    // Rate-limited platform backends: show "ESGOTADO" regardless of state
    let state = backend.state;
    if (backend.backend_type === 'platform' && backend.is_rate_limited) {
        state = 'rate_limited';
    }
    const style = STATE_STYLES[state] || STATE_STYLES.offline;
    const isPlatform = backend.backend_type === 'platform';
    const kind = isPlatform ? `Plataforma · ${esc(backend.provider || 'cloud')}` : 'Local';
    const detail = isPlatform ? kind : `Local · ${esc(backend.gpu)}`;
    const role = backend.role === 'primary' ? 'Principal' : 'Secundário';
    const latency = Number(backend.startup_latency_ms);
    const measured = Number.isFinite(latency) && latency > 0;
    const latencyLabel = measured
        ? `${latency >= 1000 ? `${(latency / 1000).toFixed(2)} s` : `${Math.round(latency)} ms`}`
        : 'medição pendente';
    const priorityLabel = backend.role === 'primary'
        ? 'Prioridade principal'
        : (backend.speed_rank ? `Prioridade #${backend.speed_rank}` : 'Sem ranking');
    return `
    <div class="p-3 rounded-xl border ${style.split(' ').slice(1).join(' ')} bg-slate-900/40 flex flex-col gap-1" data-proxy-backend="${esc(backend.port)}" data-backend-id="${esc(backend.backend_id || '')}" data-backend-type="${esc(backend.backend_type || 'local')}">
        <div class="flex items-center justify-between gap-2">
            <span class="text-ui-body-sm font-bold text-slate-200 truncate">${esc(backend.model)}</span>
            <span class="text-ui-label font-black uppercase tracking-widest ${style.split(' ')[0]}">${esc(STATE_LABELS[state] || backend.state)}</span>
        </div>
        <div class="flex items-center justify-between text-ui-label text-slate-500">
            <span>${role} · ${detail}</span>
            <span class="font-mono">porta ${esc(backend.port)}</span>
        </div>
        <div class="flex items-center justify-between text-ui-label text-slate-500">
            <span>${esc(backend.in_flight)}/${esc(backend.effective_parallel ?? backend.max_parallel)} requisição(ões) ativa(s) · base ${esc(backend.max_parallel)}</span>
            <span class="font-mono">ctx/slot ${esc(backend.ctx_per_slot)}</span>
        </div>
        <div class="flex items-center justify-between text-ui-label text-slate-500">
            <span>${esc(priorityLabel)}</span>
            <span class="font-mono">startup ${esc(latencyLabel)}</span>
        </div>
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

function bindSessionActions() {
    document.querySelectorAll('.proxy-session-delete').forEach((btn) => {
        btn.onclick = () => proxyDeleteSession(btn.dataset.key);
    });
    document.querySelectorAll('.proxy-session-reassign').forEach((btn) => {
        btn.onclick = () => proxyReassignSession(btn.dataset.key);
    });
}

function render(status, sessions) {
    renderModeBadge(status.enabled);
    syncPrimaryHint(status);
    const body = document.getElementById('proxy-panel-body');
    if (body) body.classList.toggle('hidden', !status.enabled);

    const backendsEl = document.getElementById('proxy-backends-list');
    if (backendsEl) {
        backendsEl.innerHTML = (status.backends || []).map(backendCard).join('')
            || '<p class="text-ui-label text-slate-600">Nenhuma instância online.</p>';
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

