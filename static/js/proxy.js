// Modo Proxy Inteligente — controles de configuração (task 07) e painel de
// monitoramento (task 08). Consome /proxy/config, /models/proxy,
// /proxy/status e /proxy/sessions.
import { apiFetch, showToast, showConfirm } from './auth.js?v=4.2.1';

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
    const needsPrimary = proxyModeEnabled() && !(smartProxy && smartProxy.primary_model_path);
    hint.classList.toggle('hidden', !needsPrimary);
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

export async function setProxyPrimary(checkbox, path) {
    const value = checkbox.checked ? path : null;
    try {
        const res = await apiFetch('/proxy/config', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify({ primary_model_path: value }),
        });
        if (!res.ok) throw new Error('primary');
        const data = await res.json();
        // Exclusividade: apenas um "Principal" por vez (PRD F2)
        document.querySelectorAll('.proxy-primary-checkbox').forEach((cb) => {
            if (cb !== checkbox) cb.checked = false;
        });
        showToast(
            checkbox.checked
                ? 'Modelo definido como principal do proxy'
                : 'Modelo principal removido',
            'success',
        );
        syncPrimaryHint(data.smart_proxy);
        updateProxyPanel(true);
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Falha ao definir modelo principal', 'error');
    }
}

export async function setProxyEligible(checkbox, path) {
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify({ model_path: path, proxy_eligible: checkbox.checked }),
        });
        if (!res.ok) throw new Error('eligible');
        if (window.modelConfigs[path]) {
            window.modelConfigs[path].proxy_eligible = checkbox.checked;
        }
        showToast('Participação no proxy atualizada', 'success');
        updateProxyPanel(true);
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Falha ao atualizar participação no proxy', 'error');
    }
}

export async function setProxyMaxParallel(input, path) {
    const value = parseInt(input.value, 10);
    if (!Number.isFinite(value) || value < 1) {
        input.value = 1;
        return;
    }
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: jsonHeaders(),
            body: JSON.stringify({ model_path: path, max_parallel_requests: value }),
        });
        if (!res.ok) throw new Error('parallel');
        if (window.modelConfigs[path]) {
            window.modelConfigs[path].max_parallel_requests = value;
        }
        showToast('Limite de paralelismo salvo', 'success');
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
    disabled: 'text-rose-400 border-rose-500/40 bg-rose-500/10',
    not_eligible: 'text-slate-500 border-slate-700/50 bg-slate-800/40',
    offline: 'text-slate-500 border-slate-700/50 bg-slate-800/40',
};

function renderModeBadge(enabled) {
    const badge = document.getElementById('proxy-mode-badge');
    if (!badge) return;
    if (enabled) {
        badge.innerText = 'ON';
        badge.className = 'px-3 py-1 rounded-full text-ui-label font-black tracking-widest uppercase glass border-violet-500/40 text-violet-300 bg-violet-500/10';
    } else {
        badge.innerText = 'OFF';
        badge.className = 'px-3 py-1 rounded-full text-ui-label font-black tracking-widest uppercase glass border-slate-700/50 text-slate-500';
    }
}

function renderOff() {
    renderModeBadge(false);
    const body = document.getElementById('proxy-panel-body');
    if (body) body.classList.add('hidden');
    const exposed = document.getElementById('proxy-exposed-model');
    if (exposed) exposed.innerText = '—';
}

function backendCard(backend) {
    const style = STATE_STYLES[backend.state] || STATE_STYLES.offline;
    const role = backend.role === 'primary' ? 'Principal' : 'Secundário';
    return `
    <div class="p-3 rounded-xl border ${style.split(' ').slice(1).join(' ')} bg-slate-900/40 flex flex-col gap-1" data-proxy-backend="${esc(backend.port)}">
        <div class="flex items-center justify-between gap-2">
            <span class="text-ui-body-sm font-bold text-slate-200 truncate">${esc(backend.model)}</span>
            <span class="text-ui-label font-black uppercase tracking-widest ${style.split(' ')[0]}">${esc(backend.state)}</span>
        </div>
        <div class="flex items-center justify-between text-ui-label text-slate-500">
            <span>${role} · ${esc(backend.gpu)}</span>
            <span class="font-mono">porta ${esc(backend.port)}</span>
        </div>
        <div class="flex items-center justify-between text-ui-label text-slate-500">
            <span>${esc(backend.in_flight)}/${esc(backend.max_parallel)} req ativa(s)</span>
            <span class="font-mono">ctx/slot ${esc(backend.ctx_per_slot)}</span>
        </div>
    </div>`;
}

function sessionRow(session) {
    const label = session.detected_tag || session.affinity_key;
    const lastUsed = (session.last_used_at || '').replace('T', ' ').slice(0, 19);
    const tokens = session.tokens_processed ? ` · ${session.tokens_processed} tokens` : '';
    return `
    <div class="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-slate-900/50 border border-slate-800/60" data-proxy-session="${esc(session.affinity_key)}">
        <div class="min-w-0">
            <p class="text-ui-body-sm font-bold text-slate-200 truncate">${esc(label)} <span class="text-slate-500 font-normal">→ ${esc(session.gpu || '?')} / ${esc(session.internal_model)}</span></p>
            <p class="text-ui-label text-slate-500 font-mono truncate">${esc(session.affinity_key)} · ${esc(session.request_count)} request(s)${esc(tokens)} · ${esc(lastUsed)}</p>
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
    const exposed = document.getElementById('proxy-exposed-model');
    if (exposed) exposed.innerText = status.exposed_model || '(principal offline)';

    const backendsEl = document.getElementById('proxy-backends-list');
    if (backendsEl) {
        backendsEl.innerHTML = (status.backends || []).map(backendCard).join('')
            || '<p class="text-ui-label text-slate-600">Nenhuma instância online.</p>';
    }
    const countEl = document.getElementById('proxy-sessions-count');
    if (countEl) countEl.innerText = `(${sessions.length})`;
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
