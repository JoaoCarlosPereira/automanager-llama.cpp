import { state } from './state.js?v=4.2.0';
import { apiFetch, sessionExpiredHandled, showToast } from './auth.js?v=4.2.0';
import {
    applyGpuWeightsToUI, getContextSize, setContextSize,
    hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,
    showAutoBalanceProgress, hideAutoBalanceProgress,
} from './gpu.js?v=4.2.0';
import { getTabActionsHtml } from './models.js?v=4.2.0';
import { updateProxyPanel } from './proxy.js?v=4.2.0';

export async function updateStatus() {
    try {
        const res = await apiFetch('/status');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();

        // Painel do Proxy Inteligente acompanha o polling (throttle interno)
        updateProxyPanel();

        state.activeInstances = data.instances || [];
        const runningInstances = state.activeInstances.filter(i => i.status === 'running');
        const recovery = data.recovery;
        const autoBalancing = !!(recovery?.active && recovery?.auto_balance);
        const showAutoBalanceUi = autoBalancing || state.autoBalancePending;

        if (autoBalancing) {
            state.autoBalanceSeenActive = true;
        }

        // Global Status Badge (skip while auto-balance UI is active)
        const badge = document.getElementById('status-badge');
        const hasInstances = runningInstances.length > 0;
        if (badge && !showAutoBalanceUi) {
            const dot = badge.querySelector('.status-dot');
            const txt = badge.querySelector('.status-text');
            if (hasInstances) {
                badge.className = 'px-4 py-1.5 rounded-full text-ui-label font-black tracking-widest flex items-center gap-2 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
                if (dot) dot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse';
                if (txt) txt.innerText = 'ONLINE';
            } else {
                badge.className = 'px-4 py-1.5 rounded-full text-ui-label font-black tracking-widest flex items-center gap-2 glass border-slate-700/50 text-slate-500 uppercase';
                if (dot) dot.className = 'w-1.5 h-1.5 rounded-full bg-slate-600';
                if (txt) txt.innerText = 'OFFLINE';
            }
        }

        // --- Sincronizar Abas ---
        state.activeTabs.forEach(tab => {
            const path = tab.path;
            const inst = state.activeInstances.find(i => (i.model_path || '').replace(/\\/g, '/') === path);
            const tabEl = document.getElementById(tab.id);
            if (!tabEl) return;

            if (isTabAutoBalancing(tab, recovery, showAutoBalanceUi)) {
                return;
            }

            const statusBadge = tabEl.querySelector('.tab-status-badge');
            const actions = tabEl.querySelector('.tab-actions');
            const tabBtn = document.getElementById(`btn-${tab.id}`);
            const dot = tabBtn?.querySelector('.tab-status-dot');

            const isRunning = inst && inst.status === 'running';
            // Só reescreve badge/ações quando o estado realmente muda: recriar os
            // botões a cada poll (1s) matava hover/transições e engolia cliques.
            const renderKey = isRunning
                ? `running:${inst.port}`
                : (inst && inst.status === 'stopped') ? 'stopped' : 'offline';
            const stateChanged = tabEl.dataset.lastRenderKey !== renderKey;

            if (isRunning) {
                if (stateChanged) {
                    statusBadge.innerText = 'ONLINE';
                    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-emerald-500/40 text-emerald-400 bg-emerald-500/5';
                    actions.innerHTML = getTabActionsHtml(path, tab.id, true, inst.port);
                    if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981] animate-pulse shrink-0 transition-all duration-500';
                }

                if (document.activeElement?.closest(`#${tab.id}`) === null) {
                    if (inst.config && !tabEl?.dataset.proposal) {
                        // Merge: preserva ajustes locais (mmproj/pins) ausentes em inst.config.
                        window.modelConfigs[path] = { ...(window.modelConfigs[path] || {}), ...inst.config };
                    }
                }
            } else if (stateChanged && inst && inst.status === 'stopped') {
                statusBadge.innerText = 'ERRO';
                statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-rose-500/40 text-rose-400 bg-rose-500/5';
                actions.innerHTML = getTabActionsHtml(path, tab.id, false);
                if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0 transition-all duration-500';
            } else if (stateChanged) {
                statusBadge.innerText = 'OFFLINE';
                statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-slate-700/50 text-slate-500';
                actions.innerHTML = getTabActionsHtml(path, tab.id, false);
                if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0 transition-all duration-500';
            }
            tabEl.dataset.lastRenderKey = renderKey;

            if (inst && (inst.status === 'running' || inst.status === 'stopped')) {
                if (state.currentTabId === tab.id) {
                    attachTabLogs(tab.id, inst.port);
                }
            } else if (state.currentTabId === tab.id && state.logStreamTabId === tab.id) {
                detachTabLogs();
            }
        });

        // --- Lógica de Auto-Balance / Recovery ---
        if (isAutoBalanceRunComplete(recovery)) {
            state.autoBalancePending = false;
            state.autoBalanceTabId = null;
            state.autoBalanceSeenActive = false;
            state.autoBalanceRunId = null;
            hideAutoBalanceProgress();

            const recoveryTab = findRecoveryTab(recovery);
            if (recoveryTab) {
                const tabId = recoveryTab.id;

                if (recovery.failed) {
                    showAutoBalanceCapacityAlert(recovery, tabId);
                } else if (!recovery.cancelled) {
                    if (recovery.smart_proposal) {
                        import('./models.js?v=4.2.0').then(m => {
                            m.restoreScreenSnapshot(tabId);
                            m.showProposedConfig(
                                tabId,
                                recovery.smart_proposal,
                                recovery.gpu_weights,
                            );
                        });
                    } else if (recovery.gpu_weights) {
                        applyGpuWeightsToUI(recovery.gpu_weights, false, tabId);
                    }

                    if (recovery.smart_proposal) {
                        const tabEl = document.getElementById(tabId);
                        tabEl?.querySelector('.tab-proposed-config')?.scrollIntoView({
                            behavior: 'smooth',
                            block: 'nearest',
                        });
                    }
                }
            }
            window.updateModels();
        }

        if (showAutoBalanceUi) {
            if (autoBalancing) {
                state.autoBalancePending = true;
            }

            const recoveryTab = findRecoveryTab(recovery);
            const progressRecovery = autoBalancing
                ? recovery
                : {
                    message: 'Iniciando auto-balance...',
                    smart_calibration: true,
                    attempt: 0,
                };

            if (recoveryTab) {
                state.autoBalanceTabId = recoveryTab.id;
                const tabEl = document.getElementById(recoveryTab.id);
                const statusBadge = tabEl?.querySelector('.tab-status-badge');
                const tabBtn = document.getElementById(`btn-${recoveryTab.id}`);
                const dot = tabBtn?.querySelector('.tab-status-dot');
                const statusLabel = (recovery?.smart_calibration ?? true)
                    ? 'CALIBRANDO...'
                    : 'AUTO-BALANCE...';

                if (statusBadge) {
                    statusBadge.innerHTML = `<i class="fas fa-sync animate-spin mr-2"></i> ${statusLabel}`;
                    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-amber-500/40 text-amber-400 bg-amber-500/5';
                }
                if (dot) {
                    dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-amber-500 shadow-[0_0_5px_#f59e0b] animate-pulse shrink-0 transition-all duration-500';
                }
                showAutoBalanceProgress(progressRecovery, recoveryTab.id);
                if (recovery?.gpu_weights) {
                    applyGpuWeightsToUI(recovery.gpu_weights, true, recoveryTab.id);
                }
                if (recoveryTab && state.currentTabId === recoveryTab.id) {
                    const inst = state.activeInstances.find(
                        i => normalizePath(i.model_path) === recoveryTab.path
                    );
                    attachTabLogs(recoveryTab.id, inst?.port ?? state.currentActivePort);
                }
            }

            const globalBadge = document.getElementById('status-badge');
            if (globalBadge) {
                const globalLabel = (recovery?.smart_calibration ?? state.autoBalancePending)
                    ? 'CALIBRANDO'
                    : 'AUTO-BALANCE';
                globalBadge.className = 'px-4 py-1.5 rounded-full text-ui-label font-black tracking-[0.2em] flex items-center gap-2 glass border-amber-500/30 text-amber-400 uppercase';
                globalBadge.innerHTML = `<span class="status-dot w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse"></span><span class="status-text">${globalLabel}</span>`;
            }
        } else if (!recovery?.active) {
            hideAutoBalanceProgress();
        }

        window.updateModels();
    } catch (e) { console.error("updateStatus error:", e); }
}

function normalizePath(p) {
    return (p || '').replace(/\\/g, '/');
}

function findRecoveryTab(recovery) {
    const modelPath = normalizePath(recovery?.model);
    if (modelPath) {
        // Com modelo conhecido, só casa pela aba do mesmo modelo. Cair para a aba
        // atual aplicaria badge "CALIBRANDO" e sobrescreveria os pesos de OUTRO modelo.
        const byPath = state.activeTabs.find(t => t.path === modelPath);
        if (byPath) return byPath;
        const byId = state.activeTabs.find(t => t.id === state.autoBalanceTabId);
        return byId || null;
    }
    if (state.autoBalanceTabId) {
        const byId = state.activeTabs.find(t => t.id === state.autoBalanceTabId);
        if (byId) return byId;
    }
    if (state.currentTabId) {
        const current = state.activeTabs.find(t => t.id === state.currentTabId);
        if (current) return current;
    }
    return null;
}

function isTabAutoBalancing(tab, recovery, showAutoBalanceUi) {
    if (!showAutoBalanceUi) return false;
    if (state.autoBalanceTabId && tab.id === state.autoBalanceTabId) return true;
    const modelPath = normalizePath(recovery?.model);
    return !!(modelPath && tab.path === modelPath);
}

function isAutoBalanceRunComplete(recovery) {
    if (!state.autoBalancePending || !recovery || recovery.active) return false;
    if (state.autoBalanceRunId != null && recovery.run_id != null) {
        return recovery.run_id === state.autoBalanceRunId;
    }
    return state.autoBalanceSeenActive;
}

function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1);
    return `${(n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatSpeed(bytesPerSec) {
    const n = Number(bytesPerSec) || 0;
    if (n <= 0) return '--';
    return `${formatBytes(n)}/s`;
}

function formatDuration(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    if (total < 60) return `${total}s`;
    const minutes = Math.floor(total / 60);
    const rem = total % 60;
    if (minutes < 60) return `${minutes}m ${rem}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
}

export async function cancelDownload(downloadId) {
    if (!downloadId) return;
    try {
        const res = await apiFetch('/downloads/cancel', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({download_id: downloadId}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast('Erro: ' + (err.detail || 'Falha ao cancelar download'), 'error');
            return;
        }
        await updateDownloads();
        window.updateModels?.();
    } catch (e) {
        showToast('Erro de rede ao cancelar download.', 'error');
    }
}

export async function updateDownloads() {
    try {
        const res = await apiFetch('/downloads');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const container = document.getElementById('download-list');
        const entries = Object.entries(data.downloads || {});
        if (entries.length === 0) { container.innerHTML = '<p class="text-ui-label text-slate-600 text-center uppercase tracking-widest py-4">Nenhum download ativo</p>'; return; }

        container.innerHTML = entries.map(([id, d]) => {
            const statusClass = d.status === 'completed' ? 'text-emerald-500'
                : d.status === 'failed' ? 'text-red-500'
                : d.status === 'cancelled' ? 'text-amber-500'
                : 'text-blue-500';
            const progress = Number(d.progress) || 0;
            const isActive = d.status === 'downloading' || d.status === 'cancelling';
            const elapsed = formatDuration(d.elapsed_seconds);
            const eta = d.eta_seconds != null && d.eta_seconds > 0
                ? formatDuration(d.eta_seconds)
                : '--';
            const familyLabel = d.family ? `<span class="text-ui-caption text-slate-600 uppercase tracking-widest">${d.family}</span>` : '';
            return `
                <div class="p-3 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
                    <div class="flex justify-between items-center gap-2">
                        <div class="min-w-0 flex-1">
                            <p class="text-ui-label font-bold truncate text-slate-400 font-mono">${d.filename}</p>
                            ${familyLabel}
                        </div>
                        <span class="text-ui-label font-black uppercase shrink-0 ${statusClass}">${d.status}</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-blue-500 transition-all" style="width: ${progress}%"></div>
                    </div>
                    <div class="flex justify-between items-center text-ui-label font-mono text-slate-500">
                        <span>${formatBytes(d.downloaded_bytes)} / ${formatBytes(d.total_bytes)}</span>
                        <span class="flex items-center gap-2">
                            ${isActive ? `<span class="text-blue-400">${formatSpeed(d.speed_bps)}</span>` : ''}
                            <span class="text-slate-300 font-black">${progress.toFixed(1)}%</span>
                        </span>
                    </div>
                    <div class="flex justify-between items-center text-ui-label font-mono text-slate-500">
                        <span>Tempo: ${elapsed}</span>
                        ${isActive ? `<span>ETA: ${eta}</span>` : ''}
                    </div>
                    ${isActive ? `
                        <button type="button" onclick="cancelDownload('${id}')"
                            class="w-full py-1.5 rounded-lg border border-red-500/30 text-ui-label font-black uppercase tracking-widest text-red-400 hover:bg-red-500/10 transition-all">
                            Cancelar
                        </button>` : ''}
                </div>`;
        }).join('');
    } catch (e) {}
}

export async function clearCompletedDownloads() {
    try {
        const res = await apiFetch('/downloads/clear', { method: 'POST' });
        await window.updateDownloads();
        if (res.ok) showToast('Downloads concluídos removidos da lista.', 'success');
    } catch (e) {
        showToast('Erro de rede ao limpar downloads.', 'error');
    }
}

let dashboardPollIntervals = [];

export function startDashboardPolling() {
    stopDashboardPolling();
    updateMetrics();
    dashboardPollIntervals.push(setInterval(updateMetrics, 2000));
    dashboardPollIntervals.push(setInterval(updateStatus, 1000));
    dashboardPollIntervals.push(setInterval(updateDownloads, 3000));
}

export function stopDashboardPolling() {
    dashboardPollIntervals.forEach(clearInterval);
    dashboardPollIntervals = [];
}

export function detachTabLogs() {
    if (state.logStream) state.logStream.abort();
    state.logStream = null;
    state.logStreamPort = null;
    state.logStreamTabId = null;
    state.logStreamSessionKey = null;
}

function buildLogSessionKey(port, startTime) {
    return `${port}:${startTime ?? 0}`;
}

export function attachTabLogs(tabId, portOverride = null, { force = false, sessionKey = null } = {}) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    const path = normalizePath(tab.dataset.path);
    const inst = state.activeInstances.find(
        i => normalizePath(i.model_path) === path
    );
    // Sem currentActivePort de fallback: para uma aba sem instância própria isso
    // conectaria o console aos logs de OUTRA instância. Só usa override explícito
    // ou a porta da instância desta aba.
    let port = portOverride ?? inst?.port;
    port = Number(port);
    if (!Number.isFinite(port) || port <= 0) {
        detachTabLogs();
        const box = tab.querySelector('.tab-log-box');
        if (box) {
            box.innerHTML = '';
            box.dataset.connecting = '1';
            appendLogLine(box, 'Aguardando instância...', { tone: 'muted', replaceConnecting: false });
        }
        setLogStreamStatus(tab, 'stopped');
        return;
    }

    const key = sessionKey ?? buildLogSessionKey(port, inst?.start_time);
    if (
        !force
        && state.logStreamSessionKey === key
        && state.logStreamTabId === tabId
        && state.logStream
    ) {
        return;
    }
    startLogs(port, tabId, key);
}

function consumeLogSseBuffer(buffer, box, tab) {
    const parts = buffer.split('\n');
    const remainder = parts.pop() || '';
    let added = 0;
    for (const rawLine of parts) {
        if (!rawLine.startsWith('data:')) continue;
        const lineText = rawLine.slice(5).replace(/^\s/, '');
        if (!lineText) continue;
        appendLogLine(box, lineText);
        added += lineText.length + 1;
    }
    // Contador incremental em vez de ler box.innerText (força reflow de até 800 nós a cada chunk).
    const sizeEl = tab?.querySelector('.tab-log-size');
    if (sizeEl && added) {
        const bytes = (Number(box.dataset.logBytes) || 0) + added;
        box.dataset.logBytes = String(bytes);
        sizeEl.innerText = `${(bytes / 1024).toFixed(1)} KB`;
    }
    return remainder;
}

function escapeLogHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function formatLogLine(text) {
    return escapeLogHtml(text)
        .replace(/\berror\b/gi, '<span class="text-red-500 font-black">ERRO</span>')
        .replace(/\bwarn(?:ing)?\b/gi, '<span class="text-amber-500 font-black">AVISO</span>')
        .replace(/\binfo\b/gi, '<span class="text-blue-400 font-bold">info</span>');
}

function appendLogLine(box, text, { tone = 'default', replaceConnecting = true } = {}) {
    if (replaceConnecting && box.dataset.connecting === '1') {
        box.innerHTML = '';
        delete box.dataset.connecting;
    }
    const line = document.createElement('div');
    const toneClass = tone === 'muted'
        ? 'text-slate-600'
        : tone === 'error'
            ? 'text-red-400'
            : tone === 'warn'
                ? 'text-amber-400'
                : 'text-slate-400';
    line.className = `mb-1 border-l border-slate-800 pl-3 ${toneClass}`;
    line.innerHTML = formatLogLine(text);
    // Stick-to-bottom: só rola se o usuário já estava no fim, para não arrastar
    // de volta quem rolou para cima lendo um erro.
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.appendChild(line);
    while (box.childNodes.length > 800) box.removeChild(box.firstChild);
    if (atBottom) box.scrollTop = box.scrollHeight;
}

function setLogStreamStatus(tab, status) {
    const statusEl = tab?.querySelector('.tab-log-status');
    if (!statusEl) return;
    if (status === 'live') {
        statusEl.textContent = 'Fluxo de Dados Ativo';
        statusEl.className = 'tab-log-status text-ui-label font-black text-emerald-500 uppercase tracking-widest';
    } else if (status === 'connecting') {
        statusEl.textContent = 'Conectando...';
        statusEl.className = 'tab-log-status text-ui-label font-black text-slate-500 uppercase tracking-widest';
    } else {
        statusEl.textContent = 'Fluxo interrompido';
        statusEl.className = 'tab-log-status text-ui-label font-black text-amber-500 uppercase tracking-widest';
    }
}

export async function startLogs(port, tabId, sessionKey = null) {
    port = Number(port);
    if (!Number.isFinite(port) || port <= 0) return;

    if (state.logStream) state.logStream.abort();

    state.logStream = new AbortController();
    state.logStreamPort = port;
    state.logStreamTabId = tabId;
    state.logStreamSessionKey = sessionKey ?? buildLogSessionKey(port, Date.now());

    const tab = document.getElementById(tabId);
    const box = tab?.querySelector('.tab-log-box');
    if (!box) return;

    box.innerHTML = '';
    box.dataset.connecting = '1';
    box.dataset.logBytes = '0';
    appendLogLine(box, 'Conectando ao console da instancia...', { tone: 'muted', replaceConnecting: false });
    setLogStreamStatus(tab, 'connecting');

    try {
        const response = await fetch(`/logs?port=${port}`, {
            signal: state.logStream.signal,
            credentials: 'include',
        });
        if (!response.ok) {
            appendLogLine(box, `Erro ao conectar logs (HTTP ${response.status}).`, { tone: 'error' });
            setLogStreamStatus(tab, 'stopped');
            return;
        }

        setLogStreamStatus(tab, 'live');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            buffer = consumeLogSseBuffer(buffer, box, tab);
        }
    } catch (e) {
        if (e?.name === 'AbortError') return;
        appendLogLine(box, 'Conexao de logs interrompida.', { tone: 'warn' });
        setLogStreamStatus(tab, 'stopped');
    }
}

function formatApiTokenDisplay(key) {
    if (!key || key.length <= 20) return key;
    return key.substring(0, 11) + '…' + key.substring(key.length - 8);
}

function copyTextFallback(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
}

export async function copyApiToken() {
    const el = document.getElementById('api-token');
    const token = el?.dataset?.fullToken;
    if (!token) return;

    let copied = false;
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(token);
            copied = true;
        }
    } catch (_) {}

    if (!copied) {
        copied = copyTextFallback(token);
    }

    if (!copied) return;

    const btn = el.parentElement?.querySelector('button');
    const icon = btn?.querySelector('i');
    if (!icon) return;
    icon.classList.remove('far', 'fa-copy');
    icon.classList.add('fas', 'fa-check', 'text-emerald-500');
    setTimeout(() => {
        icon.classList.remove('fas', 'fa-check', 'text-emerald-500');
        icon.classList.add('far', 'fa-copy');
    }, 1500);
}

export async function refreshApiToken() {
    // Popula o token após login SPA: o GET / público renderiza o campo
    // vazio de propósito (não vazar o segredo para anônimos).
    const el = document.getElementById('api-token');
    if (!el || el.dataset.fullToken) return;
    try {
        const res = await apiFetch('/api/key');
        if (!res.ok) return;
        const data = await res.json();
        if (data.key) {
            el.dataset.fullToken = data.key;
            el.textContent = formatApiTokenDisplay(data.key);
        }
    } catch (e) {}
}

export async function renewToken() {
    try {
        const res = await apiFetch('/api/key/renew', { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            const el = document.getElementById('api-token');
            if (el && data.key) {
                el.dataset.fullToken = data.key;
                el.textContent = formatApiTokenDisplay(data.key);
            }
        }
    } catch (e) {}
}

export async function updateMetrics() {
    try {
        const res = await apiFetch('/metrics');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        
        const cpuVal = document.getElementById('cpu-val');
        const cpuBar = document.getElementById('cpu-bar');
        const ramVal = document.getElementById('ram-val');
        const ramBar = document.getElementById('ram-bar');
        
        if (cpuVal) cpuVal.innerText = `${data.cpu ?? 0}%`;
        if (cpuBar) cpuBar.style.width = `${data.cpu ?? 0}%`;
        if (ramVal) ramVal.innerText = `${data.ram ?? 0}%`;
        if (ramBar) ramBar.style.width = `${data.ram ?? 0}%`;

        // Per-GPU VRAM usage bars inside each model tab's allocation table
        (data.gpus || []).forEach(g => {
            document.querySelectorAll(`.gpu-row[data-index="${g.index}"]`).forEach(row => {
                const vramText = row.querySelector('.gpu-vram-text');
                const vramBar = row.querySelector('.gpu-vram-bar');
                if (vramText) vramText.innerText = `${g.mem_used} / ${g.mem_total} MB`;
                if (vramBar) vramBar.style.width = `${g.vram_pct ?? 0}%`;
            });
        });
        
        // Mini GPU Cards — atualização keyed (sem rebuild via innerHTML a cada 2s,
        // que causava flicker e reiniciava transições).
        renderMiniGpuCards(data.gpus || []);
    } catch (e) {}
}

function renderMiniGpuCards(gpus) {
    const miniGpu = document.getElementById('mini-gpu-metrics');
    if (!miniGpu) return;

    // (Re)cria os cards apenas quando o conjunto de GPUs muda.
    const key = gpus.map(g => g.index).join(',');
    if (miniGpu.dataset.gpuKey !== key) {
        miniGpu.dataset.gpuKey = key;
        miniGpu.innerHTML = gpus.map(g => `
            <div data-gpu-card="${g.index}" class="glass min-w-[120px] p-3 rounded-xl border-b-2 border-blue-500/50 flex flex-col justify-between shrink-0">
                <div class="flex justify-between items-center mb-1">
                    <span class="text-ui-label font-black text-slate-500 uppercase">GPU ${g.index}</span>
                    <span data-gpu-temp class="text-ui-label font-mono text-slate-400">--°C</span>
                </div>
                <div class="flex items-center gap-2">
                    <span data-gpu-util class="text-xs font-bold text-white">0%</span>
                    <div class="flex-1 h-0.5 bg-slate-800 rounded-full overflow-hidden">
                        <div data-gpu-util-bar class="h-full bg-blue-500 transition-all duration-700" style="width: 0%"></div>
                    </div>
                </div>
                <p data-gpu-mem class="text-ui-label text-slate-400 font-mono mt-1">0MB</p>
            </div>
        `).join('');
    }

    for (const g of gpus) {
        const card = miniGpu.querySelector(`[data-gpu-card="${g.index}"]`);
        if (!card) continue;
        const temp = card.querySelector('[data-gpu-temp]');
        const util = card.querySelector('[data-gpu-util]');
        const bar = card.querySelector('[data-gpu-util-bar]');
        const mem = card.querySelector('[data-gpu-mem]');
        if (temp) temp.textContent = `${g.temp || '--'}°C`;
        if (util) util.textContent = `${g.util}%`;
        if (bar) bar.style.width = `${g.util}%`;
        if (mem) mem.textContent = `${g.mem_used}MB`;
    }
}
