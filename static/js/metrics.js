import { state } from './state.js';
import { apiFetch, sessionExpiredHandled } from './auth.js';
import {
    applyGpuWeightsToUI, getContextSize, setContextSize,
    hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,
} from './gpu.js';
import { getTabActionsHtml } from './models.js';

export async function updateStatus() {
    try {
        const res = await apiFetch('/status');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        
        state.activeInstances = data.instances || [];
        const runningInstances = state.activeInstances.filter(i => i.status === 'running');

        // Global Status Badge
        const badge = document.getElementById('status-badge');
        const hasInstances = runningInstances.length > 0;
        if (badge) {
            const dot = badge.querySelector('.status-dot');
            const txt = badge.querySelector('.status-text');
            if (hasInstances) {
                badge.className = 'px-4 py-1.5 rounded-full text-[9px] font-black tracking-widest flex items-center gap-2 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
                if (dot) dot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse';
                if (txt) txt.innerText = 'ONLINE';
            } else {
                badge.className = 'px-4 py-1.5 rounded-full text-[9px] font-black tracking-widest flex items-center gap-2 glass border-slate-700/50 text-slate-500 uppercase';
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

            const statusBadge = tabEl.querySelector('.tab-status-badge');
            const actions = tabEl.querySelector('.tab-actions');
            const tabBtn = document.getElementById(`btn-${tab.id}`);
            const dot = tabBtn?.querySelector('.tab-status-dot');
            
            const isRunning = inst && inst.status === 'running';
            if (isRunning) {
                statusBadge.innerText = 'ONLINE';
                statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-[10px] font-black tracking-[0.2em] uppercase glass border-emerald-500/40 text-emerald-400 bg-emerald-500/5';
                actions.innerHTML = getTabActionsHtml(path, tab.id, true, inst.port);
                
                if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981] animate-pulse shrink-0 transition-all duration-500';
                
                // Real-time config sync if not interacting
                if (document.activeElement?.closest(`#${tab.id}`) === null) {
                    if (inst.config) {
                        window.modelConfigs[path] = inst.config;
                        // applyModelConfig(path, tab.id); // Maybe too aggressive?
                    }
                }
                
                // Start logs for this tab if not already streaming
                if (!state.logStream || state.logStreamPort !== inst.port) {
                     // We need a way to track multiple streams or switch context
                     // For now, let's just ensure the CURRENT tab is streaming
                     if (state.currentTabId === tab.id) startLogs(inst.port, tab.id);
                }
            } else if (inst && inst.status === 'stopped') {
                statusBadge.innerText = 'ERRO';
                statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-[10px] font-black tracking-[0.2em] uppercase glass border-rose-500/40 text-rose-400 bg-rose-500/5';
                actions.innerHTML = getTabActionsHtml(path, tab.id, false);

                if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0 transition-all duration-500';
            } else {
                statusBadge.innerText = 'OFFLINE';
                statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-[10px] font-black tracking-[0.2em] uppercase glass border-slate-700/50 text-slate-500';
                actions.innerHTML = getTabActionsHtml(path, tab.id, false);
                
                if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0 transition-all duration-500';
            }
        });

        // --- Lógica de Auto-Balance / Recovery ---
        if (state.autoBalancePending && data.recovery && !data.recovery.active) {
            state.autoBalancePending = false;
            
            const recoveryTab = state.activeTabs.find(t => t.path === normalizePath(data.recovery.model));
            if (recoveryTab) {
                const tabId = recoveryTab.id;
                
                if (data.recovery.failed) {
                    showAutoBalanceCapacityAlert(data.recovery, tabId);
                } else if (!data.recovery.cancelled) {
                    // IF it was a smart calibration, show the proposal
                    if (data.recovery.smart_proposal) {
                        import('./models.js').then(m => m.showProposedConfig(tabId, data.recovery.smart_proposal));
                    }
                    
                    // Always apply weights if balanced successfully
                    if (data.recovery.gpu_weights) {
                        applyGpuWeightsToUI(data.recovery.gpu_weights, false, tabId);
                    }
                }
            }
            window.updateModels();
        }

        if (data.recovery && data.recovery.active) {
            const recoveryTab = state.activeTabs.find(t => t.path === normalizePath(data.recovery.model));
            if (recoveryTab) {
                const tabEl = document.getElementById(recoveryTab.id);
                const statusBadge = tabEl?.querySelector('.tab-status-badge');
                if (statusBadge) {
                    statusBadge.innerHTML = `<i class="fas fa-sync animate-spin mr-2"></i> ${data.recovery.auto_balance ? 'CALIBRANDO...' : 'RECARREGANDO...'}`;
                    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-[10px] font-black tracking-[0.2em] uppercase glass border-amber-500/40 text-amber-400 bg-amber-500/5';
                }
                if (data.recovery.gpu_weights) applyGpuWeightsToUI(data.recovery.gpu_weights, true, recoveryTab.id);
            }
        }

        window.updateModels();
    } catch (e) { console.error("updateStatus error:", e); }
}

function normalizePath(p) {
    return (p || '').replace(/\\/g, '/');
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
            alert('Erro: ' + (err.detail || 'Falha ao cancelar download'));
            return;
        }
        await updateDownloads();
        window.updateModels?.();
    } catch (e) {
        alert('Erro de rede ao cancelar download.');
    }
}

export async function updateDownloads() {
    try {
        const res = await apiFetch('/downloads');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const container = document.getElementById('download-list');
        const entries = Object.entries(data.downloads || {});
        if (entries.length === 0) { container.innerHTML = '<p class="text-[9px] text-slate-600 text-center uppercase tracking-widest py-4">Nenhum download ativo</p>'; return; }

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
            const familyLabel = d.family ? `<span class="text-[7px] text-slate-600 uppercase tracking-widest">${d.family}</span>` : '';
            return `
                <div class="p-3 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
                    <div class="flex justify-between items-center gap-2">
                        <div class="min-w-0 flex-1">
                            <p class="text-[9px] font-bold truncate text-slate-400 font-mono">${d.filename}</p>
                            ${familyLabel}
                        </div>
                        <span class="text-[8px] font-black uppercase shrink-0 ${statusClass}">${d.status}</span>
                    </div>
                    <div class="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-blue-500 transition-all" style="width: ${progress}%"></div>
                    </div>
                    <div class="flex justify-between items-center text-[8px] font-mono text-slate-500">
                        <span>${formatBytes(d.downloaded_bytes)} / ${formatBytes(d.total_bytes)}</span>
                        <span class="flex items-center gap-2">
                            ${isActive ? `<span class="text-blue-400">${formatSpeed(d.speed_bps)}</span>` : ''}
                            <span class="text-slate-300 font-black">${progress.toFixed(1)}%</span>
                        </span>
                    </div>
                    <div class="flex justify-between items-center text-[8px] font-mono text-slate-500">
                        <span>Tempo: ${elapsed}</span>
                        ${isActive ? `<span>ETA: ${eta}</span>` : ''}
                    </div>
                    ${isActive ? `
                        <button type="button" onclick="cancelDownload('${id}')"
                            class="w-full py-1.5 rounded-lg border border-red-500/30 text-[8px] font-black uppercase tracking-widest text-red-400 hover:bg-red-500/10 transition-all">
                            Cancelar
                        </button>` : ''}
                </div>`;
        }).join('');
    } catch (e) {}
}

export async function clearCompletedDownloads() {
    try {
        await apiFetch('/downloads/clear', { method: 'POST' });
        window.updateDownloads();
    } catch (e) {}
}

let dashboardPollIntervals = [];

export function startDashboardPolling() {
    stopDashboardPolling();
    updateMetrics();
    dashboardPollIntervals.push(setInterval(updateMetrics, 2000));
    dashboardPollIntervals.push(setInterval(updateStatus, 3000));
    dashboardPollIntervals.push(setInterval(updateDownloads, 3000));
}

export function stopDashboardPolling() {
    dashboardPollIntervals.forEach(clearInterval);
    dashboardPollIntervals = [];
}

export async function startLogs(port, tabId) {
    if (state.logStream) state.logStream.abort();
    state.logStream = new AbortController();
    state.logStreamPort = port;
    
    const tab = document.getElementById(tabId);
    const box = tab?.querySelector('.tab-log-box');
    if (!box) return;
    
    try {
        const url = `/logs?port=${port}`;
        const response = await fetch(url, { signal: state.logStream.signal, credentials: 'include' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const text = decoder.decode(value);
            const lines = text.split('\n').filter(Boolean);
            for (const rawLine of lines) {
                const lineText = rawLine.startsWith('data: ') ? rawLine.slice(6) : rawLine;
                const formatted = lineText
                    .replace(/error/gi, '<span class="text-red-500 font-black">ERRO</span>')
                    .replace(/warn/gi, '<span class="text-amber-500 font-black">AVISO</span>')
                    .replace(/info/gi, '<span class="text-blue-400 font-bold">info</span>');

                const line = document.createElement('div');
                line.className = 'mb-1 border-l border-slate-800 pl-3';
                line.innerHTML = formatted;
                box.appendChild(line);
            }
            
            if (box.childNodes.length > 500) box.removeChild(box.firstChild);
            box.scrollTop = box.scrollHeight;
            
            // Stats
            const sizeEl = tab.querySelector('.tab-log-size');
            if (sizeEl) sizeEl.innerText = `${(box.innerText.length / 1024).toFixed(1)} KB`;
        }
    } catch (e) {}
}

export async function renewToken() {
    try {
        // Needs proper endpoint
        // await apiFetch('/api/key/renew', { method: 'POST' });
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
        
        // Mini GPU Cards
        const miniGpu = document.getElementById('mini-gpu-metrics');
        if (miniGpu) {
            miniGpu.innerHTML = (data.gpus || []).map(g => `
                <div class="glass min-w-[120px] p-3 rounded-xl border-b-2 border-blue-500/50 flex flex-col justify-between shrink-0">
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-[8px] font-black text-slate-500 uppercase">GPU ${g.index}</span>
                        <span class="text-[8px] font-mono text-slate-400">${g.temp || '--'}°C</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="text-xs font-bold text-white">${g.util}%</span>
                        <div class="flex-1 h-0.5 bg-slate-800 rounded-full overflow-hidden">
                            <div class="h-full bg-blue-500" style="width: ${g.util}%"></div>
                        </div>
                    </div>
                    <p class="text-[7px] text-slate-600 font-mono mt-1">${g.mem_used}MB</p>
                </div>
            `).join('');
        }
    } catch (e) {}
}
