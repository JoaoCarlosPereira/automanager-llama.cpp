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

        // Global Status Badge
        const badge = document.getElementById('status-badge');
        const hasInstances = state.activeInstances.length > 0;
        if (hasInstances) {
            badge.className = 'px-4 py-1.5 rounded-full text-[9px] font-black tracking-widest flex items-center gap-2 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
            badge.querySelector('.status-dot').className = 'w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse';
            badge.querySelector('.status-text').innerText = 'ONLINE';
        } else {
            badge.className = 'px-4 py-1.5 rounded-full text-[9px] font-black tracking-widest flex items-center gap-2 glass border-slate-700/50 text-slate-500 uppercase';
            badge.querySelector('.status-dot').className = 'w-1.5 h-1.5 rounded-full bg-slate-600';
            badge.querySelector('.status-text').innerText = 'OFFLINE';
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
            
            if (inst) {
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

export async function updateDownloads() {
    try {
        const res = await apiFetch('/downloads');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const container = document.getElementById('download-list');
        const entries = Object.entries(data.downloads || {});
        if (entries.length === 0) { container.innerHTML = '<p class="text-[9px] text-slate-600 text-center uppercase tracking-widest py-4">Nenhum download ativo</p>'; return; }

        container.innerHTML = entries.map(([id, d]) => {
            const statusClass = d.status === 'completed' ? 'text-emerald-500' : d.status === 'failed' ? 'text-red-500' : 'text-blue-500';
            return `
                <div class="p-3 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <p class="text-[9px] font-bold truncate flex-1 text-slate-400 font-mono">${d.filename}</p>
                        <span class="text-[8px] font-black uppercase ${statusClass}">${d.status}</span>
                    </div>
                    <div class="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-blue-500 transition-all" style="width: ${d.progress}%"></div>
                    </div>
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

export function startDashboardPolling() {
    updateMetrics();
    setInterval(updateMetrics, 2000);
    setInterval(updateStatus, 3000);
    setInterval(updateDownloads, 3000);
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
        const response = await fetch(url, { signal: state.logStream.signal });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const text = decoder.decode(value);
            const formatted = text
                .replace(/error/gi, '<span class="text-red-500 font-black">ERRO</span>')
                .replace(/warn/gi, '<span class="text-amber-500 font-black">AVISO</span>')
                .replace(/info/gi, '<span class="text-blue-400 font-bold">info</span>');
            
            const line = document.createElement('div');
            line.className = 'mb-1 border-l border-slate-800 pl-3';
            line.innerHTML = formatted;
            box.appendChild(line);
            
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
