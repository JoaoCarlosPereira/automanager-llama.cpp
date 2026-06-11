import { state } from './state.js';
import { apiFetch, sessionExpiredHandled } from './auth.js';
import {
    applyGpuWeightsToUI, getContextSize, setContextSize,
    hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,
    updateThinkingBadge, updateMtpBadge,
} from './gpu.js';

export async function updateStatus() {
    try {
        const res = await apiFetch('/status');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const badge = document.getElementById('status-badge');
        const card = document.getElementById('active-card');

        state.activeInstances = data.instances || [];
        updateTabs();

        // Determinar qual instância mostrar no card ativo
        let currentInst = state.activeInstances.find(i => i.port === state.currentActivePort);
        if (!currentInst && state.activeInstances.length > 0) {
            currentInst = state.activeInstances[0];
            state.currentActivePort = currentInst.port;
        }

        if (currentInst) {
            const config = currentInst.config;
            if (config && config.path) {
                window.modelConfigs[config.path] = window.modelConfigs[config.path] || {};
                Object.assign(window.modelConfigs[config.path], config);
                
                if (state.currentSelectedModel === config.path) {
                    updateAutoBalanceProfileBadge(config.auto_balance_profile);
                }

                const configPath = config.path.replace(/\\/g, '/');
                const maySyncPanel = !state.currentSelectedModel || state.currentSelectedModel === configPath;
                
                if (maySyncPanel) {
                    const thinkingToggle = document.getElementById('thinking-toggle');
                    if (thinkingToggle && config.thinking_enabled !== undefined) {
                        thinkingToggle.checked = !!config.thinking_enabled;
                        updateThinkingBadge(!!config.thinking_enabled);
                    }
                    const mtpToggle = document.getElementById('mtp-toggle');
                    if (mtpToggle && config.mtp_enabled !== undefined) {
                        mtpToggle.checked = !!config.mtp_enabled;
                        updateMtpBadge(!!config.mtp_enabled);
                    }
                    if (config.mtp_draft_tokens) {
                        const mtpDraftTokens = document.getElementById('mtp-draft-tokens');
                        if (mtpDraftTokens) {
                            mtpDraftTokens.value = String(config.mtp_draft_tokens);
                        }
                    }
                }

                // Sincronizar pesos de GPU se não houver override manual
                if (!state.manualGpuOverride && config.gpu_weights) {
                    applyGpuWeightsToUI(config.gpu_weights, false);
                }
            }

            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
            badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE';
            card.classList.remove('hidden');
            document.getElementById('active-model-name').innerText = `${currentInst.model}`;
            const titleEl = document.getElementById('active-panel-title');
            if (titleEl) {
                const count = state.activeInstances.length;
                const idx = state.activeInstances.indexOf(currentInst) + 1;
                titleEl.innerText = count > 1 ? `Motor de Computação ${idx}/${count} (Porta ${currentInst.port})` : 'Motor de Computação Primário';
            }
            
            const controls = document.getElementById('active-instance-controls');
            if (controls) {
                controls.innerHTML = `
                    <button onclick="openNativeChat(${currentInst.port}, '${currentInst.model}')" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-[9px] font-black rounded-xl flex items-center gap-2 uppercase tracking-widest shadow-lg shadow-blue-600/20 transition-all whitespace-nowrap">
                        <i class="fas fa-comments text-[8px]"></i> CHAT
                    </button>
                    <button onclick="stopModel(${currentInst.port})" class="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 text-[9px] font-black rounded-xl transition-all uppercase tracking-widest whitespace-nowrap">
                        ENCERRAR
                    </button>
                `;
            }
            
            if (!state.logStream || state.logStreamPort !== currentInst.port) {
                startLogs(currentInst.port);
            }
            
            updateUptime(currentInst.start_time);
            state.currentRunningModelPath = currentInst.model_path;
            
            const chatLink = document.getElementById('chat-link');
            if (chatLink) {
                chatLink.onclick = () => window.openNativeChat(currentInst.port, currentInst.model);
                chatLink.href = 'javascript:void(0)';
                chatLink.classList.remove('pointer-events-none', 'opacity-40');
                chatLink.setAttribute('aria-disabled', 'false');

                const apiLink = document.getElementById('api-link');
                if (apiLink) apiLink.innerText = `http://${window.fixedIp}:${currentInst.port}/v1`;
            }
        } else {
            state.startTime = null;
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase';
            badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE';
            card.classList.add('hidden');
            const controlsOff = document.getElementById('active-instance-controls');
            if (controlsOff) controlsOff.innerHTML = '';
            if (state.logStream) { 
                state.logStream.abort(); 
                state.logStream = null; 
                state.logStreamPort = null;
            }
            state.currentRunningModelPath = null;
            const chatLinkOff = document.getElementById('chat-link');
            if (chatLinkOff) {
                chatLinkOff.classList.add('pointer-events-none', 'opacity-40');
                chatLinkOff.setAttribute('aria-disabled', 'true');
            }
        }

        // --- Lógica de Auto-Balance / Recovery (Geralmente na porta 8085 ou principal) ---
        const autoBalancing = !!(data.recovery && data.recovery.active && data.recovery.auto_balance);
        syncAutoBalanceCancelButton(autoBalancing);
        
        if (autoBalancing && data.recovery.gpu_weights) {
             applyGpuWeightsToUI(data.recovery.gpu_weights, true);
        }

        ensureStatusPolling(autoBalancing);

        if (state.autoBalancePending && data.recovery && !data.recovery.active) {
            state.autoBalancePending = false;
            syncAutoBalanceCancelButton(false);
            const abToggle = document.getElementById('auto-balance-toggle');
            if (data.recovery.cancelled) {
                hideAutoBalanceCapacityAlert();
                if (abToggle) abToggle.checked = false;
                badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-500/50 text-slate-400 uppercase';
                badge.innerHTML = '<i class="fas fa-ban mr-1"></i> AUTO BALANCE CANCELADO';
            } else if (!data.recovery.failed) {
                hideAutoBalanceCapacityAlert();
                if (abToggle) abToggle.checked = false;
                const finalWeights = data.recovery.gpu_weights;
                if (finalWeights) applyGpuWeightsToUI(finalWeights, false);
                updateAutoBalanceProfileBadge(true);
                window.updateModels();
            } else if (data.recovery.hardware_capacity_exceeded) {
                showAutoBalanceCapacityAlert(data.recovery);
                alert(data.recovery.message || 'Modelo além da capacidade do hardware.');
                window.updateModels();
            }
        }

        if (data.recovery && data.recovery.failed && !state.autoBalancePending) {
             // ... handle failure ...
        }

        if (data.recovery && data.recovery.active) {
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-amber-500/50 text-amber-500 uppercase';
            badge.innerHTML = data.recovery.auto_balance 
                ? `<i class="fas fa-sync animate-spin mr-1"></i> AUTO BALANCE: ${(data.recovery.message || 'calibrando...').toUpperCase()}`
                : '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
        }

        // Atualizar lista de modelos para refletir quais estão rodando
        document.querySelectorAll('.model-item-container').forEach(el => {
            const m_js = el.dataset.path.replace(/\\/g, '/');
            const isRunning = state.activeInstances.some(i => i.model_path.replace(/\\/g, '/') === m_js);
            const actionBtnContainer = el.querySelector('.action-btn-container');

            if (isRunning) el.classList.add('running-now');
            else el.classList.remove('running-now');

            if (state.currentSelectedModel === el.dataset.path) el.classList.add('active-selection');
            else el.classList.remove('active-selection');

            const newButtonsHtml = window.getModelButtonsHtml(el.dataset.path, el.id, isRunning);
            if (actionBtnContainer.innerHTML.trim() !== newButtonsHtml.trim()) {
                actionBtnContainer.innerHTML = newButtonsHtml;
            }
        });
    } catch (e) { console.error("updateStatus error:", e); }
}

export function updateTabs() {
    const container = document.getElementById('instance-tabs');
    if (!container) return;
    
    if (state.activeInstances.length <= 1) {
        container.innerHTML = '';
        return;
    }

        const tabsHtml = state.activeInstances.map(inst => {
            const isActive = inst.port === state.currentActivePort;
            const isMain = inst.port === 8085;
            const activeClass = isActive 
                ? 'bg-blue-600 text-white border-blue-500 shadow-lg shadow-blue-500/20' 
                : 'bg-slate-800/50 text-slate-400 border-slate-700 hover:bg-slate-700/50';
            
            const badge = isMain ? '<span class="px-1.5 py-0.5 rounded bg-blue-500/20 text-[7px] text-blue-300 ml-1 border border-blue-500/20">PRINCIPAL</span>' : '';

            return `
                <button onclick="switchInstance(${inst.port})" class="px-4 py-2 rounded-xl border text-[10px] font-black uppercase tracking-widest transition-all flex items-center gap-2 ${activeClass}">
                    <i class="fas fa-cube ${isActive ? 'text-blue-200' : 'text-slate-500'}"></i>
                    ${inst.model} ${badge}
                </button>
            `;
        }).join('');
    
    container.innerHTML = tabsHtml;
}

window.switchInstance = (port) => {
    state.currentActivePort = port;
    const inst = state.activeInstances.find(i => i.port === port);
    if (inst && inst.model_path) {
        state.currentSelectedModel = inst.model_path.replace(/\\/g, '/');
    }
    updateStatus();
};

export function ensureStatusPolling(fast) {
    const ms = fast ? 1000 : 3000;
    if (state.statusPollIntervalMs === ms && state.statusPollTimer) return;
    state.statusPollIntervalMs = ms;
    if (state.statusPollTimer) clearInterval(state.statusPollTimer);
    state.statusPollTimer = setInterval(updateStatus, ms);
}

export async function updateDownloads() {
    try {
        const res = await apiFetch('/downloads');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const container = document.getElementById('download-status');
        const entries = Object.entries(data);
        if (entries.length === 0) { container.innerHTML = ''; return; }

        let html = entries.map(([id, d]) => {
            const statusClass = d.status === 'completed' ? 'bg-emerald-500/10 text-emerald-500' : d.status === 'failed' ? 'bg-red-500/10 text-red-500' : 'bg-blue-500/10 text-blue-500';
            return `
                <div class="p-4 md:p-5 bg-slate-900 border border-slate-800 rounded-2xl">
                    <div class="flex justify-between items-center mb-3 md:mb-4">
                        <p class="text-xs md:text-sm font-bold truncate flex-1 mr-3 md:mr-4 text-slate-300 font-mono">${d.filename}</p>
                        <span class="text-[8px] md:text-[10px] font-black uppercase px-2 md:px-3 py-0.5 md:py-1 rounded ${statusClass}">${d.status}</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-blue-500 transition-all duration-500" style="width: ${d.progress}%"></div>
                    </div>
                </div>`;
        }).join('');
        container.innerHTML = html;
    } catch (e) {}
}

export async function clearCompletedDownloads() {
    try {
        const res = await apiFetch('/downloads/clear', { method: 'POST' });
        if (res.ok) window.updateDownloads();
    } catch (e) {}
}

export function startDashboardPolling() {
    stopDashboardPolling();
    updateMetrics();
    state.metricsTimer = setInterval(updateMetrics, 2000);
    ensureStatusPolling(false);
    state.downloadsTimer = setInterval(updateDownloads, 3000);
    state.modelsTimer = setInterval(() => window.updateModels(), 5000);
}

export function updateUptime(serverStartTime) {
    let diff;
    if (serverStartTime) {
        diff = Math.floor(Date.now() / 1000 - serverStartTime);
    } else if (state.startTime) {
        diff = Math.floor((new Date() - state.startTime) / 1000);
    } else { return; }
    document.getElementById('uptime-val').innerText = `${Math.floor(diff/3600)}h ${Math.floor((diff%3600)/60)}m ${diff%60}s`;
}

export async function startLogs(port = null) {
    if (state.logStream) state.logStream.abort();
    state.logStream = new AbortController();
    state.logStreamPort = port || state.currentActivePort;
    const box = document.getElementById('log-box');
    box.innerHTML = `<div class="text-slate-500 italic">[Conectando logs da porta ${state.logStreamPort}...]</div>`;
    
    try {
        const url = port ? `/logs?port=${port}` : '/logs';
        const response = await fetch(url, { signal: state.logStream.signal });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const formatted = decoder.decode(value)
                .replace(/error/gi, '<span class="text-red-500 font-black px-1 rounded bg-red-500/10">ERRO</span>')
                .replace(/warn/gi, '<span class="text-amber-500 font-black px-1 rounded bg-amber-500/10">AVISO</span>')
                .replace(/info/gi, '<span class="text-blue-400 font-bold uppercase tracking-tighter">info</span>');
            const line = document.createElement('div');
            line.className = 'terminal-line mb-1 md:mb-2 border-l border-slate-800 md:border-l-2 pl-3 md:pl-4';
            line.innerHTML = formatted;
            box.appendChild(line);
            box.scrollTop = box.scrollHeight;
            if (box.childNodes.length > 500) box.removeChild(box.firstChild);
        }
    } catch (e) {}
}

export async function renewToken() {
    try {
        const res = await fetch('/api/key/renew', { method: 'POST' });
        const data = await res.json();
        document.getElementById('api-token').innerText = data.key;
        alert("Nova chave gerada!");
    } catch (e) {}
}

function setBarWidth(el, pct) { if (el) el.style.width = `${pct}%`; }
function setText(el, text) { if (el) el.innerText = text; }

export async function updateMetrics() {
    try {
        const res = await apiFetch('/metrics');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        setText(document.getElementById('cpu-val'), `${data.cpu ?? 0}%`);
        setBarWidth(document.getElementById('cpu-bar'), data.cpu ?? 0);
        setText(document.getElementById('ram-val'), `${data.ram ?? 0}%`);
        setBarWidth(document.getElementById('ram-bar'), data.ram ?? 0);
        
        (data.gpus || []).forEach((g) => {
            const row = document.querySelector(`.gpu-row[data-index="${g.index}"]`);
            if (!row) return;
            setText(row.querySelector('.gpu-util-val'), `${g.util}%`);
            setBarWidth(row.querySelector('.gpu-util-bar'), g.util);
            setText(row.querySelector('.gpu-temp-val'), `${g.temp || '--'}°C`);
            setText(row.querySelector('.gpu-power-val'), `${g.power || '--'}W`);
            setText(row.querySelector('.gpu-vram-text'), `${g.mem_used} / ${g.mem_total} MB`);
            setBarWidth(row.querySelector('.gpu-vram-bar'), g.vram_pct ?? 0);
        });
    } catch (e) {}
}

export function stopDashboardPolling() {
    if (state.statusPollTimer) clearInterval(state.statusPollTimer);
    if (state.metricsTimer) clearInterval(state.metricsTimer);
    if (state.downloadsTimer) clearInterval(state.downloadsTimer);
    if (state.modelsTimer) clearInterval(state.modelsTimer);
    if (state.logStream) state.logStream.abort();
}
