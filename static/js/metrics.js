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

        state.activeInstances = data.instances || [];

        // Determinar qual instância é a principal (porta 8085 ou a primeira)
        let mainInst = state.activeInstances.find(i => i.port === 8085)
            || state.activeInstances[0];

        // Garantir que currentActivePort esteja sincronizado
        if (mainInst && state.activeInstances.length > 0) {
            const portExists = state.activeInstances.some(i => i.port === state.currentActivePort);
            if (!portExists) {
                state.currentActivePort = mainInst.port;
            }
        }

        const hasInstances = state.activeInstances.length > 0;

        // --- Sincronizar config da instância principal no painel global ---
        if (mainInst) {
            const config = mainInst.config;
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

            state.currentRunningModelPath = mainInst.model_path;

            const apiLink = document.getElementById('api-link');
            if (apiLink) apiLink.innerText = `http://${window.fixedIp}:${mainInst.port}/v1`;
        } else {
            if (state.logStream) { 
                state.logStream.abort(); 
                state.logStream = null; 
                state.logStreamPort = null;
            }
            state.currentRunningModelPath = null;
        }

        // --- Renderizar cards de todas as instâncias ---
        renderActiveCards();

        // --- Status badge ---
        if (hasInstances) {
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
            badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE';
        } else {
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase';
            badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE';
        }

        // --- Log stream para a instância ativa ---
        if (mainInst && (!state.logStream || state.logStreamPort !== mainInst.port)) {
            startLogs(mainInst.port);
        }

        // --- Lógica de Auto-Balance / Recovery ---
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

        if (data.recovery && data.recovery.active) {
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-amber-500/50 text-amber-500 uppercase';
            badge.innerHTML = data.recovery.auto_balance 
                ? `<i class="fas fa-sync animate-spin mr-1"></i> AUTO BALANCE: ${(data.recovery.message || 'calibrando...').toUpperCase()}`
                : '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
        }

        // Atualizar lista de modelos para refletir quais estão rodando
        document.querySelectorAll('.model-item-container').forEach(el => {
            const m_js = normalizePath(el.dataset.path);
            const isRunning = state.activeInstances.some(i => normalizePath(i.model_path) === m_js);
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

function normalizePath(p) {
    return (p || '').replace(/\\/g, '/');
}

function renderActiveCards() {
    const container = document.getElementById('active-cards-container');
    if (!container) return;

    if (state.activeInstances.length === 0) {
        container.innerHTML = '';
        return;
    }

    const mainInst = state.activeInstances.find(i => i.port === 8085)
        || state.activeInstances[0];
    const count = state.activeInstances.length;
    let html = '';

    for (const inst of state.activeInstances) {
        const port = inst.port;
        const isActive = port === state.currentActivePort;
        const isMain = port === 8085;
        const modelName = inst.model || 'Modelo';
        const modelPath = inst.model_path || '';
        const config = inst.config || {};

        const titleText = count > 1
            ? `Motor de Computação ${state.activeInstances.indexOf(inst) + 1}/${count} (Porta ${port})`
            : 'Motor de Computação Primário';

        const cardBorder = isActive
            ? 'border-blue-500/50 shadow-lg shadow-blue-500/10'
            : 'border-blue-500/30 shadow-md shadow-blue-500/5';

        const uptimeText = inst.start_time
            ? (() => {
                const diff = Math.floor(Date.now() / 1000 - inst.start_time);
                const h = Math.floor(diff / 3600);
                const m = Math.floor((diff % 3600) / 60);
                const s = diff % 60;
                return `${h}h ${m}m ${s}s`;
            })()
            : 'Calculando...';

        const normalizedName = normalizePath(modelPath);

        html += `
        <div class="active-instance-card bg-gradient-to-r from-blue-900/40 to-slate-900/40 backdrop-blur-xl p-5 md:p-8 rounded-[2rem] border ${cardBorder} transition-all duration-500 ${isActive ? '' : 'opacity-80'}" data-port="${port}">
            <div class="flex flex-col lg:flex-row items-center justify-between gap-6 md:gap-8">
                <div class="flex items-center gap-4 md:gap-6 w-full lg:w-auto lg:flex-1">
                    <div class="w-12 h-12 md:w-16 md:h-16 rounded-2xl md:rounded-3xl bg-blue-600 flex items-center justify-center text-white shadow-2xl shadow-blue-500/40 shrink-0">
                        <i class="fas fa-robot text-lg md:text-2xl"></i>
                    </div>
                    <div class="min-w-0 flex-1">
                        <p class="text-blue-400 text-[9px] md:text-[10px] font-black uppercase tracking-[0.3em] mb-1 font-mono">${titleText}</p>
                        <h2 class="text-base md:text-xl font-bold text-white truncate">${modelName}</h2>
                        <div class="flex gap-4 mt-2">
                            <div class="flex items-center gap-2 text-[9px] md:text-[10px] font-mono text-slate-400">
                                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                                Ativo há: <span>${uptimeText}</span>
                            </div>
                            ${isMain ? '<span class="text-[8px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded border border-blue-500/20 font-black">PRINCIPAL</span>' : ''}
                        </div>
                    </div>
                </div>
                <div class="flex flex-wrap items-center gap-3 shrink-0 w-full lg:w-auto">
                    <a href="/ui/${port}/" target="_blank" rel="noopener noreferrer" class="px-4 md:px-6 py-2.5 md:py-3 btn-gradient text-white rounded-2xl text-[9px] md:text-xs font-black transition-all shadow-xl shadow-blue-600/30 active:scale-95 flex items-center justify-center gap-2 md:gap-3 uppercase tracking-widest whitespace-nowrap">
                        <i class="fas fa-comments text-[9px] md:text-sm"></i> <span class="hidden sm:inline">ABRIR CHAT</span><span class="sm:hidden">CHAT</span>
                    </a>
                    <button onclick="stopModel(${port})" class="px-4 md:px-6 py-2.5 md:py-3 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/30 rounded-2xl text-[9px] md:text-xs font-black transition-all active:scale-95 uppercase tracking-widest whitespace-nowrap">
                        ENCERRAR
                    </button>
                </div>
            </div>
        </div>`;
    }

    container.innerHTML = html;

    // Atualizar api-link com a porta da instância principal
    if (mainInst) {
        const apiLink = document.getElementById('api-link');
        if (apiLink) apiLink.innerText = `http://${window.fixedIp}:${mainInst.port}/v1`;
    }
}

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
