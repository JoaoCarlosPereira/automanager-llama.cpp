import { state } from './state.js';
import { apiFetch, sessionExpiredHandled } from './auth.js';
import {
    applyGpuWeightsToUI, getContextSize, setContextSize,
    hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,
    updateThinkingBadge,
} from './gpu.js';

export async function updateStatus() {
    try {
        const res = await apiFetch('/status');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const badge = document.getElementById('status-badge');
        const card = document.getElementById('active-card');

        if (data.running && data.config && data.config.path) {
            window.modelConfigs[data.config.path] = window.modelConfigs[data.config.path] || {};
            Object.assign(window.modelConfigs[data.config.path], data.config);
            if (state.currentSelectedModel === data.config.path) {
                updateAutoBalanceProfileBadge(data.config.auto_balance_profile);
            }
        }

        const autoBalancing = !!(data.recovery && data.recovery.active && data.recovery.auto_balance);
        syncAutoBalanceCancelButton(autoBalancing);
        const weightsToApply = autoBalancing && data.recovery.gpu_weights
            ? data.recovery.gpu_weights
            : (data.config && data.config.gpu_weights ? data.config.gpu_weights : null);
        const maySyncWeights = weightsToApply && (
            autoBalancing || !data.recovery || !data.recovery.active
        );
        if (maySyncWeights) {
            applyGpuWeightsToUI(weightsToApply, autoBalancing);
            if (data.running && !state.currentSelectedModel && data.config) {
                if (data.config.context_size) setContextSize(data.config.context_size);
                if (data.config.parallel_slots) document.getElementById('parallel-slots').value = data.config.parallel_slots;
                if (data.config.batch_size) document.getElementById('batch-size').value = data.config.batch_size;
                if (data.config.mmproj_path !== undefined) document.getElementById('mmproj-path').value = data.config.mmproj_path || "";
                const thinkingToggle = document.getElementById('thinking-toggle');
                if (thinkingToggle && data.config.thinking_enabled !== undefined) {
                    thinkingToggle.checked = !!data.config.thinking_enabled;
                    updateThinkingBadge(!!data.config.thinking_enabled);
                }
            }
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
                const finalWeights = data.recovery.gpu_weights
                    || (data.config && data.config.gpu_weights);
                if (finalWeights) {
                    applyGpuWeightsToUI(finalWeights, false);
                }
                if (state.currentSelectedModel) {
                    window.modelConfigs[state.currentSelectedModel] =
                        window.modelConfigs[state.currentSelectedModel] || {};
                    Object.assign(window.modelConfigs[state.currentSelectedModel], {
                        auto_balance: false,
                        auto_balance_profile: true,
                        gpu_weights: finalWeights,
                    });
                }
                updateAutoBalanceProfileBadge(true);
                window.updateModels();
            } else if (data.recovery.hardware_capacity_exceeded) {
                showAutoBalanceCapacityAlert(data.recovery);
                alert(data.recovery.message || 'Modelo além da capacidade do hardware.');
                if (state.currentSelectedModel) {
                    window.modelConfigs[state.currentSelectedModel] =
                        window.modelConfigs[state.currentSelectedModel] || {};
                    window.modelConfigs[state.currentSelectedModel].hardware_incapable = true;
                    window.modelConfigs[state.currentSelectedModel].hardware_incapable_message =
                        data.recovery.message;
                }
                window.updateModels();
            }
            return;
        }

        if (data.recovery && data.recovery.failed) {
            state.autoBalancePending = false;
            syncAutoBalanceCancelButton(false);
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-red-500/50 text-red-500 uppercase';
            if (data.recovery.hardware_capacity_exceeded) {
                showAutoBalanceCapacityAlert(data.recovery);
                badge.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i> HARDWARE INSUFICIENTE';
                window.updateModels();
            } else {
                badge.innerHTML = `<i class="fas fa-exclamation-triangle mr-1"></i> FALHA: ${(data.recovery.message || 'erro').toUpperCase()}`;
            }
            return;
        }

        if (data.recovery && data.recovery.active) {
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-amber-500/50 text-amber-500 uppercase';
            if (data.recovery.auto_balance) {
                const msg = data.recovery.message || 'calibrando GPUs...';
                badge.innerHTML = `<i class="fas fa-sync animate-spin mr-1"></i> AUTO BALANCE: ${msg.toUpperCase()}`;
            } else {
                badge.innerHTML = '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
            }
            return;
        }

        if (data.running) {
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-emerald-500/30 text-emerald-500 uppercase glow-online';
            badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE';
            card.classList.remove('hidden');
            document.getElementById('active-model-name').innerText = data.model;
            if (!state.logStream) startLogs();
            updateUptime(data.start_time);
            state.currentRunningModelPath = data.model_path;
            if (!state.currentSelectedModel && state.currentRunningModelPath) {
                state.currentSelectedModel = state.currentRunningModelPath.replace(/\\\\\\\\/g, '/');
            }
            const chatLink = document.getElementById('chat-link');
            if (chatLink) {
                chatLink.classList.remove('pointer-events-none', 'opacity-40');
                chatLink.setAttribute('aria-disabled', 'false');
            }
        } else {
            state.startTime = null;
            badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase';
            badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE';
            card.classList.add('hidden');
            if (state.logStream) { state.logStream.abort(); state.logStream = null; }
            state.currentRunningModelPath = null;
            const chatLinkOff = document.getElementById('chat-link');
            if (chatLinkOff) {
                chatLinkOff.classList.add('pointer-events-none', 'opacity-40');
                chatLinkOff.setAttribute('aria-disabled', 'true');
            }
        }

        document.querySelectorAll('.model-item-container').forEach(el => {
            const m_js = el.dataset.path;
            const actionBtnContainer = el.querySelector('.action-btn-container');
            const renameBtn = el.querySelector('.rename-btn');
            const deleteBtn = el.querySelector('.delete-btn');
            const normalizedM = m_js.replace(/\\\\\\\\/g, '/');
            const normalizedR = state.currentRunningModelPath ? state.currentRunningModelPath.replace(/\\\\\\\\/g, '/') : null;
            const isRunning = normalizedR && normalizedM === normalizedR;

            if (isRunning) {
                el.classList.add('running-now');
                if (renameBtn) renameBtn.classList.add('hidden');
                if (deleteBtn) deleteBtn.classList.add('hidden');
            } else {
                el.classList.remove('running-now');
                if (renameBtn) renameBtn.classList.remove('hidden');
                if (deleteBtn) deleteBtn.classList.remove('hidden');
            }
            if (state.currentSelectedModel === m_js) el.classList.add('active-selection');
            else el.classList.remove('active-selection');

            const newButtonsHtml = window.getModelButtonsHtml(m_js, el.id, isRunning);
            if (actionBtnContainer.innerHTML.trim() !== newButtonsHtml.trim()) {
                actionBtnContainer.innerHTML = newButtonsHtml;
            }
        });
    } catch (e) { console.error("updateStatus error:", e); }
}

export function ensureStatusPolling(fast) {
    const ms = fast ? 1000 : 3000;
    if (state.statusPollIntervalMs === ms && state.statusPollTimer) return;
    state.statusPollIntervalMs = ms;
    if (state.statusPollTimer) clearInterval(state.statusPollTimer);
    state.statusPollTimer = setInterval(updateStatus, ms);
}

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatSpeed(bps) {
    if (!bps || bps <= 0) return '--';
    return formatBytes(bps) + '/s';
}

export async function updateDownloads() {
    try {
        const res = await apiFetch('/downloads');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const container = document.getElementById('download-status');
        const entries = Object.entries(data);
        if (entries.length === 0) { container.innerHTML = ''; return; }

        const hasCompleted = entries.some(([_, d]) => d.status === 'completed');
        const hasActive = entries.some(([_, d]) => d.status === 'downloading');

        let html = '';
        for (const [id, d] of entries) {
            const statusClass = d.status === 'completed'
                ? 'bg-emerald-500/10 text-emerald-500'
                : d.status === 'failed'
                    ? 'bg-red-500/10 text-red-500'
                    : 'bg-blue-500/10 text-blue-500';
            const statusLabel = d.status === 'completed'
                ? 'Concluído'
                : d.status === 'failed'
                    ? 'Falhou'
                    : 'Baixando';

            const progressColor = d.status === 'failed'
                ? 'bg-red-500'
                : d.status === 'completed'
                    ? 'bg-emerald-500'
                    : 'bg-blue-500';
            const progressShadow = d.status === 'downloading'
                ? 'shadow-[0_0_10px_rgba(37,99,235,0.5)]'
                : '';

            html += `
                <div class="p-4 md:p-5 bg-slate-900 border border-slate-800 rounded-2xl">
                    <div class="flex justify-between items-center mb-3 md:mb-4">
                        <p class="text-xs md:text-sm font-bold truncate flex-1 mr-3 md:mr-4 text-slate-300 font-mono" title="${d.filename}">${d.filename}</p>
                        <span class="text-[8px] md:text-[10px] font-black uppercase px-2 md:px-3 py-0.5 md:py-1 rounded ${statusClass}">
                            ${statusLabel}
                        </span>
                    </div>
                    <div class="flex justify-between items-center mb-2 text-[10px] md:text-xs font-mono">
                        <span class="text-slate-500">${d.progress.toFixed(1)}%</span>
                        <span class="text-blue-400">${d.status === 'downloading' ? formatSpeed(d.speed_bps) : (d.total_bytes ? formatBytes(d.total_bytes) : '--')}/${d.total_bytes ? formatBytes(d.downloaded_bytes || d.total_bytes) : '--'}</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full ${progressColor} ${progressShadow} transition-all duration-500" style="width: ${d.progress}%"></div>
                    </div>
                </div>
            `;
        }

        if (hasCompleted) {
            html += `
                <button onclick="clearCompletedDownloads()" class="mt-4 w-full py-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-blue-500/40 text-slate-400 hover:text-blue-400 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2">
                    <i class="fas fa-broom text-[10px]"></i> Limpar Downloads Concluídos
                </button>
            `;
        }

        container.innerHTML = html;

        if (hasCompleted) window.updateModels();
    } catch (e) {}
}

export async function clearCompletedDownloads() {
    if (!confirm('Remover downloads concluídos da exibição?')) return;
    try {
        const res = await apiFetch('/downloads/clear', { method: 'POST' });
        if (res.ok) window.updateDownloads();
    } catch (e) {}
}

export function startDashboardPolling() {
    stopDashboardPolling();
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

export async function startLogs() {
    if (state.logStream) state.logStream.abort();
    state.logStream = new AbortController();
    const box = document.getElementById('log-box');
    box.innerHTML = '';
    try {
        const response = await fetch('/logs', { signal: state.logStream.signal });
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
    if (!confirm("Deseja realmente gerar uma nova chave de API?")) return;
    try {
        const res = await fetch('/api/key/renew', { method: 'POST' });
        const data = await res.json();
        document.getElementById('api-token').innerText = data.key;
        alert("Nova chave gerada!");
    } catch (e) {
        alert("Erro ao renovar token.");
    }
}

export async function updateMetrics() {
    try {
        const res = await apiFetch('/metrics');
        if (sessionExpiredHandled || !res.ok) return;
        const data = await res.json();
        const metricsPanel = document.getElementById('metrics-panel');
        if (metricsPanel) {
            if (!state.currentRunningModelPath) metricsPanel.classList.add('metric-dimmed');
            else metricsPanel.classList.remove('metric-dimmed');
        }
        document.getElementById('cpu-val').innerText = data.cpu + '%';
        document.getElementById('cpu-bar').style.width = data.cpu + '%';
        document.getElementById('ram-val').innerText = data.ram + '%';
        document.getElementById('ram-bar').style.width = data.ram + '%';

        const cpuRow = document.querySelector('.cpu-row');
        if (cpuRow) {
            const cpuUtil = data.cpu ?? 0;
            const ramUsed = data.ram_used_mb ?? 0;
            const ramTotal = data.ram_total_mb ?? 0;
            const ramPct = ramTotal > 0
                ? Math.round((ramUsed / ramTotal) * 1000) / 10
                : (data.ram ?? 0);

            const utilVal = cpuRow.querySelector('.cpu-util-val');
            const utilBar = cpuRow.querySelector('.cpu-util-bar');
            if (utilVal) utilVal.innerText = `${cpuUtil}%`;
            if (utilBar) utilBar.style.width = `${cpuUtil}%`;

            const ramVal = cpuRow.querySelector('.cpu-ram-val');
            if (ramVal) ramVal.innerText = `${ramUsed} / ${ramTotal} MB`;

            const ramText = cpuRow.querySelector('.cpu-ram-text');
            const ramBar = cpuRow.querySelector('.cpu-ram-bar');
            if (ramText) ramText.innerText = `${ramUsed} / ${ramTotal} MB`;
            if (ramBar) ramBar.style.width = `${ramPct}%`;
        }

        data.gpus.forEach(g => {
            const row = document.querySelector(`.gpu-row[data-index="${g.index}"]`);
            if (row) {
                row.querySelector('.gpu-util-val').innerText = g.util + '%';
                row.querySelector('.gpu-util-bar').style.width = g.util + '%';
                row.querySelector('.gpu-temp-val').innerText = (g.temp || '--') + '°C';
                row.querySelector('.gpu-power-val').innerText = (g.power || '--') + 'W';
                row.querySelector('.gpu-vram-text').innerText = `${g.mem_used} / ${g.mem_total} MB`;
                row.querySelector('.gpu-vram-bar').style.width = g.vram_pct + '%';
            }
        });
    } catch (e) {}
}

export function stopDashboardPolling() {
    if (state.statusPollTimer) {
        clearInterval(state.statusPollTimer);
        state.statusPollTimer = null;
    }
    if (state.metricsTimer) {
        clearInterval(state.metricsTimer);
        state.metricsTimer = null;
    }
    if (state.downloadsTimer) {
        clearInterval(state.downloadsTimer);
        state.downloadsTimer = null;
    }
    if (state.modelsTimer) {
        clearInterval(state.modelsTimer);
        state.modelsTimer = null;
    }
    if (state.logStream) {
        state.logStream.abort();
        state.logStream = null;
    }
    syncAutoBalanceCancelButton(false);
}
