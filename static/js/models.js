import { state } from './state.js';
import { apiFetch, sessionExpiredHandled } from './auth.js';
import {
    getContextSize, setContextSize, resetToDefaults, applyGpuWeightsToUI,
    updateTotal, hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,
    isModelHardwareIncapable, modelIncapableBadgeHtml, modelIncapableRowClass,
    bindGpuManualListeners, syncContextSizeCustomVisibility,
    updateThinkingBadge, updateMtpBadge, validateDeviceWeights,
    collectDeviceWeightsFromUI,
} from './gpu.js';
import { startLogs, updateUptime } from './metrics.js';

export function initDashboard() {
    bindGpuManualListeners();
    syncContextSizeCustomVisibility();

    const thinkingToggle = document.getElementById('thinking-toggle');
    if (thinkingToggle) {
        thinkingToggle.addEventListener('change', () => {
            updateThinkingBadge(thinkingToggle.checked);
        });
    }

    const mtpToggle = document.getElementById('mtp-toggle');
    if (mtpToggle) {
        mtpToggle.addEventListener('change', () => {
            updateMtpBadge(mtpToggle.checked);
        });
    }

    window.updateStatus();
    window.updateMetrics();
    window.updateDownloads();
    window.updateModels();
    updateTotal();
}

export function getModelButtonsHtml(path, elementId, isRunning) {
    if (isRunning) {
        return `<div class="flex items-center gap-3">
            <a href="http://${window.fixedIp}:8085/" target="_blank" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-[9px] font-black rounded-xl flex items-center gap-2 uppercase tracking-widest shadow-lg shadow-blue-600/20 transition-all whitespace-nowrap">
                <i class="fas fa-comments text-[8px]"></i> ABRIR INTERFACE
            </a>
            <button onclick="stopModel()" class="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 text-[9px] font-black rounded-xl transition-all uppercase tracking-widest whitespace-nowrap">
                ENCERRAR
            </button>
            <div class="flex items-center gap-2 text-[9px] font-mono text-emerald-400 bg-emerald-500/5 px-3 py-2 rounded-xl border border-emerald-500/10">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span id="uptime-val">--</span>
            </div>
        </div>`;
    }
    return `<button onclick="startModel('${path}', '${elementId}')" class="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-2xl active:scale-95 flex items-center gap-3 uppercase tracking-widest shadow-xl shadow-blue-600/20 transition-all">
        <i class="fas fa-play text-[9px]"></i> <span class="hidden sm:inline">CARREGAR</span><span class="sm:hidden">LOAD</span>
    </button>`;
}

export function selectModel(path, elementId) {
    state.currentSelectedModel = path;
    document.querySelectorAll('.model-item-container').forEach(el => {
        el.classList.remove('active-selection');
    });
    const selectedEl = document.getElementById(elementId);
    if (selectedEl) selectedEl.classList.add('active-selection');
    if (window.modelConfigs[path]) {
        applyModelConfig(path);
    } else {
        resetToDefaults();
    }
}

export function applyModelConfig(path) {
    const cfg = window.modelConfigs[path];
    if (!cfg) return;
    if (cfg.context_size) setContextSize(cfg.context_size);
    if (cfg.parallel_slots) document.getElementById('parallel-slots').value = cfg.parallel_slots;
    if (cfg.batch_size) document.getElementById('batch-size').value = cfg.batch_size;
    if (cfg.split_mode) document.getElementById('split-mode').value = cfg.split_mode;
    if (cfg.mmproj_path !== undefined) {
        const select = document.getElementById('mmproj-path');
        let found = false;
        for (let i = 0; i < select.options.length; i++) {
            if (select.options[i].value === cfg.mmproj_path) {
                select.value = cfg.mmproj_path;
                found = true;
                break;
            }
        }
        if (!found && cfg.mmproj_path) {
            const opt = document.createElement('option');
            opt.value = cfg.mmproj_path;
            opt.text = cfg.mmproj_path.split('/').pop() + " (Salvo)";
            select.add(opt);
            select.value = cfg.mmproj_path;
        } else if (!cfg.mmproj_path) {
            select.value = "";
        }
    }
    const abToggle = document.getElementById('auto-balance-toggle');
    if (abToggle) abToggle.checked = !!cfg.auto_balance;
    updateAutoBalanceProfileBadge(cfg.auto_balance_profile);
    const thinkingToggle = document.getElementById('thinking-toggle');
    if (thinkingToggle) {
        thinkingToggle.checked = cfg.thinking_enabled !== false;
        updateThinkingBadge(cfg.thinking_enabled !== false);
    }
    const mtpToggle = document.getElementById('mtp-toggle');
    if (mtpToggle) {
        mtpToggle.checked = !!cfg.mtp_enabled;
        updateMtpBadge(!!cfg.mtp_enabled);
    }
    const mtpDraftTokens = document.getElementById('mtp-draft-tokens');
    if (mtpDraftTokens && cfg.mtp_draft_tokens) {
        mtpDraftTokens.value = String(cfg.mtp_draft_tokens);
    }
    state.manualGpuOverride = false;
    if (cfg.gpu_weights) {
        applyGpuWeightsToUI(cfg.gpu_weights, false);
    }
    const nameEl = document.querySelector(`[data-path="${path}"] .model-name`);
    if (nameEl) {
        nameEl.classList.add('text-emerald-400');
        setTimeout(() => { nameEl.classList.remove('text-emerald-400'); }, 1000);
    }
}

export async function setDefaultModel(checkbox, path) {
    if (checkbox.checked) document.querySelectorAll('.model-default-checkbox').forEach(cb => {
        if (cb !== checkbox) cb.checked = false;
    });
    try {
        await fetch('/set_default', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: checkbox.checked ? path : null}),
        });
    } catch (e) {
        alert("Erro ao salvar configuracao.");
    }
}

export async function downloadModel() {
    const url = document.getElementById('download-url').value.trim();
    if (!url) return;
    try {
        const res = await fetch('/downloads', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url}),
        });
        if (res.ok) document.getElementById('download-url').value = '';
        window.updateDownloads();
    } catch (e) {}
}

export async function updateModels() {
    try {
        const [res, cfgRes] = await Promise.all([
            apiFetch('/models'),
            apiFetch('/config'),
        ]);
        if (sessionExpiredHandled || !res.ok || !cfgRes.ok) return;
        const data = await res.json();
        const cfg = await cfgRes.json();
        document.getElementById('model-count').innerText = `${data.models.length} UNIDADES`;
        const oldContainer = document.getElementById('model-list-container');
        const newHtml = data.models.map(m => {
            const m_js = m.path.replace(/\\\\\\\\/g, '/');
            if (m.last_config) window.modelConfigs[m.path] = m.last_config;
            const incapable = !!(m.last_config && m.last_config.hardware_incapable);
            const hasConfigClass = m.last_config && !incapable ? 'text-blue-400' : 'text-slate-100';
            const incapableBadge = modelIncapableBadgeHtml(incapable);
            const incapableRow = modelIncapableRowClass(incapable);
            const historyIcon = m.last_config && !incapable ? '<i class="fas fa-history text-[8px] text-blue-500/50" title="Configuração salva disponível"></i>' : '';
            const isRunning = state.currentRunningModelPath && m_js === state.currentRunningModelPath.replace(/\\\\\\\\/g, '/');
            const isActive = state.currentSelectedModel === m_js ? 'active-selection' : '';
            const runningClass = isRunning ? 'running-now' : '';
            const hashId = m.id;
            const buttonsHtml = getModelButtonsHtml(m_js, hashId, isRunning);
            return `<div id="${hashId}" class="model-item-container group flex items-center justify-between p-4 md:p-5 mb-3 md:mb-4 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg ${isActive} ${runningClass} ${incapableRow}" data-path="${m_js}" data-hardware-incapable="${incapable}">
                <div class="flex-1 min-w-0 mr-4 md:mr-6 cursor-pointer" onclick="selectModel('${m_js}', '${hashId}')">
                    <div class="flex items-center gap-2 md:gap-3 mb-1 md:mb-2 flex-wrap">
                        <i class="fas fa-cube text-blue-400 text-[10px] md:text-xs"></i>
                        <p class="model-name text-sm md:text-base font-bold ${hasConfigClass} break-all line-clamp-2" title="${m.name}">${m.name}</p>
                        ${incapableBadge}
                        ${historyIcon}
                    </div>
                    <p class="text-[9px] md:text-xs text-slate-500 truncate uppercase tracking-tighter font-mono">${m.dir}</p>
                </div>
                <div class="flex items-center gap-3 md:gap-6">
                    <div class="flex items-center gap-1">
                        <button onclick="renameModel('${m_js}')" class="rename-btn w-10 h-10 flex items-center justify-center rounded-xl hover:bg-blue-500/20 text-slate-600 hover:text-blue-500 transition-all ${isRunning ? 'hidden' : ''}" title="Renomear Modelo">
                            <i class="fas fa-edit text-[10px] md:text-xs"></i>
                        </button>
                        <button onclick="deleteModel('${m_js}')" class="delete-btn w-10 h-10 flex items-center justify-center rounded-xl hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all ${isRunning ? 'hidden' : ''}" title="Excluir Modelo">
                            <i class="fas fa-trash-alt text-[10px] md:text-xs"></i>
                        </button>
                    </div>
                    <div class="flex flex-col items-center gap-1 md:gap-1.5">
                        <span class="text-[8px] md:text-[10px] font-black text-slate-600 uppercase tracking-tighter">Padrão</span>
                        <input type="checkbox" class="model-default-checkbox w-4 h-4 md:w-5 md:h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" ${m.path === cfg.default_model ? 'checked' : ''} onclick="setDefaultModel(this, '${m_js}')">
                    </div>
                    <div class="action-btn-container">${buttonsHtml}</div>
                </div>
            </div>`;
        }).join('');
        if (oldContainer.innerHTML !== newHtml) oldContainer.innerHTML = newHtml;

        const projSelect = document.getElementById('mmproj-path');
        const currentVal = projSelect.value;
        let projHtml = '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>';
        data.projectors.forEach(p => {
            projHtml += `<option value="${p.path}" class="bg-slate-900">${p.name}</option>`;
        });
        if (projSelect.innerHTML.trim() !== projHtml.trim()) {
            projSelect.innerHTML = projHtml;
            projSelect.value = currentVal;
            if (projSelect.value !== currentVal) projSelect.value = "";
        }
    } catch (e) {}
}

export async function renameModel(path) {
    const currentName = path.split('/').pop().replace('.gguf', '');
    const newName = prompt("Digite o novo nome para o modelo:", currentName);
    if (!newName || newName === currentName) return;
    try {
        const res = await fetch('/rename', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path, new_name: newName}),
        });
        if (res.ok) updateModels();
        else { const err = await res.json(); alert("Erro ao renomear: " + (err.detail || "Erro desconhecido")); }
    } catch (e) {
        alert("Erro de rede ao renomear modelo.");
    }
}

export async function deleteModel(path) {
    if (!confirm("TEM CERTEZA QUE DESEJA EXCLUIR ESTE MODELO DO DISCO?\\nEsta ação é irreversível.")) return;
    try {
        const res = await fetch('/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path}),
        });
        if (res.ok) updateModels();
        else { const err = await res.json(); alert("Erro ao excluir: " + (err.detail || "Erro desconhecido")); }
    } catch (e) {
        alert("Erro de rede ao excluir modelo.");
    }
}

export async function startModel(path, elementId) {
    if (isModelHardwareIncapable(path)) {
        const cfg = window.modelConfigs[path] || {};
        const detail = cfg.hardware_incapable_message
            ? `\\n\\n${cfg.hardware_incapable_message}`
            : '';
        if (!confirm(
            'Este modelo está marcado como INCOMPATÍVEL com o hardware após auto balance.'
            + detail
            + '\\n\\nDeseja tentar carregar mesmo assim?'
        )) return;
    }
    if (state.currentSelectedModel !== path) {
        selectModel(path, elementId);
        await new Promise(r => setTimeout(r, 100));
    }
    document.getElementById('log-box').innerHTML = '';
    const weights = collectDeviceWeightsFromUI();
    const weightValidation = validateDeviceWeights(weights);
    if (!weightValidation.ok) {
        return alert(weightValidation.message);
    }
    if (!weights.some(w => w.is_main)) return alert("DEFINA A GPU PRINCIPAL (coluna Principal)");
    const mmprojPath = document.getElementById('mmproj-path').value;
    const splitMode = document.getElementById('split-mode').value;
    const parallelSlots = Math.max(1, Math.min(64, parseInt(document.getElementById('parallel-slots').value) || window.__constants.DEFAULT_PARALLEL_SLOTS));
    const batchSize = parseInt(document.getElementById('batch-size').value, 10) || window.__constants.DEFAULT_BATCH_SIZE;
    const autoBalance = document.getElementById('auto-balance-toggle').checked;
    const thinkingToggle = document.getElementById('thinking-toggle');
    const thinkingEnabled = thinkingToggle ? thinkingToggle.checked : true;
    const mtpToggleEl = document.getElementById('mtp-toggle');
    const mtpEnabled = mtpToggleEl ? mtpToggleEl.checked : false;
    const mtpDraftRaw = parseInt(document.getElementById('mtp-draft-tokens')?.value, 10);
    const mtpDraftTokens = Math.max(1, Math.min(6, Number.isFinite(mtpDraftRaw) ? mtpDraftRaw : 3));
    document.getElementById('parallel-slots').value = parallelSlots;
    document.getElementById('status-badge').innerHTML = autoBalance
        ? '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> AUTO BALANCE...'
        : '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> INICIALIZANDO...';
    const contextSize = getContextSize();
    if (contextSize === null) {
        alert('Informe um contexto válido em K (mínimo 1). Ex.: 100 = 100K tokens por slot.');
        return;
    }
    try {
        const res = await apiFetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                path,
                mmproj_path: mmprojPath || null,
                gpu_weights: weights,
                context_size: contextSize,
                parallel_slots: parallelSlots,
                batch_size: batchSize,
                split_mode: splitMode,
                auto_balance: autoBalance,
                manual_gpu_override: autoBalance ? false : state.manualGpuOverride,
                thinking_enabled: thinkingEnabled,
                mtp_enabled: mtpEnabled,
                mtp_draft_tokens: mtpDraftTokens,
            }),
        });
        if (!res.ok) {
            if (sessionExpiredHandled) return;
            const err = await res.json();
            alert("Erro ao iniciar: " + (err.detail || "Erro desconhecido"));
            return;
        }
        const startData = await res.json();
        if (startData.probing) {
            state.manualGpuOverride = false;
            state.autoBalancePending = true;
            syncAutoBalanceCancelButton(true);
            hideAutoBalanceCapacityAlert();
        } else if (!autoBalance && state.manualGpuOverride) {
            state.manualGpuOverride = false;
            if (window.modelConfigs[path]) {
                window.modelConfigs[path].auto_balance_profile = false;
            }
            updateAutoBalanceProfileBadge(false);
        }
    } catch (e) {
        alert("Erro ao iniciar modelo.");
    }
    setTimeout(window.updateStatus, 2000);
}

export async function stopModel() {
    if (confirm("ENCERRAR PROCESSO?")) {
        const res = await apiFetch('/stop', {method: 'POST'});
        if (!sessionExpiredHandled && res.ok) setTimeout(window.updateStatus, 1000);
    }
}
