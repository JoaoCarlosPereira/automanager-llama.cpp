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
import { startLogs } from './metrics.js';
import { checkForUpdates } from './version.js';

// --- TAB MANAGEMENT ---

export function toggleSidebar(forceOpen = null) {
    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('main-content');
    const isCollapsed = sidebar.classList.contains('collapsed');
    
    const shouldOpen = forceOpen !== null ? forceOpen : isCollapsed;
    
    if (shouldOpen) {
        sidebar.classList.remove('collapsed');
        sidebar.classList.add('open');
        main.classList.remove('full');
    } else {
        sidebar.classList.add('collapsed');
        sidebar.classList.remove('open');
        main.classList.add('full');
    }
}

export function createModelTab(path, name, id) {
    const m_js = path.replace(/\\/g, '/');
    const tabId = `tab-${id}`;
    
    // If tab exists, just switch to it
    if (document.getElementById(tabId)) {
        switchTab(tabId);
        return;
    }

    // 1. Create Tab Button
    const tabBar = document.getElementById('tab-bar');
    const btn = document.createElement('button');
    btn.id = `btn-${tabId}`;
    btn.className = 'tab-btn px-4 h-full flex items-center gap-3 text-[10px] font-bold text-slate-500 border-b-2 border-transparent hover:text-slate-300 transition-all group relative min-w-[120px] max-w-[240px]';
    btn.onclick = () => switchTab(tabId);
    btn.innerHTML = `
        <div class="tab-status-dot w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0 transition-all duration-500"></div>
        <span class="truncate flex-1 text-left">${name}</span>
        <span onclick="event.stopPropagation(); closeTab('${tabId}')" class="tab-close-btn w-4 h-4 flex items-center justify-center rounded hover:bg-red-500/20 hover:text-red-500 text-slate-600 transition-all">
            <i class="fas fa-times text-[8px]"></i>
        </span>
    `;
    tabBar.appendChild(btn);

    // 2. Create Tab Content from Template
    const template = document.getElementById('model-tab-template');
    const content = template.content.cloneNode(true);
    const tabDiv = content.querySelector('.tab-content');
    tabDiv.id = tabId;
    tabDiv.dataset.path = m_js;
    
    // Customize inner elements
    tabDiv.querySelector('.model-tab-name').innerText = name;
    tabDiv.querySelector('.model-tab-path').innerText = path;
    
    // Unique name for radio group to avoid crosstalk
    tabDiv.querySelectorAll('.gpu-main-radio').forEach(r => r.name = `main-gpu-${tabId}`);
    
    document.getElementById('tabs-container').appendChild(tabDiv);

    // 3. Register state
    state.activeTabs.push({id: tabId, path: m_js, name});
    
    // 4. Initial Switch
    switchTab(tabId);
    
    // 5. Load Configs
    if (window.modelConfigs[m_js]) {
        applyModelConfig(m_js, tabId);
    }
    
    // 6. Bind Listeners for this tab
    bindTabListeners(tabId);
}

function bindTabListeners(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    
    const path = tab.dataset.path;
    
    tab.querySelector('.tab-thinking-toggle')?.addEventListener('change', (e) => persistThinkingEnabled(path, e.target.checked));
    tab.querySelector('.tab-context-size')?.addEventListener('change', () => syncContextSizeCustomVisibility(tabId));
    tab.querySelector('.tab-context-size-custom')?.addEventListener('input', () => updateTotal(tabId));
    tab.querySelector('.tab-auto-balance-toggle')?.addEventListener('change', () => syncAutoBalanceCancelButton(false, tabId));
    
    // Weights and Pins
    tab.querySelectorAll('.gpu-weight, .cpu-weight').forEach(el => {
        el.oninput = () => balanceWeights(el, tabId);
    });
    
    tab.querySelectorAll('.gpu-pin, .cpu-pin').forEach(el => {
        el.onchange = () => updateTotal(tabId);
    });
    
    tab.querySelectorAll('.gpu-checkbox, .cpu-checkbox').forEach(el => {
        el.onchange = () => balanceWeights(null, tabId);
    });
    
    tab.querySelector('.tab-reset-defaults-btn').onclick = () => resetToDefaults(tabId);
    tab.querySelector('.tab-clear-logs-btn').onclick = () => {
        tab.querySelector('.tab-log-box').innerHTML = '';
        state.modelLogs[path] = '';
    };

    // New Smart Calibrate Button
    tab.querySelector('.tab-smart-calibrate-btn').onclick = () => startSmartCalibration(path, tabId);
    
    // Apply/Discard Proposed Config
    tab.querySelector('.tab-apply-config-btn').onclick = () => applyProposedConfig(path, tabId);
    tab.querySelector('.tab-discard-config-btn').onclick = () => hideProposedConfig(tabId);

    // Handle Pins visual state
    tab.querySelectorAll('input[class*="tab-pin-"]').forEach(pin => {
        pin.addEventListener('change', () => {
            const icon = pin.nextElementSibling;
            if (pin.checked) {
                icon.classList.remove('text-slate-700');
                icon.classList.add('text-blue-500');
            } else {
                icon.classList.remove('text-blue-500');
                icon.classList.add('text-slate-700');
            }
        });
    });

    updateTotal(tabId);
}

export async function startSmartCalibration(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    const pinnedFields = {
        context_size: tab.querySelector('.tab-pin-context').checked,
        parallel_slots: tab.querySelector('.tab-pin-slots').checked,
        batch_size: tab.querySelector('.tab-pin-batch').checked,
        ubatch_size: tab.querySelector('.tab-pin-ubatch').checked,
        cache_type: tab.querySelector('.tab-pin-cache').checked,
        threads: tab.querySelector('.tab-pin-threads').checked,
    };

    const currentValues = {
        context_size: getContextSize(tabId),
        parallel_slots: parseInt(tab.querySelector('.tab-parallel-slots').value, 10),
        batch_size: parseInt(tab.querySelector('.tab-batch-size').value, 10),
        ubatch_size: parseInt(tab.querySelector('.tab-ubatch-size').value, 10),
        cache_type_k: tab.querySelector('.tab-cache-type-k').value,
        cache_type_v: tab.querySelector('.tab-cache-type-v').value,
        threads: parseInt(tab.querySelector('.tab-threads').value, 10) || 0,
        threads_batch: parseInt(tab.querySelector('.tab-threads-batch').value, 10) || 0,
    };

    const statusBadge = tab.querySelector('.tab-status-badge');
    statusBadge.innerHTML = '<i class="fas fa-magic animate-spin mr-2"></i> OTIMIZANDO...';
    
    try {
        const res = await apiFetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                path,
                gpu_weights: collectDeviceWeightsFromUI(tabId),
                context_size: currentValues.context_size,
                parallel_slots: currentValues.parallel_slots,
                batch_size: currentValues.batch_size,
                ubatch_size: currentValues.ubatch_size,
                cache_type_k: currentValues.cache_type_k,
                cache_type_v: currentValues.cache_type_v,
                threads: currentValues.threads,
                threads_batch: currentValues.threads_batch,
                auto_balance: true,
                smart_calibration: true,
                pinned_fields: pinnedFields,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            alert("Erro na calibração: " + (err.detail || "Falha"));
            window.updateStatus();
            return;
        }

        const data = await res.json();
        if (data.probing) {
            state.autoBalancePending = true;
            // The polling will detect the end of probing and we'll handle the proposal then
        }
    } catch (e) {
        alert("Erro de rede.");
        window.updateStatus();
    }
}

export function showProposedConfig(tabId, proposal) {
    const tab = document.getElementById(tabId);
    if (!tab || !proposal) return;

    const area = tab.querySelector('.tab-proposed-config');
    const details = tab.querySelector('.tab-proposed-details');
    
    // Store proposal in tab element for Apply button
    tab.dataset.proposal = JSON.stringify(proposal);

    const formatDiff = (label, oldVal, newVal) => {
        const changed = oldVal !== newVal;
        return `
            <div class="flex flex-col p-2 bg-slate-900/50 rounded-lg border border-slate-800">
                <span class="text-[7px] text-slate-500 uppercase">${label}</span>
                <div class="flex items-center gap-2 mt-1">
                    <span class="line-through opacity-40">${oldVal}</span>
                    <i class="fas fa-arrow-right text-[7px] text-blue-500"></i>
                    <span class="font-bold ${changed ? 'text-emerald-400' : 'text-slate-300'}">${newVal}</span>
                </div>
            </div>
        `;
    };

    const oldCfg = window.modelConfigs[tab.dataset.path] || {};
    
    details.innerHTML = `
        ${formatDiff('Contexto', (oldCfg.context_size || 0)/1024 + 'K', (proposal.context_size || 0)/1024 + 'K')}
        ${formatDiff('Slots', oldCfg.parallel_slots || 1, proposal.parallel_slots || 1)}
        ${formatDiff('Batch', oldCfg.batch_size || 2048, proposal.batch_size || 2048)}
        ${formatDiff('Cache', oldCfg.cache_type_k || 'f16', proposal.cache_type_k || 'f16')}
    `;

    area.classList.remove('hidden');
}

export function hideProposedConfig(tabId) {
    const tab = document.getElementById(tabId);
    if (tab) tab.querySelector('.tab-proposed-config').classList.add('hidden');
}

export async function applyProposedConfig(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab?.dataset.proposal) return;
    
    const proposal = JSON.parse(tab.dataset.proposal);
    
    // 1. Update the UI fields
    window.modelConfigs[path] = { ...window.modelConfigs[path], ...proposal };
    applyModelConfig(path, tabId);
    
    // 2. Hide proposal area
    hideProposedConfig(tabId);
    
    // 3. Start model with new settings
    startModel(path, tabId);
}

export function switchTab(tabId) {
    // Hide all
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    const btn = document.getElementById(`btn-${tabId}`);
    const content = document.getElementById(tabId);
    
    if (btn && content) {
        btn.classList.add('active');
        content.classList.add('active');
        state.currentTabId = tabId;
        state.currentSelectedModel = content.dataset.path;
        
        document.getElementById('no-tab-content').classList.add('hidden');
    }
}

export function closeTab(tabId) {
    const btn = document.getElementById(`btn-${tabId}`);
    const content = document.getElementById(tabId);
    
    if (btn) btn.remove();
    if (content) content.remove();
    
    state.activeTabs = state.activeTabs.filter(t => t.id !== tabId);
    
    if (state.currentTabId === tabId) {
        if (state.activeTabs.length > 0) {
            switchTab(state.activeTabs[state.activeTabs.length - 1].id);
        } else {
            state.currentTabId = null;
            state.currentSelectedModel = null;
            document.getElementById('no-tab-content').classList.remove('hidden');
        }
    }
}

// --- CORE LOGIC ---

export function resolveMmprojPath(model) {
    const candidates = model.mmproj_candidates || [];
    if (!candidates.length) return null;
    const modelJs = model.path.replace(/\\/g, '/');
    const cfg = window.modelConfigs[modelJs] || model.last_config || {};
    const saved = cfg.mmproj_path;
    if (saved && candidates.includes(saved)) return saved;
    return candidates[0];
}

export function buildModelVisionControlsHtml(model, modelJs) {
    // Simplified for library sidebar
    const candidates = model.mmproj_candidates || [];
    if (candidates.length > 0) {
        return `<i class="fas fa-eye text-violet-500 text-[10px]" title="Suporta Visão"></i>`;
    }
    return '';
}

export function getSelectedMmprojForModel(modelPath) {
    const normalized = modelPath.replace(/\\/g, '/');
    const cfg = window.modelConfigs[normalized];
    return cfg?.mmproj_path || null;
}

export function openVisionImportModal(modelPath) {
    const modal = document.getElementById('vision-import-modal');
    const pathInput = document.getElementById('vision-import-model-path');
    const urlInput = document.getElementById('vision-import-url');
    if (!modal || !pathInput || !urlInput) return;
    pathInput.value = modelPath;
    urlInput.value = '';
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    urlInput.focus();
}

export function closeVisionImportModal() {
    const modal = document.getElementById('vision-import-modal');
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

export async function submitVisionImport(event) {
    event.preventDefault();
    const modelPath = document.getElementById('vision-import-model-path')?.value.trim();
    const url = document.getElementById('vision-import-url')?.value.trim();
    if (!modelPath || !url) return;
    try {
        const res = await fetch('/downloads', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url, model_path: modelPath}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Erro ao iniciar download: ' + (err.detail || 'Erro desconhecido'));
            return;
        }
        closeVisionImportModal();
        window.updateDownloads();
        window.updateModels();
    } catch (e) {
        alert('Erro de rede ao iniciar download do projetor.');
    }
}

export async function onMmprojChange(modelPath, selectEl) {
    const mmprojPath = selectEl?.value || null;
    if (!window.modelConfigs[modelPath]) window.modelConfigs[modelPath] = {};
    window.modelConfigs[modelPath].mmproj_path = mmprojPath;
    try {
        await apiFetch('/models/mmproj', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_path: modelPath, mmproj_path: mmprojPath}),
        });
    } catch (e) {
        alert('Erro ao salvar projetor de visao.');
    }
}

export function formatRepoStorageLabel(storage) {
    if (!storage || storage.total_gb == null || storage.used_gb == null) {
        return '-- / -- GB';
    }
    const used = Number(storage.used_gb).toFixed(1);
    const total = Number(storage.total_gb).toFixed(1);
    return `${used} / ${total} GB`;
}

async function persistThinkingEnabled(modelPath, enabled) {
    if (!modelPath) return;
    if (!window.modelConfigs[modelPath]) window.modelConfigs[modelPath] = {};
    window.modelConfigs[modelPath].thinking_enabled = enabled;
    try {
        await apiFetch('/models/thinking', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_path: modelPath, thinking_enabled: enabled}),
        });
    } catch (e) {
        console.error('Erro ao salvar thinking_enabled:', e);
    }
}

export function initDashboard() {
    window.updateStatus();
    window.updateMetrics();
    window.updateDownloads();
    window.updateModels();
    checkForUpdates();
    
    // Close sidebar on small screens initially
    if (window.innerWidth < 1024) toggleSidebar(false);
}

export function getTabActionsHtml(path, tabId, isRunning, port = 8085) {
    if (isRunning) {
        return `
            <a href="/ui/${port}/" target="_blank" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-xl flex items-center gap-2 uppercase tracking-widest shadow-xl shadow-blue-600/20 transition-all active:scale-95">
                <i class="fas fa-comments"></i> Chat
            </a>
            <button onclick="stopModel(${port})" class="px-5 py-2.5 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/20 text-[10px] font-black rounded-xl transition-all uppercase tracking-widest active:scale-95">
                Encerrar
            </button>
        `;
    }
    return `
        <button onclick="startModel('${path}', '${tabId}')" class="px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-2xl active:scale-95 flex items-center gap-3 uppercase tracking-[0.2em] shadow-2xl shadow-blue-600/30 transition-all">
            <i class="fas fa-bolt"></i> Iniciar Instância
        </button>
    `;
}

export function selectModel(path, elementId) {
    const container = document.getElementById(`lib-${elementId}`);
    const name = container?.querySelector('.model-name')?.innerText || 'Modelo';
    createModelTab(path, name, elementId);
    
    // Visual feedback in sidebar
    document.querySelectorAll('.model-item-container').forEach(el => el.classList.remove('active-selection'));
    if (container) container.classList.add('active-selection');
}

export function applyModelConfig(path, tabId) {
    const cfg = window.modelConfigs[path];
    const tab = document.getElementById(tabId);
    if (!cfg || !tab) return;
    
    if (cfg.context_size) setContextSize(cfg.context_size, tabId);
    if (cfg.parallel_slots) tab.querySelector('.tab-parallel-slots').value = cfg.parallel_slots;
    if (cfg.batch_size) tab.querySelector('.tab-batch-size').value = cfg.batch_size;
    if (cfg.cache_type_k) tab.querySelector('.tab-cache-type-k').value = cfg.cache_type_k;
    if (cfg.cache_type_v) tab.querySelector('.tab-cache-type-v').value = cfg.cache_type_v;
    if (cfg.ubatch_size) tab.querySelector('.tab-ubatch-size').value = cfg.ubatch_size;
    if (cfg.numa_enabled !== undefined) tab.querySelector('.tab-numa-toggle').checked = !!cfg.numa_enabled;
    if (cfg.threads !== undefined) tab.querySelector('.tab-threads').value = String(cfg.threads);
    if (cfg.threads_batch !== undefined) tab.querySelector('.tab-threads-batch').value = String(cfg.threads_batch);
    if (cfg.split_mode) tab.querySelector('.tab-split-mode').value = cfg.split_mode;
    
    const abToggle = tab.querySelector('.tab-auto-balance-toggle');
    if (abToggle) abToggle.checked = !!cfg.auto_balance;
    
    const thinkingToggle = tab.querySelector('.tab-thinking-toggle');
    if (thinkingToggle) thinkingToggle.checked = cfg.thinking_enabled !== false;
    
    const mtpToggle = tab.querySelector('.tab-mtp-toggle');
    if (mtpToggle) mtpToggle.checked = !!cfg.mtp_enabled;
    
    if (cfg.gpu_weights) {
        applyGpuWeightsToUI(cfg.gpu_weights, false, tabId);
    }
    
    updateTotal(tabId);
}

export async function setDefaultModel(checkbox, path) {
    try {
        await fetch('/set_default', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path, add: checkbox.checked }),
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

export async function saveModelsDir() {
    const input = document.getElementById('models-dir-input');
    const modelsDir = input?.value.trim();
    if (!modelsDir) return;
    try {
        const res = await apiFetch('/models/dir', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({models_dir: modelsDir}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Erro: ' + (err.detail || 'Inacessivel'));
            return;
        }
        await updateModels();
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
        
        document.getElementById('model-count').innerText = data.models.length;
        document.getElementById('repo-storage').innerText = formatRepoStorageLabel(data.storage);
        
        const dirInput = document.getElementById('models-dir-input');
        if (dirInput && data.storage?.path && document.activeElement !== dirInput) {
            dirInput.value = data.storage.path;
        }
        
        const container = document.getElementById('model-list-container');
        container.innerHTML = data.models.map(m => {
            const m_js = m.path.replace(/\\/g, '/');
            if (m.last_config) window.modelConfigs[m_js] = m.last_config;
            
            const isDefault = (cfg.default_models || []).includes(m_js) || cfg.default_model === m_js;
            const status = (state.activeInstances || []).find(i => i.model_path.replace(/\\/g, '/') === m_js);
            const runningClass = status ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-slate-700/50 bg-slate-800/40';
            const selectedClass = state.currentSelectedModel === m_js ? 'active-selection' : '';

            return `
            <div id="lib-${m.id}" class="model-item-container group p-3 rounded-xl border transition-all cursor-pointer ${runningClass} ${selectedClass}" 
                 onclick="selectModel('${m_js}', '${m.id}')">
                <div class="flex items-start justify-between gap-2 overflow-hidden">
                    <div class="flex-1 min-w-0">
                        <p class="model-name text-[11px] font-bold text-slate-100 truncate">${m.name}</p>
                        <p class="text-[8px] text-slate-500 font-mono uppercase truncate mt-0.5">${m.dir}</p>
                    </div>
                    ${status ? '<div class="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981]"></div>' : ''}
                </div>
                <div class="flex items-center justify-between mt-3 pt-2 border-t border-slate-700/30">
                    <div class="flex items-center gap-1">
                        <button onclick="event.stopPropagation(); renameModel('${m_js}')" class="w-6 h-6 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-blue-400"><i class="fas fa-edit text-[9px]"></i></button>
                        <button onclick="event.stopPropagation(); deleteModel('${m_js}')" class="w-6 h-6 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-red-400"><i class="fas fa-trash-alt text-[9px]"></i></button>
                    </div>
                    <label class="flex items-center gap-1.5 cursor-pointer" onclick="event.stopPropagation()">
                        <span class="text-[8px] font-black text-slate-600 uppercase">Auto-Start</span>
                        <input type="checkbox" class="w-3 h-3 bg-slate-900 border-slate-700 rounded text-blue-600" ${isDefault ? 'checked' : ''} onclick="setDefaultModel(this, '${m_js}')">
                    </label>
                </div>
            </div>`;
        }).join('');
    } catch (e) {}
}

export async function renameModel(path) {
    const currentName = path.split('/').pop().replace('.gguf', '');
    const newName = prompt("Novo nome:", currentName);
    if (!newName || newName === currentName) return;
    try {
        const res = await fetch('/rename', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path, new_name: newName}),
        });
        if (res.ok) updateModels();
    } catch (e) {}
}

export async function deleteModel(path) {
    if (!confirm("Excluir modelo permanentemente?")) return;
    try {
        const res = await fetch('/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path}),
        });
        if (res.ok) updateModels();
    } catch (e) {}
}

export async function startModel(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    tab.querySelector('.tab-log-box').innerHTML = '';
    const weights = collectDeviceWeightsFromUI(tabId);
    const weightValidation = validateDeviceWeights(weights);
    if (!weightValidation.ok) return alert(weightValidation.message);
    
    const mmprojPath = getSelectedMmprojForModel(path);
    const splitMode = tab.querySelector('.tab-split-mode').value;
    const cacheTypeK = tab.querySelector('.tab-cache-type-k').value;
    const cacheTypeV = tab.querySelector('.tab-cache-type-v').value;
    const ubatchSize = parseInt(tab.querySelector('.tab-ubatch-size').value, 10) || 512;
    const numaEnabled = tab.querySelector('.tab-numa-toggle').checked;
    const threads = parseInt(tab.querySelector('.tab-threads').value, 10) || 0;
    const threadsBatch = parseInt(tab.querySelector('.tab-threads-batch').value, 10) || 0;
    const parallelSlots = parseInt(tab.querySelector('.tab-parallel-slots').value, 10) || 1;
    const batchSize = parseInt(tab.querySelector('.tab-batch-size').value, 10) || 2048;
    const autoBalance = tab.querySelector('.tab-auto-balance-toggle').checked;
    const thinkingEnabled = tab.querySelector('.tab-thinking-toggle').checked;
    const mtpEnabled = tab.querySelector('.tab-mtp-toggle').checked;
    
    const contextSize = getContextSize(tabId);
    if (contextSize === null) return alert('Contexto inválido');

    const statusBadge = tab.querySelector('.tab-status-badge');
    statusBadge.innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2"></i> INICIANDO...';
    statusBadge.className = 'tab-status-badge px-4 py-2 rounded-xl text-[9px] font-black tracking-widest uppercase glass border-blue-500/50 text-blue-400';

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
                ubatch_size: ubatchSize,
                cache_type_k: cacheTypeK,
                cache_type_v: cacheTypeV,
                numa_enabled: numaEnabled,
                threads: threads,
                threads_batch: threadsBatch,
                split_mode: splitMode,
                auto_balance: autoBalance,
                thinking_enabled: thinkingEnabled,
                mtp_enabled: mtpEnabled,
            }),
        });
        
        if (!res.ok) {
            const err = await res.json();
            alert("Erro: " + (err.detail || "Falha ao iniciar"));
            window.updateStatus();
            return;
        }
        
        const startData = await res.json();
        if (startData.port) {
             state.currentActivePort = startData.port;
             // Link this tab to this port for logs if needed
        }
        
        await new Promise(r => setTimeout(r, 2000));
        await window.updateStatus();
        const inst = (state.activeInstances || []).find(
            i => (i.model_path || '').replace(/\\/g, '/') === path.replace(/\\/g, '/')
        );
        if (!inst || inst.status !== 'running') {
            alert('O servidor encerrou logo após iniciar. Verifique os logs abaixo.');
        }
    } catch (e) {
        alert("Erro de rede.");
        window.updateStatus();
    }
}

export async function stopModel(port = null) {
    if (!confirm("Encerrar esta instância?")) return;
    try {
        const url = port ? `/stop?port=${port}` : '/stop';
        await apiFetch(url, {method: 'POST'});
        setTimeout(window.updateStatus, 1000);
    } catch (e) {}
}
