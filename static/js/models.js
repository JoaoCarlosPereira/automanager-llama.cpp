import { state } from './state.js?v=4.0.2';
import { apiFetch, sessionExpiredHandled } from './auth.js?v=4.0.2';
import {
    getContextSize, setContextSize, resetToDefaults, applyGpuWeightsToUI,
    updateTotal, hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge, cancelAutoBalance, showAutoBalanceProgress,
    hideAutoBalanceProgress,
    isModelHardwareIncapable, modelIncapableBadgeHtml, modelIncapableRowClass,
    bindGpuManualListeners, syncContextSizeCustomVisibility,
    updateThinkingBadge, updateMtpBadge, getMtpDraftTokens, syncMtpDraftTokensState,
    validateDeviceWeights,
    collectDeviceWeightsFromUI,
} from './gpu.js?v=4.0.2';
import { startLogs } from './metrics.js?v=4.0.2';
import { checkForUpdates } from './version.js?v=4.0.2';

// --- TAB MANAGEMENT ---

function fallbackModelId(path) {
    let hash = 0;
    for (let i = 0; i < path.length; i++) {
        hash = ((hash << 5) - hash + path.charCodeAt(i)) >>> 0;
    }
    return hash.toString(16).padStart(8, '0').slice(0, 12);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function formatContextLabel(contextSize) {
    const size = Number(contextSize);
    if (!Number.isFinite(size) || size <= 0) return null;
    if (size >= 1024 && size % 1024 === 0) return `${size / 1024}K`;
    if (size >= 1024) return `${(size / 1024).toFixed(1)}K`;
    return `${size}`;
}

export function pickFrequentModels(models, cfg, limit = 6) {
    const modelConfigs = cfg?.model_configs || {};
    const defaultSet = new Set(
        [...(cfg?.default_models || []), cfg?.default_model]
            .filter(Boolean)
            .map(p => p.replace(/\\/g, '/'))
    );
    const runningPaths = new Set(
        (state.activeInstances || [])
            .filter(i => i.status === 'running')
            .map(i => (i.model_path || '').replace(/\\/g, '/'))
    );

    const ranked = (models || []).map(model => {
        const path = model.path.replace(/\\/g, '/');
        const saved = modelConfigs[path] || model.last_config || window.modelConfigs[path] || {};
        const isDefault = defaultSet.has(path);
        const isRunning = runningPaths.has(path);
        const hasSavedConfig = Object.keys(saved).length > 0;
        let score = 0;

        if (isDefault) score += 1_000_000;
        if (isRunning) score += 500_000;
        if (hasSavedConfig) score += 10_000;
        if (saved.last_started) {
            const ts = Date.parse(saved.last_started);
            if (!Number.isNaN(ts)) score += ts / 1000;
        }

        return { model, path, saved, isDefault, isRunning, hasSavedConfig, score };
    })
        .filter(item => item.isDefault || item.isRunning || item.hasSavedConfig)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);

    return ranked;
}

export function renderNoTabShortcuts(models = null, cfg = null) {
    if (state.activeTabs.length > 0) return;

    const modelList = models ?? state.lastModelsList ?? [];
    const config = cfg ?? state.lastConfig ?? {};

    const panel = document.getElementById('no-tab-shortcuts');
    const grid = document.getElementById('no-tab-shortcuts-grid');
    const countEl = document.getElementById('no-tab-shortcuts-count');
    const emptyEl = document.getElementById('no-tab-shortcuts-empty');
    if (!panel || !grid) return;

    const shortcuts = pickFrequentModels(modelList, config);
    if (!shortcuts.length) {
        panel.classList.add('hidden');
        grid.innerHTML = '';
        if (countEl) countEl.textContent = '';
        if (emptyEl) emptyEl.classList.remove('hidden');
        return;
    }

    if (emptyEl) emptyEl.classList.add('hidden');
    panel.classList.remove('hidden');
    if (countEl) countEl.textContent = `${shortcuts.length} atalho${shortcuts.length === 1 ? '' : 's'}`;

    grid.innerHTML = shortcuts.map(({ model, path, saved, isDefault, isRunning }) => {
        const displayName = model.name.replace(/\.gguf$/i, '');
        const contextLabel = formatContextLabel(saved.context_size);
        const badges = [
            isRunning ? '<span class="text-[7px] font-black uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Online</span>' : '',
            isDefault ? '<span class="text-[7px] font-black uppercase tracking-wider text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">Auto-Start</span>' : '',
        ].filter(Boolean).join('');

        const meta = [
            contextLabel ? `Ctx ${contextLabel}` : null,
            saved.parallel_slots ? `${saved.parallel_slots} slot${saved.parallel_slots === 1 ? '' : 's'}` : null,
            saved.mtp_enabled ? `MTP ${saved.mtp_draft_tokens || 3}` : null,
        ].filter(Boolean).join(' · ');

        return `
            <button type="button" onclick="selectModel('${path.replace(/'/g, "\\'")}', '${model.id}')"
                class="group glass rounded-2xl border border-slate-800/80 hover:border-blue-500/40 bg-slate-900/40 hover:bg-slate-900/70 p-4 text-left transition-all active:scale-[0.99]">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0 flex-1">
                        <p class="text-[12px] font-bold text-slate-100 truncate group-hover:text-white">${escapeHtml(displayName)}</p>
                        <p class="text-[8px] text-slate-500 font-mono uppercase truncate mt-1">${escapeHtml(model.dir || '/')}</p>
                    </div>
                    <i class="fas fa-arrow-right text-[9px] text-slate-600 group-hover:text-blue-400 transition-colors mt-1"></i>
                </div>
                ${badges ? `<div class="flex flex-wrap gap-1.5 mt-3">${badges}</div>` : ''}
                ${meta ? `<p class="text-[8px] text-slate-500 font-mono mt-3 uppercase tracking-wider">${escapeHtml(meta)}</p>` : ''}
            </button>
        `;
    }).join('');
}

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

export function createModelTab(path, name, id, activate = true) {
    const m_js = path.replace(/\\/g, '/');
    const tabId = `tab-${id}`;
    
    // If tab exists, just switch to it
    if (document.getElementById(tabId)) {
        if (activate) switchTab(tabId);
        return tabId;
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
    if (activate) switchTab(tabId);
    
    // 5. Load Configs
    if (window.modelConfigs[m_js]) {
        applyModelConfig(m_js, tabId);
    }
    
    // 6. Bind Listeners for this tab
    bindTabListeners(tabId);
    return tabId;
}

function bindTabListeners(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    
    const path = tab.dataset.path;
    
    tab.querySelector('.tab-thinking-toggle')?.addEventListener('change', (e) => persistThinkingEnabled(path, e.target.checked));
    tab.querySelector('.tab-mtp-toggle')?.addEventListener('change', () => syncMtpDraftTokensState(tabId));
    tab.querySelector('.tab-context-size')?.addEventListener('change', () => syncContextSizeCustomVisibility(tabId));
    tab.querySelector('.tab-context-size-custom')?.addEventListener('input', () => updateTotal(tabId));
    
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
    tab.querySelector('.tab-auto-balance-cancel-btn')?.addEventListener('click', async () => {
        await cancelAutoBalance();
        window.updateStatus?.();
    });
    
    // Apply/Discard Proposed Config
    tab.querySelector('.tab-apply-config-btn').onclick = () => applyProposedConfig(path, tabId);
    tab.querySelector('.tab-discard-config-btn').onclick = () => hideProposedConfig(tabId);

    bindPinnedFieldListeners(tabId);
    updateTotal(tabId);
    syncMtpDraftTokensState(tabId);
}

function collectPinnedFieldsFromTab(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return {};
    return {
        context_size: tab.querySelector('.tab-pin-context')?.checked ?? false,
        parallel_slots: tab.querySelector('.tab-pin-slots')?.checked ?? false,
        batch_size: tab.querySelector('.tab-pin-batch')?.checked ?? false,
        ubatch_size: tab.querySelector('.tab-pin-ubatch')?.checked ?? false,
        cache_type: tab.querySelector('.tab-pin-cache')?.checked ?? false,
        threads: tab.querySelector('.tab-pin-threads')?.checked ?? false,
        thinking: tab.querySelector('.tab-pin-thinking')?.checked ?? false,
        mtp: tab.querySelector('.tab-pin-mtp')?.checked ?? false,
        numa: tab.querySelector('.tab-pin-numa')?.checked ?? false,
        split_mode: tab.querySelector('.tab-pin-split-mode')?.checked ?? false,
    };
}

function syncPinnedFieldIcon(pin) {
    const icon = pin?.nextElementSibling;
    if (!icon) return;
    if (pin.checked) {
        icon.classList.remove('text-slate-700');
        icon.classList.add('text-blue-500');
    } else {
        icon.classList.remove('text-blue-500');
        icon.classList.add('text-slate-700');
    }
}

function bindPinnedFieldListeners(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    tab.querySelectorAll('input[class*="tab-pin-"]').forEach(pin => {
        pin.addEventListener('change', () => syncPinnedFieldIcon(pin));
    });
}

function applyPinnedFieldsToTab(tabId, pinnedFields) {
    if (!pinnedFields) return;
    const tab = document.getElementById(tabId);
    if (!tab) return;
    const selectors = {
        context_size: '.tab-pin-context',
        parallel_slots: '.tab-pin-slots',
        batch_size: '.tab-pin-batch',
        ubatch_size: '.tab-pin-ubatch',
        cache_type: '.tab-pin-cache',
        threads: '.tab-pin-threads',
        thinking: '.tab-pin-thinking',
        mtp: '.tab-pin-mtp',
        numa: '.tab-pin-numa',
        split_mode: '.tab-pin-split-mode',
    };
    for (const [key, selector] of Object.entries(selectors)) {
        const pin = tab.querySelector(selector);
        if (!pin || pinnedFields[key] === undefined) continue;
        pin.checked = !!pinnedFields[key];
        syncPinnedFieldIcon(pin);
    }
}

export async function startSmartCalibration(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    const pinnedFields = collectPinnedFieldsFromTab(tabId);

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
    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-[10px] font-black tracking-[0.2em] uppercase glass border-amber-500/40 text-amber-400 bg-amber-500/5';

    state.autoBalancePending = true;
    state.autoBalanceTabId = tabId;
    state.autoBalanceSeenActive = false;
    showAutoBalanceProgress({ message: 'Iniciando calibração smart...', smart_calibration: true }, tabId);
    
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
                thinking_enabled: tab.querySelector('.tab-thinking-toggle').checked,
                mtp_enabled: tab.querySelector('.tab-mtp-toggle').checked,
                mtp_draft_tokens: getMtpDraftTokens(tabId),
                numa_enabled: tab.querySelector('.tab-numa-toggle').checked,
                split_mode: tab.querySelector('.tab-split-mode').value,
                auto_balance: true,
                smart_calibration: true,
                pinned_fields: pinnedFields,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            alert("Erro na calibração: " + (err.detail || "Falha"));
            state.autoBalancePending = false;
            state.autoBalanceTabId = null;
            state.autoBalanceSeenActive = false;
            hideAutoBalanceProgress(tabId);
            window.updateStatus();
            return;
        }

        const data = await res.json();
        if (data.run_id != null) {
            state.autoBalanceRunId = data.run_id;
        }
        if (data.probing) {
            state.autoBalancePending = true;
            state.autoBalanceTabId = tabId;
            showAutoBalanceProgress({ message: 'Calibração smart em andamento...', smart_calibration: true }, tabId);
        }
        window.updateStatus?.();
    } catch (e) {
        alert("Erro de rede.");
        state.autoBalancePending = false;
        state.autoBalanceTabId = null;
        state.autoBalanceSeenActive = false;
        hideAutoBalanceProgress(tabId);
        window.updateStatus();
    }
}

export function showProposedConfig(tabId, proposal, gpuWeights = null) {
    const tab = document.getElementById(tabId);
    if (!tab || !proposal) return;

    const area = tab.querySelector('.tab-proposed-config');
    const details = tab.querySelector('.tab-proposed-details');

    const payload = { ...proposal };
    if (gpuWeights) payload.gpu_weights = gpuWeights;
    tab.dataset.proposal = JSON.stringify(payload);

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

    const modelPath = (tab.dataset.path || '').replace(/\\/g, '/');
    if (modelPath) {
        window.modelConfigs[modelPath] = {
            ...(window.modelConfigs[modelPath] || {}),
            ...payload,
        };
        applyModelConfig(modelPath, tabId);
    }
}

export function hideProposedConfig(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    tab.querySelector('.tab-proposed-config')?.classList.add('hidden');
    delete tab.dataset.proposal;
}

function collectStartPayloadFromTab(path, tabId, { autoBalanceProfile = false } = {}) {
    const tab = document.getElementById(tabId);
    if (!tab) return null;

    const normalized = path.replace(/\\/g, '/');
    const contextSize = getContextSize(tabId);
    if (contextSize === null) return null;

    return {
        path: normalized,
        mmproj_path: getSelectedMmprojForModel(normalized) || null,
        gpu_weights: collectDeviceWeightsFromUI(tabId),
        context_size: contextSize,
        parallel_slots: parseInt(tab.querySelector('.tab-parallel-slots').value, 10) || 1,
        batch_size: parseInt(tab.querySelector('.tab-batch-size').value, 10) || 2048,
        ubatch_size: parseInt(tab.querySelector('.tab-ubatch-size').value, 10) || 512,
        cache_type_k: tab.querySelector('.tab-cache-type-k').value,
        cache_type_v: tab.querySelector('.tab-cache-type-v').value,
        numa_enabled: tab.querySelector('.tab-numa-toggle').checked,
        threads: parseInt(tab.querySelector('.tab-threads').value, 10) || 0,
        threads_batch: parseInt(tab.querySelector('.tab-threads-batch').value, 10) || 0,
        split_mode: tab.querySelector('.tab-split-mode').value,
        auto_balance: false,
        auto_balance_profile: autoBalanceProfile,
        pinned_fields: collectPinnedFieldsFromTab(tabId),
        thinking_enabled: tab.querySelector('.tab-thinking-toggle').checked,
        mtp_enabled: tab.querySelector('.tab-mtp-toggle').checked,
        mtp_draft_tokens: getMtpDraftTokens(tabId),
    };
}

export async function applyProposedConfig(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab?.dataset.proposal) return;

    const normalized = path.replace(/\\/g, '/');
    const proposal = JSON.parse(tab.dataset.proposal);

    window.modelConfigs[normalized] = {
        ...(window.modelConfigs[normalized] || {}),
        ...proposal,
    };
    applyModelConfig(normalized, tabId);

    const payload = collectStartPayloadFromTab(normalized, tabId, {
        autoBalanceProfile: true,
    });
    if (!payload) return alert('Contexto inválido');

    const weightValidation = validateDeviceWeights(payload.gpu_weights);
    if (!weightValidation.ok) return alert(weightValidation.message);

    window.modelConfigs[normalized] = {
        ...(window.modelConfigs[normalized] || {}),
        ...payload,
    };

    hideProposedConfig(tabId);

    tab.querySelector('.tab-log-box').innerHTML = '';
    const statusBadge = tab.querySelector('.tab-status-badge');
    statusBadge.innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2"></i> SALVANDO...';
    statusBadge.className = 'tab-status-badge px-4 py-2 rounded-xl text-[9px] font-black tracking-widest uppercase glass border-amber-500/40 text-amber-400';

    try {
        const res = await apiFetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            alert('Erro ao salvar/iniciar: ' + (err.detail || 'Falha'));
            window.updateStatus();
            return;
        }

        const startData = await res.json();
        if (startData.port) state.currentActivePort = startData.port;

        await window.updateModels?.();
        await new Promise(r => setTimeout(r, 2000));
        await window.updateStatus();
    } catch (e) {
        alert('Erro de rede ao salvar configuração.');
        window.updateStatus();
    }
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
            renderNoTabShortcuts();
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

export async function syncRunningModelTabsOnLoad() {
    if (state.initialTabsSynced) return;

    try {
        const [statusRes, modelsRes] = await Promise.all([
            apiFetch('/status'),
            apiFetch('/models'),
        ]);
        if (sessionExpiredHandled || !statusRes.ok || !modelsRes.ok) return;

        const statusData = await statusRes.json();
        const modelsData = await modelsRes.json();

        state.activeInstances = statusData.instances || [];
        const running = state.activeInstances.filter(i => i.status === 'running');
        if (!running.length) {
            state.initialTabsSynced = true;
            return;
        }

        let firstTabId = null;
        for (const inst of running) {
            const path = (inst.model_path || '').replace(/\\/g, '/');
            if (!path) continue;

            const model = modelsData.models.find(
                m => m.path.replace(/\\/g, '/') === path
            );
            const id = model?.id || fallbackModelId(path);
            const name = model?.name || inst.model || path.split('/').pop();
            if (inst.config) window.modelConfigs[path] = inst.config;
            const tabId = `tab-${id}`;

            if (!document.getElementById(tabId)) {
                createModelTab(path, name, id, false);
            }
            if (!firstTabId) firstTabId = tabId;
        }

        if (firstTabId) switchTab(firstTabId);
        state.initialTabsSynced = true;
    } catch (e) {
        console.error('syncRunningModelTabsOnLoad error:', e);
    }
}

export async function initDashboard() {
    await syncRunningModelTabsOnLoad();
    await window.updateStatus();
    window.updateMetrics();
    window.updateDownloads();
    await window.updateModels();
    checkForUpdates();
    
    if (typeof window.toggleSidebar === 'function') {
        window.toggleSidebar(false);
    } else {
        toggleSidebar(false);
    }
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
    
    const thinkingToggle = tab.querySelector('.tab-thinking-toggle');
    if (thinkingToggle) thinkingToggle.checked = cfg.thinking_enabled !== false;
    
    const mtpToggle = tab.querySelector('.tab-mtp-toggle');
    if (mtpToggle) mtpToggle.checked = !!cfg.mtp_enabled;
    const mtpDraft = tab.querySelector('.tab-mtp-draft-tokens');
    if (mtpDraft) {
        const tokens = cfg.mtp_draft_tokens ?? window.__constants?.DEFAULT_MTP_DRAFT_TOKENS ?? 3;
        mtpDraft.value = String(Math.max(1, Math.min(4, tokens)));
    }
    syncMtpDraftTokensState(tabId);
    
    if (cfg.gpu_weights) {
        applyGpuWeightsToUI(cfg.gpu_weights, false, tabId);
    }

    applyPinnedFieldsToTab(tabId, cfg.pinned_fields);
    
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

function tabHasPendingProposal(modelPath) {
    const normalized = modelPath.replace(/\\/g, '/');
    return state.activeTabs.some(t => {
        if (t.path.replace(/\\/g, '/') !== normalized) return false;
        const el = document.getElementById(t.id);
        const panel = el?.querySelector('.tab-proposed-config');
        return !!(el?.dataset.proposal && panel && !panel.classList.contains('hidden'));
    });
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
        state.lastModelsList = data.models;
        state.lastConfig = cfg;
        
        document.getElementById('model-count').innerText = data.models.length;
        document.getElementById('repo-storage').innerText = formatRepoStorageLabel(data.storage);
        
        const dirInput = document.getElementById('models-dir-input');
        if (dirInput && data.storage?.path && document.activeElement !== dirInput) {
            dirInput.value = data.storage.path;
        }
        
        const container = document.getElementById('model-list-container');
        container.innerHTML = data.models.map(m => {
            const m_js = m.path.replace(/\\/g, '/');
            if (m.last_config && !tabHasPendingProposal(m_js)) {
                window.modelConfigs[m_js] = m.last_config;
            }
            
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

        renderNoTabShortcuts(data.models, cfg);
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
    if (!confirm('Excluir modelo permanentemente?')) return;
    const normalized = path.replace(/\\/g, '/');
    try {
        const res = await apiFetch('/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Erro: ' + (err.detail || 'Falha ao excluir'));
            return;
        }

        const openTab = state.activeTabs.find(t => t.path === normalized);
        if (openTab) closeTab(openTab.id);
        delete window.modelConfigs[normalized];
        await updateModels();
    } catch (e) {
        alert('Erro de rede ao excluir modelo.');
    }
}

export async function startModel(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    tab.querySelector('.tab-log-box').innerHTML = '';
    const payload = collectStartPayloadFromTab(path, tabId, { autoBalanceProfile: false });
    if (!payload) return alert('Contexto inválido');

    const weightValidation = validateDeviceWeights(payload.gpu_weights);
    if (!weightValidation.ok) return alert(weightValidation.message);

    const statusBadge = tab.querySelector('.tab-status-badge');
    statusBadge.innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2"></i> INICIANDO...';
    statusBadge.className = 'tab-status-badge px-4 py-2 rounded-xl text-[9px] font-black tracking-widest uppercase glass border-blue-500/50 text-blue-400';

    try {
        const res = await apiFetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
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
