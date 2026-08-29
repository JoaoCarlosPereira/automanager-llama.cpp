import { state } from './state.js?v=4.2.3';
import { apiFetch, sessionExpiredHandled, showToast, showConfirm, showPrompt } from './auth.js?v=4.2.7';
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
    fetchLlamaBins,
    populateLlamaBinSelect,
    getSelectedLlamaBin,
    syncTurboquantPanelVisibility,
    applyTurboquantConfig,
    applyTurboquantPreset,
    applySavedCacheTypes,
    syncTurboquantToCacheFields,
    syncMainCacheToTurboFields,
    getTurboquantPreset,
    detectTurboquantPreset,
    getEffectiveCacheTypes,
    isTurboquantBin,
} from './gpu.js?v=4.2.3';

const tabLogHeightObservers = new Map();
let tabLogHeightResizeTimer = null;
let deferredModelListUpdate = null;
const platformActions = new Set();
let cliproxyAuthPollTimer = null;
let cliproxyAuthSessionId = null;

const MMproj_SELECT_ATTRS =
    'onmousedown="event.stopPropagation()" onpointerdown="event.stopPropagation()" '
    + 'onclick="event.stopPropagation()" onchange="onMmprojChange(\'__PATH__\', this)"';

function syncTabLogPanelHeight(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab?.classList.contains('active')) return;

    const configPanel = tab.querySelector('.tab-config-panel');
    const layoutRow = tab.querySelector('.tab-layout-row');
    if (!configPanel || !layoutRow) return;

    const height = Math.round(configPanel.getBoundingClientRect().height);
    if (height <= 0) return;

    layoutRow.style.setProperty('--tab-config-height', `${height}px`);
}

function bindTabLogPanelHeightSync(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    const configPanel = tab.querySelector('.tab-config-panel');
    if (!configPanel) return;

    tabLogHeightObservers.get(tabId)?.disconnect();

    const observer = new ResizeObserver(() => {
        syncTabLogPanelHeight(tabId);
    });
    observer.observe(configPanel);
    tabLogHeightObservers.set(tabId, observer);

    requestAnimationFrame(() => syncTabLogPanelHeight(tabId));
}

function unbindTabLogPanelHeightSync(tabId) {
    const observer = tabLogHeightObservers.get(tabId);
    if (!observer) return;
    observer.disconnect();
    tabLogHeightObservers.delete(tabId);
}

if (typeof window !== 'undefined') {
    window.addEventListener('resize', () => {
        clearTimeout(tabLogHeightResizeTimer);
        tabLogHeightResizeTimer = setTimeout(() => {
            if (state.currentTabId) syncTabLogPanelHeight(state.currentTabId);
        }, 100);
    });

    document.addEventListener('focusout', (event) => {
        if (!event.target?.matches?.('.model-mmproj-select')) return;
        if (!deferredModelListUpdate) return;
        const render = deferredModelListUpdate;
        deferredModelListUpdate = null;
        requestAnimationFrame(() => {
            render().catch(() => {});
        });
    }, true);
}
import { setProxyPrimary, setProxyEligible, setProxyMaxParallel } from './proxy.js?v=4.2.22';
import { attachTabLogs, detachTabLogs } from './metrics.js?v=4.2.12';
import { checkForUpdates } from './version.js?v=4.2.3';

// --- TAB MANAGEMENT ---

let tabInstanceCounter = 0;

function nextTabInstanceId(modelId) {
    tabInstanceCounter += 1;
    return `tab-${modelId}-${tabInstanceCounter}`;
}

function refreshTabLabelsForPath(path) {
    const normalized = path.replace(/\\/g, '/');
    const tabs = state.activeTabs.filter(t => t.path === normalized);
    tabs.forEach((tab, index) => {
        const label = tabs.length > 1 ? `${tab.name} (${index + 1})` : tab.name;
        const btn = document.getElementById(`btn-${tab.id}`);
        const labelEl = btn?.querySelector('.tab-label');
        if (labelEl) labelEl.textContent = label;
    });
}

function findTabForModel(path, { forceNew = false } = {}) {
    const normalized = path.replace(/\\/g, '/');
    if (forceNew) return null;
    return state.activeTabs.find(t => t.path === normalized && document.getElementById(t.id)) || null;
}

function findTabForBackend(backendId, { forceNew = false } = {}) {
    if (!backendId || forceNew) return null;
    return state.activeTabs.find(
        t => t.kind === 'platform' && t.backendId === backendId && document.getElementById(t.id)
    ) || null;
}

const PLATFORM_LIMITS_INFO = {
    codex: 'Modelos via assinatura OpenAI/Codex. Limites de taxa e uso seguem a politica da conta autenticada no CLIProxyAPI.',
    claude: 'Modelos via assinatura Claude. Limites de taxa e uso seguem a politica da conta autenticada no CLIProxyAPI.',
    antigravity: 'Modelos via Google Antigravity. Limites de taxa e uso seguem a politica da conta autenticada no CLIProxyAPI.',
};

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
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

function jsString(value) {
    return String(value ?? '')
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'");
}

function platformDomId(backendId) {
    return `platform-${String(backendId || '').replace(/[^a-zA-Z0-9_-]/g, '-')}`;
}

function platformDisplayState(platform) {
    const backendId = platform.backend_id;
    const runtime = (state.platforms || []).find(p => p.backend_id === backendId) || {};
    const activeInstance = (state.activeInstances || []).find(i => i.backend_id === backendId);
    const merged = { ...platform, ...runtime };
    if (activeInstance?.status === 'running' || merged.active) {
        merged.status = 'running';
        merged.active = true;
        merged.sidecar_port = merged.sidecar_port || activeInstance?.port;
    }
    return merged;
}

const PLATFORM_STATUS_LABELS = {
    running: 'RODANDO',
    detected: 'DISPONIVEL',
    stopped: 'PARADO',
    not_ready: 'INDISPONIVEL',
    missing: 'NAO DETECTADO',
};

function platformAuthSummary(platform) {
    const auth = platform.cliproxy_auth || {};
    if (platform.provider === 'generic-openai' && !platform.account_id) {
        const count = (auth.accounts || []).length;
        return count > 0 ? `Autenticado (${count} contas)` : 'Nao configurado';
    }
    if (platform.provider !== 'ollama-cloud' && (!platform.detected || platform.cliproxy_detected === false)) return '';
    if (auth.authenticated) {
        const count = (auth.accounts || []).length;
        return count > 1 ? `Autenticado (${count} contas)` : 'Autenticado';
    }
    return 'Nao autenticado';
}

function platformAuthClass(platform) {
    const auth = platform.cliproxy_auth || {};
    if (auth.authenticated) {
        return 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10';
    }
    return 'text-amber-300 border-amber-500/30 bg-amber-500/10';
}

function buildPlatformAuthButton(platform, safeBackendId) {
    if (platform.provider !== 'generic-openai' && platform.provider !== 'ollama-cloud' && (!platform.detected || platform.cliproxy_detected === false)) return '';
    const auth = platform.cliproxy_auth || {};
    const provider = platform.provider || '';
    const label = auth.authenticated || (provider === 'generic-openai' && (auth.accounts || []).length > 0) ? 'Gerenciar Contas' : 'Autenticar';
    const displayName = platform.display_name || platform.name || provider;

    let action = 'startCliproxyAuth';
    let args = `'${safeBackendId}', '${jsString(provider)}', '${jsString(displayName)}'`;

    if (provider === 'ollama-cloud') {
        action = 'manageOllamaCloudAuth';
        args = `'${safeBackendId}', '${jsString(displayName)}'`;
    } else if (provider === 'generic-openai') {
        action = 'manageGenericOpenAIAuth';
        args = `'${safeBackendId}', '${jsString(displayName)}'`;
    }

    return `<button type="button" onclick="event.stopPropagation(); ${action}(${args})" title="${label}" aria-label="${label}" class="w-8 h-8 flex items-center justify-center rounded bg-amber-600/10 text-amber-300 hover:bg-amber-600/20"><i class="fas fa-key text-ui-label"></i></button>`;
}

function platformStatusClass(status) {
    if (status === 'running') return 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10';
    if (status === 'detected' || status === 'stopped') return 'text-blue-300 border-blue-500/30 bg-blue-500/10';
    if (status === 'not_ready') return 'text-amber-300 border-amber-500/30 bg-amber-500/10';
    return 'text-slate-500 border-slate-700/50 bg-slate-800/50';
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
            isRunning ? '<span class="text-ui-caption font-black uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20">Online</span>' : '',
            isDefault ? '<span class="text-ui-caption font-black uppercase tracking-wider text-blue-400 bg-blue-500/10 px-2 py-1 rounded border border-blue-500/20">Auto-Start</span>' : '',
        ].filter(Boolean).join('');

        const meta = [
            contextLabel ? `Ctx ${contextLabel}` : null,
            saved.parallel_slots ? `${saved.parallel_slots} slot${saved.parallel_slots === 1 ? '' : 's'}` : null,
            saved.mtp_enabled ? `MTP ${saved.mtp_draft_tokens || 3}` : null,
        ].filter(Boolean).join(' · ');

        return `
            <button type="button" onclick="selectModelFromEvent(event, '${path.replace(/'/g, "\\'")}', '${model.id}')"
                class="group glass rounded-2xl border border-slate-800/80 hover:border-blue-500/40 bg-slate-900/40 hover:bg-slate-900/70 p-4 text-left transition-all active:scale-[0.99]">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0 flex-1">
                        <p class="text-ui-body-sm font-bold text-slate-100 truncate group-hover:text-white">${escapeHtml(displayName)}</p>
                        <p class="text-ui-label text-slate-400 font-mono uppercase truncate mt-1">${escapeHtml(model.dir || '/')}</p>
                    </div>
                    <i class="fas fa-arrow-right text-ui-label text-slate-600 group-hover:text-blue-400 transition-colors mt-1"></i>
                </div>
                ${badges ? `<div class="flex flex-wrap gap-1.5 mt-3">${badges}</div>` : ''}
                ${meta ? `<p class="text-ui-label text-slate-400 font-mono mt-3 uppercase tracking-wider">${escapeHtml(meta)}</p>` : ''}
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

export function createModelTab(path, name, id, activate = true, forceNew = false) {
    const m_js = path.replace(/\\/g, '/');
    const modelId = id || fallbackModelId(m_js);

    const existing = findTabForModel(m_js, { forceNew });
    if (existing) {
        if (activate) switchTab(existing.id);
        return existing.id;
    }

    const tabId = nextTabInstanceId(modelId);
    const tabLabel = escapeHtml(name);

    // 1. Create Tab Button
    const tabBar = document.getElementById('tab-bar');
    const btn = document.createElement('button');
    btn.id = `btn-${tabId}`;
    btn.className = 'tab-btn px-4 h-full flex items-center gap-3 text-ui-body-sm font-bold text-slate-500 border-b-2 border-transparent hover:text-slate-300 transition-all group relative min-w-[120px] max-w-[240px]';
    btn.onclick = () => switchTab(tabId);
    btn.innerHTML = `
        <div class="tab-status-dot w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0 transition-all duration-500"></div>
        <span class="tab-label truncate flex-1 text-left">${tabLabel}</span>
        <span onclick="event.stopPropagation(); closeTab('${tabId}')" class="tab-close-btn w-4 h-4 flex items-center justify-center rounded hover:bg-red-500/20 hover:text-red-500 text-slate-600 transition-all">
            <i class="fas fa-times text-ui-label"></i>
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

    populateLlamaBinSelect(tabId);

    // 3. Register state
    state.activeTabs.push({ id: tabId, path: m_js, name, modelId });

    refreshTabLabelsForPath(m_js);
    
    // 4. Initial Switch
    if (activate) switchTab(tabId);
    
    // 5. Load Configs
    const listed = state.lastModelsList?.find(m => m.path.replace(/\\/g, '/') === m_js);
    if (listed?.last_config) {
        mergeModelConfigFromServer(m_js, listed.last_config);
    }

    if (window.modelConfigs[m_js]) {
        applyModelConfig(m_js, tabId);
    } else {
        syncTurboquantPanelVisibility(tabId);
        persistLlamaBinSettings(m_js, tabId, { silent: true });
    }
    
    // 6. Bind Listeners for this tab
    bindTabListeners(tabId);
    bindGpuManualListeners(tabId);
    bindTabLogPanelHeightSync(tabId);
    const localAliasSelect = tabDiv.querySelector('.local-cursor-alias-select');
    tabDiv.querySelector('.local-cursor-save-alias')?.addEventListener('click', () => {
        const alias = localAliasSelect?.value;
        if (!alias) return;
        saveModelAlias(alias, m_js, tabDiv).then(() => refreshLocalCursorSection(tabDiv, m_js));
    });
    refreshLocalCursorSection(tabDiv, m_js);
    return tabId;
}

function platformTabId(backendId) {
    return `ptab-${String(backendId || '').replace(/[^a-zA-Z0-9_-]/g, '-')}`;
}

export function createPlatformTab(backendId, displayName, forceNew = false) {
    const existing = findTabForBackend(backendId, { forceNew });
    if (existing) {
        switchTab(existing.id);
        return existing.id;
    }

    const tabId = nextTabInstanceId(platformTabId(backendId));
    const tabLabel = escapeHtml(displayName || backendId);

    const tabBar = document.getElementById('tab-bar');
    const btn = document.createElement('button');
    btn.id = `btn-${tabId}`;
    btn.className = 'tab-btn px-4 h-full flex items-center gap-3 text-ui-body-sm font-bold text-slate-500 border-b-2 border-transparent hover:text-slate-300 transition-all group relative min-w-[120px] max-w-[240px]';
    btn.onclick = () => switchTab(tabId);
    btn.innerHTML = `
        <div class="tab-status-dot w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0 transition-all duration-500"></div>
        <span class="tab-label truncate flex-1 text-left"><i class="fas fa-cloud text-violet-500/70 text-ui-label mr-1"></i>${tabLabel}</span>
        <span onclick="event.stopPropagation(); closeTab('${tabId}')" class="tab-close-btn w-4 h-4 flex items-center justify-center rounded hover:bg-red-500/20 hover:text-red-500 text-slate-600 transition-all">
            <i class="fas fa-times text-ui-label"></i>
        </span>
    `;
    tabBar.appendChild(btn);

    const template = document.getElementById('platform-tab-template');
    const content = template.content.cloneNode(true);
    const tabDiv = content.querySelector('.tab-content');
    tabDiv.id = tabId;
    tabDiv.dataset.backendId = backendId;
    tabDiv.dataset.path = backendId;
    tabDiv.querySelector('.platform-tab-name').innerText = displayName || backendId;
    document.getElementById('tabs-container').appendChild(tabDiv);

    state.activeTabs.push({
        id: tabId,
        path: backendId,
        backendId,
        name: displayName || backendId,
        kind: 'platform',
    });

    bindPlatformTabListeners(tabId, backendId);
    bindTabLogPanelHeightSync(tabId);
    populatePlatformTab(tabId, backendId);
    switchTab(tabId);
    loadPlatformTabDetails(tabId, backendId);
    return tabId;
}

function bindPlatformTabListeners(tabId, backendId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    const safeId = jsString(backendId);

    tab.querySelector('.platform-auth-btn')?.addEventListener('click', () => {
        const platform = platformDisplayState(
            (state.lastPlatformList || []).find(p => p.backend_id === backendId) || { backend_id: backendId }
        );
        const displayName = platform.display_name || platform.name || backendId;
        if (platform.provider === 'generic-openai') {
            manageGenericOpenAIAuth(backendId, displayName);
        } else if (platform.provider === 'ollama-cloud') {
            manageOllamaCloudAuth(backendId, displayName);
        } else {
            startCliproxyAuth(backendId, platform.provider || '', displayName);
        }
    });

    tab.querySelector('.platform-refresh-models-btn')?.addEventListener('click', () => {
        loadPlatformTabDetails(tabId, backendId);
    });

    tab.querySelector('.platform-cursor-save-alias')?.addEventListener('click', () => {
        const alias = tab.querySelector('.platform-cursor-alias-select')?.value;
        const target = tab.querySelector('.platform-cursor-target-select')?.value;
        if (alias && target) saveModelAlias(alias, target, tab);
    });

    refreshPlatformCursorSection(tab);

    tab.querySelector('.platform-proxy-primary')?.addEventListener('change', (e) => {
        setProxyPrimary(e.target, null, backendId);
    });
    tab.querySelector('.platform-proxy-eligible')?.addEventListener('change', (e) => {
        setProxyEligible(e.target, null, backendId);
    });
    tab.querySelector('.platform-vision-enabled')?.addEventListener('change', (e) => {
        setPlatformVisionEnabled(e.target, backendId);
    });
    tab.querySelector('.platform-proxy-parallel')?.addEventListener('change', (e) => {
        setProxyMaxParallel(e.target, null, backendId);
    });
    tab.querySelector('.platform-autostart')?.addEventListener('change', (e) => {
        setPlatformAutoStart(e.target, backendId);
    });
    tab.querySelector('.platform-proxy-default-model')?.addEventListener('change', (e) => {
        setPlatformDefaultModel(e.target, backendId);
    });

    const clearLogsBtn = tab.querySelector('.tab-clear-logs-btn');
    if (clearLogsBtn) {
        clearLogsBtn.onclick = () => {
            tab.querySelector('.tab-log-box').innerHTML = '';
            tab.querySelector('.tab-log-box').dataset.connecting = '0';
            const sizeEl = tab.querySelector('.tab-log-size');
            if (sizeEl) sizeEl.innerText = '0 KB';
        };
    }
}

function formatPlatformTimestamp(value) {
    if (!value) return '—';
    const date = new Date(Number(value) * 1000);
    if (Number.isNaN(date.getTime())) return '—';
    return date.toLocaleString();
}

function renderPlatformModelsList(tab, models) {
    const list = tab.querySelector('.platform-models-list');
    if (!list) return;
    if (!models?.length) {
        list.innerHTML = '<p class="text-ui-label text-slate-600 italic">Nenhum modelo listado. Verifique autenticação e se a integração está rodando.</p>';
        syncPlatformCursorSelectors(tab, models || []);
        return;
    }
    list.innerHTML = models.map((model) => {
        const id = model.id || model.name || 'modelo';
        const cursorId = model.cursor_id || model.id;
        const safeId = jsString(id);
        const owned = escapeHtml(model.owned_by || 'cliproxy');
        return `
            <div class="flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl bg-slate-950/50 border border-slate-800/60">
                <div class="min-w-0">
                    <span class="text-sm font-mono text-slate-200 truncate block">${escapeHtml(id)}</span>
                    <span class="text-ui-label text-cyan-400/80 font-mono truncate block">Alias: ${escapeHtml(cursorId)}</span>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                    <span class="text-ui-label text-slate-600 uppercase">${owned}</span>
                    <button type="button" class="platform-copy-alias px-2 py-1 rounded-lg border border-cyan-500/30 text-cyan-300 text-ui-label font-black uppercase hover:bg-cyan-500/10" data-copy-alias="${escapeHtml(cursorId)}" title="Copiar alias">Copiar</button>
                </div>
            </div>`;
    }).join('');
    list.querySelectorAll('.platform-copy-alias').forEach((btn) => {
        btn.addEventListener('click', () => copyPlatformAlias(btn.dataset.copyAlias));
    });
    syncPlatformCursorSelectors(tab, models);
}

function populatePlatformDefaultModelSelect(tab, models, currentValue) {
    const select = tab.querySelector('.platform-proxy-default-model');
    if (!select) return;
    const options = ['<option value="">— Nenhum (não encaminhar) —</option>'];
    (models || []).forEach((model) => {
        const value = model.cursor_id || model.id;
        if (!value) return;
        const label = model.id || model.name || value;
        options.push(
            `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`
        );
    });
    if (currentValue && !(models || []).some(m => (m.cursor_id || m.id) === currentValue)) {
        options.push(
            `<option value="${escapeHtml(currentValue)}">${escapeHtml(currentValue)}</option>`
        );
    }
    if (document.activeElement === select) return;
    select.innerHTML = options.join('');
    select.value = currentValue || '';
    select.dataset.prevValue = currentValue || '';
}

async function fetchModelAliasState() {
    const res = await apiFetch('/model-aliases');
    if (!res.ok) return { aliases: {}, cursor_compatible_names: [] };
    return res.json();
}

function renderPlatformCursorAliases(tab, aliasState) {
    const list = tab.querySelector('.platform-cursor-aliases-list');
    if (!list) return;
    const aliases = aliasState?.aliases || {};
    const entries = Object.entries(aliases);
    if (!entries.length) {
        list.innerHTML = '<li class="text-slate-600 italic">Nenhum alias configurado.</li>';
        return;
    }
    list.innerHTML = entries.map(([alias, target]) => `
        <li class="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-slate-950/40 border border-slate-800/50">
            <span><span class="text-cyan-300">${escapeHtml(alias)}</span> → <span class="text-slate-300">${escapeHtml(target)}</span></span>
            <button type="button" class="text-rose-400 hover:text-rose-300 text-ui-label uppercase font-black" data-alias="${escapeHtml(alias)}">Remover</button>
        </li>`).join('');
    list.querySelectorAll('button[data-alias]').forEach((btn) => {
        btn.addEventListener('click', () => saveModelAlias(btn.dataset.alias, null, tab));
    });
}

function renderLocalCursorAliases(tab, aliasState, modelPath) {
    const list = tab.querySelector('.local-cursor-aliases-list');
    const select = tab.querySelector('.local-cursor-alias-select');
    if (!list || !select) return;
    const normalizedPath = (modelPath || '').replace(/\\/g, '/');
    const aliases = Object.entries(aliasState?.aliases || {}).filter(([, target]) => {
        const normalizedTarget = String(target || '').replace(/\\/g, '/');
        return normalizedTarget === normalizedPath
            || normalizedTarget.split('/').pop() === normalizedPath.split('/').pop();
    });
    if (!aliases.length) {
        list.innerHTML = '<li class="text-slate-600 italic">Nenhum alias configurado para este modelo.</li>';
    } else {
        list.innerHTML = aliases.map(([alias]) => `
            <li class="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-slate-950/40 border border-slate-800/50">
                <span class="text-cyan-300">${escapeHtml(alias)}</span>
                <button type="button" class="text-rose-400 hover:text-rose-300 text-ui-label uppercase font-black" data-local-alias="${escapeHtml(alias)}">Remover</button>
            </li>`).join('');
        list.querySelectorAll('button[data-local-alias]').forEach((button) => {
            button.addEventListener('click', () => saveModelAlias(button.dataset.localAlias, null, tab));
        });
    }
    const names = aliasState?.cursor_compatible_names || [
        'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo', 'o3-mini', 'gpt-5.5'
    ];
    const current = aliases[0]?.[0] || '';
    select.innerHTML = names.map(name => `
        <option value="${escapeHtml(name)}">${escapeHtml(name)}</option>
    `).join('');
    if (current && [...select.options].some(option => option.value === current)) {
        select.value = current;
    }
}

function syncPlatformCursorSelectors(tab, models) {
    const aliasSelect = tab.querySelector('.platform-cursor-alias-select');
    const targetSelect = tab.querySelector('.platform-cursor-target-select');
    if (!aliasSelect || !targetSelect) return;
    const names = state.cursorCompatibleNames?.length
        ? state.cursorCompatibleNames
        : ['gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo', 'o3-mini', 'gpt-5.5'];
    if (!aliasSelect.options.length) {
        aliasSelect.innerHTML = names.map(n => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');
    }
    const currentTarget = targetSelect.value;
    targetSelect.innerHTML = (models || []).map((m) => {
        const id = m.id || m.name;
        return `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`;
    }).join('');
    if (currentTarget && [...targetSelect.options].some(o => o.value === currentTarget)) {
        targetSelect.value = currentTarget;
    }
}

async function saveModelAlias(alias, target, tab) {
    try {
        const res = await apiFetch('/model-aliases', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ alias, target: target || null }),
        });
        if (!res.ok) throw new Error('alias');
        const data = await res.json();
        window.lastConfig = { ...(window.lastConfig || {}), model_aliases: data.aliases };
        if (tab?.querySelector('.local-cursor-aliases-list')) {
            renderLocalCursorAliases(tab, data, tab.dataset.path || '');
        } else if (tab) {
            renderPlatformCursorAliases(tab, data);
        }
        if (target) {
            showToast(`Alias "${alias}" → "${target}". Selecione "${alias}" no cliente.`, 'success');
        } else {
            showToast(`Alias "${alias}" removido.`, 'success');
        }
    } catch {
        showToast('Falha ao salvar alias.', 'error');
    }
}

export async function configureLocalModelAlias(target) {
    const normalizedTarget = (target || '').replace(/\\/g, '/');
    if (!normalizedTarget) return;
    const aliasState = await fetchModelAliasState();
    const aliases = aliasState?.aliases || {};
    const existing = Object.entries(aliases).find(([, value]) => {
        const normalizedValue = String(value || '').replace(/\\/g, '/');
        return normalizedValue === normalizedTarget
            || normalizedValue.split('/').pop() === normalizedTarget.split('/').pop();
    });
    const currentAlias = existing?.[0] || '';
    const alias = await showPrompt(
        currentAlias
            ? 'Alias do modelo local (deixe vazio para remover):'
            : 'Nome do alias para este modelo local:',
        currentAlias,
        { confirmLabel: currentAlias ? 'Salvar' : 'Adicionar' }
    );
    if (alias === null || alias === undefined) return;
    if (!alias.trim() && !currentAlias) return;
    await saveModelAlias(alias.trim() || currentAlias, alias.trim() ? normalizedTarget : null);
    await updateModels();
}

async function copyPlatformAlias(alias) {
    if (!alias) return;
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(alias);
        } else {
            const input = document.createElement('textarea');
            input.value = alias;
            input.setAttribute('readonly', '');
            input.style.position = 'absolute';
            input.style.left = '-9999px';
            document.body.appendChild(input);
            input.select();
            document.execCommand('copy');
            document.body.removeChild(input);
        }
        showToast(`Alias "${alias}" copiado.`, 'success');
    } catch {
        showToast('Falha ao copiar alias.', 'error');
    }
}

async function refreshPlatformCursorSection(tab) {
    const aliasState = await fetchModelAliasState();
    state.cursorCompatibleNames = aliasState.cursor_compatible_names || [];
    renderPlatformCursorAliases(tab, aliasState);
}

async function refreshLocalCursorSection(tab, modelPath) {
    const aliasState = await fetchModelAliasState();
    state.cursorCompatibleNames = aliasState.cursor_compatible_names || [];
    renderLocalCursorAliases(tab, aliasState, modelPath);
}

export function populatePlatformTab(tabId, backendId, detail = null) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    const catalog = (state.lastPlatformList || []).find(p => p.backend_id === backendId) || {};
    const runtime = (state.platforms || []).find(p => p.backend_id === backendId) || {};
    const platform = platformDisplayState(detail || { ...catalog, ...runtime });
    const cfg = window.lastConfig || {};
    const pCfg = (cfg.platform_configs || window.platformConfigs || {})[backendId] || {};
    const smartProxy = cfg.smart_proxy || {};
    const auth = platform.cliproxy_auth || detail?.cliproxy_auth || {};
    const sidecar = detail?.sidecar || state.sidecarStatus || {};

    tab.querySelector('.platform-tab-name').innerText = platform.display_name || platform.name || backendId;
    tab.querySelector('.platform-tab-provider').innerText = (platform.provider || 'platform').toUpperCase();
    tab.querySelector('.platform-info-backend-id').textContent = backendId;
    tab.querySelector('.platform-info-executable').textContent = platform.executable_path || '—';
    tab.querySelector('.platform-info-cliproxy').textContent = platform.cliproxy_executable_path || '—';

    const port = platform.sidecar_port || sidecar.port;
    tab.querySelector('.platform-info-sidecar').textContent = port ? `127.0.0.1:${port}` : '—';
    tab.querySelector('.platform-info-start-time').textContent = formatPlatformTimestamp(platform.start_time);

    const errEl = tab.querySelector('.platform-info-error');
    const errText = platform.last_error || platform.reason || '';
    if (errText) {
        errEl.textContent = errText;
        errEl.classList.remove('hidden');
    } else {
        errEl.classList.add('hidden');
        errEl.textContent = '';
    }

    const authSummary = tab.querySelector('.platform-auth-summary');
    authSummary.innerHTML = `<span class="inline-flex px-2 py-0.5 rounded border ${platformAuthClass(platform)}">${escapeHtml(platformAuthSummary(platform) || 'Status desconhecido')}</span>`;

    const accountsEl = tab.querySelector('.platform-auth-accounts');
    const accounts = auth.accounts || [];
    const accountDetails = auth.account_details || [];
    if (platform.provider === 'ollama-cloud' && accountDetails.length) {
        accountsEl.innerHTML = accountDetails.map(account => `
            <li class="flex items-center gap-2 rounded-lg border border-slate-800/60 bg-slate-950/40 px-2 py-1.5">
                <i class="fas fa-user-circle text-slate-600 shrink-0"></i>
                <span class="min-w-0 flex-1">
                    <span class="block truncate text-slate-400">${escapeHtml(account.label || account.id)}</span>
                    ${account.api_key ? `<span class="block truncate text-slate-600">${escapeHtml(account.api_key)}</span>` : ''}
                </span>
                <button type="button" class="ollama-account-delete w-7 h-7 shrink-0 rounded bg-red-600/10 text-red-400 hover:bg-red-600/20" data-account-id="${escapeHtml(account.id)}" title="Apagar credencial" aria-label="Apagar credencial">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </li>`).join('');
        accountsEl.querySelectorAll('.ollama-account-delete').forEach(button => {
            button.addEventListener('click', () => {
                deleteOllamaCloudAccount(button.dataset.accountId, backendId, tab.id);
            });
        });
    } else {
        accountsEl.innerHTML = accounts.length
            ? accounts.map(a => `<li class="truncate"><i class="fas fa-user-circle text-slate-600 mr-1"></i>${escapeHtml(a)}</li>`).join('')
            : '<li class="text-slate-600 italic">Nenhuma conta autenticada</li>';
    }

    const methods = auth.available_methods || [];
    tab.querySelector('.platform-auth-methods').textContent = methods.length
        ? `Métodos: ${methods.join(', ')} (padrão: ${auth.default_method || 'oauth'})`
        : 'Métodos de login não disponíveis';

    const limitsInfo = PLATFORM_LIMITS_INFO[platform.provider] || 'Limites definidos pela conta autenticada no provedor cloud.';
    tab.querySelector('.platform-limits-info').textContent = limitsInfo;
    tab.querySelector('.platform-limits-parallel').textContent = String(
        pCfg.max_parallel_requests || platform.max_parallel_requests || 1
    );

    const primaryCb = tab.querySelector('.platform-proxy-primary');
    const eligibleCb = tab.querySelector('.platform-proxy-eligible');
    const visionCb = tab.querySelector('.platform-vision-enabled');
    const parallelInput = tab.querySelector('.platform-proxy-parallel');
    const autoStartCb = tab.querySelector('.platform-autostart');
    if (primaryCb && document.activeElement !== primaryCb) {
        primaryCb.checked = smartProxy.primary_backend_id === backendId;
    }
    if (eligibleCb && document.activeElement !== eligibleCb) {
        eligibleCb.checked = pCfg.proxy_eligible === true;
    }
    if (visionCb && document.activeElement !== visionCb) {
        visionCb.checked = pCfg.vision_enabled !== false;
    }
    if (parallelInput && document.activeElement !== parallelInput) {
        parallelInput.value = pCfg.max_parallel_requests || platform.max_parallel_requests || 1;
    }
    if (autoStartCb && document.activeElement !== autoStartCb) {
        autoStartCb.checked = pCfg.auto_start === true;
    }

    if (detail?.available_models) {
        const cursorIds = detail.cursor_model_ids || [];
        const models = detail.available_models.map((m, index) => ({
            ...m,
            cursor_id: cursorIds[index] || m.id,
        }));
        renderPlatformModelsList(tab, models);
        populatePlatformDefaultModelSelect(tab, models, pCfg.default_model || '');
    }

    refreshPlatformTabStatus(tabId, platform, port);
}

export function getPlatformTabActionsHtml(backendId, tabId, isRunning, platform = {}) {
    const safeId = jsString(backendId);
    const provider = jsString(platform.provider || '');
    const displayName = jsString(platform.display_name || platform.name || backendId);
    const deleteAction = platform.provider === 'generic-openai' && platform.account_id
        ? `<button type="button" onclick="deleteGenericOpenAIAccount('${jsString(platform.account_id)}', '${displayName}', '${jsString(tabId)}')" class="px-5 py-2.5 bg-red-600/10 hover:bg-red-600/20 text-red-400 border border-red-500/20 text-ui-body-sm font-black rounded-xl transition-all uppercase tracking-widest active:scale-95"><i class="fas fa-trash-alt"></i> Excluir API</button>`
        : '';
    const authAction = platform.provider === 'generic-openai' ? `manageGenericOpenAIAuth('${safeId}', '${displayName}')` : (platform.provider === 'ollama-cloud' ? `manageOllamaCloudAuth('${safeId}', '${displayName}')` : `startCliproxyAuth('${safeId}', '${provider}', '${displayName}')`);
    if (isRunning) {
        return `
            <button type="button" onclick="stopPlatform('${safeId}')" class="px-5 py-2.5 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/20 text-ui-body-sm font-black rounded-xl transition-all uppercase tracking-widest active:scale-95">
                Encerrar
            </button>
            <button type="button" onclick="${authAction}" class="px-5 py-2.5 bg-amber-600/10 hover:bg-amber-600/20 text-amber-300 border border-amber-500/20 text-ui-body-sm font-black rounded-xl transition-all uppercase tracking-widest active:scale-95">
                <i class="fas fa-key"></i> Conta
            </button>
            ${deleteAction}`;
    }
    const canStart = platform.detected && platform.status !== 'not_ready' && platform.status !== 'missing';
    return `
        <button type="button" ${canStart ? '' : 'disabled'} onclick="startPlatform('${safeId}')" class="px-8 py-3 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:pointer-events-none text-white text-ui-body-sm font-black rounded-2xl active:scale-95 flex items-center gap-3 uppercase tracking-[0.2em] shadow-2xl shadow-violet-600/30 transition-all">
            <i class="fas fa-bolt"></i> Iniciar Integração
        </button>
        <button type="button" onclick="${authAction}" class="px-5 py-2.5 bg-amber-600/10 hover:bg-amber-600/20 text-amber-300 border border-amber-500/20 text-ui-body-sm font-black rounded-xl transition-all uppercase tracking-widest active:scale-95">
            <i class="fas fa-key"></i> Autenticar
        </button>
        ${deleteAction}`;
}

function refreshPlatformTabStatus(tabId, platform, port) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    const statusBadge = tab.querySelector('.tab-status-badge');
    const actions = tab.querySelector('.tab-actions');
    const tabBtn = document.getElementById(`btn-${tabId}`);
    const dot = tabBtn?.querySelector('.tab-status-dot');
    const isRunning = platform.status === 'running';

    if (isRunning) {
        statusBadge.innerText = 'ONLINE';
        statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-emerald-500/40 text-emerald-400 bg-emerald-500/5';
        if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981] animate-pulse shrink-0 transition-all duration-500';
    } else if (platform.status === 'not_ready' || platform.status === 'missing') {
        statusBadge.innerText = 'INDISPONIVEL';
        statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-rose-500/40 text-rose-400 bg-rose-500/5';
        if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0 transition-all duration-500';
    } else {
        statusBadge.innerText = 'OFFLINE';
        statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-slate-700/50 text-slate-500';
        if (dot) dot.className = 'tab-status-dot w-1.5 h-1.5 rounded-full bg-slate-700 shrink-0 transition-all duration-500';
    }

    if (actions) {
        actions.innerHTML = getPlatformTabActionsHtml(platform.backend_id, tabId, isRunning, platform);
    }

    if (state.currentTabId === tabId) {
        if (isRunning && port) {
            attachTabLogs(tabId, port, {
                force: false,
                sessionKey: `platform:${port}:${platform.start_time ?? 0}`,
            });
        } else {
            detachTabLogs();
            const box = tab.querySelector('.tab-log-box');
            if (box) {
                box.innerHTML = '';
                box.dataset.connecting = '1';
                appendPlatformLogPlaceholder(box, isRunning ? 'Aguardando logs do sidecar...' : 'Inicie a integração para ver requisições.');
            }
        }
    }
}

function appendPlatformLogPlaceholder(box, text) {
    if (!box) return;
    box.innerHTML = `<div class="text-slate-600 text-ui-label italic border-l border-slate-800 pl-3">${escapeHtml(text)}</div>`;
    delete box.dataset.connecting;
}

export async function loadPlatformTabDetails(tabId, backendId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    const list = tab.querySelector('.platform-models-list');
    if (list) {
        list.innerHTML = '<p class="text-ui-label text-slate-600 italic"><i class="fas fa-sync animate-spin mr-1"></i> Carregando modelos...</p>';
    }
    try {
        const res = await apiFetch(`/platforms/${encodeURIComponent(backendId)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            if (list) list.innerHTML = `<p class="text-ui-label text-rose-400">${escapeHtml(err.detail || 'Falha ao carregar')}</p>`;
            return;
        }
        const detail = await res.json();
        state.sidecarStatus = detail.sidecar || state.sidecarStatus;
        populatePlatformTab(tabId, backendId, detail);
        await refreshPlatformCursorSection(tab);
    } catch {
        if (list) list.innerHTML = '<p class="text-ui-label text-rose-400">Erro de rede ao carregar detalhes.</p>';
    }
}

export function refreshPlatformTabsFromStatus() {
    state.activeTabs
        .filter(t => t.kind === 'platform')
        .forEach((tab) => {
            populatePlatformTab(tab.id, tab.backendId);
        });
}

function bindTabListeners(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    
    const path = tab.dataset.path;
    
    tab.querySelector('.tab-thinking-toggle')?.addEventListener('change', (e) => persistThinkingEnabled(path, e.target.checked));
    tab.querySelector('.tab-mtp-toggle')?.addEventListener('change', () => syncMtpDraftTokensState(tabId));
    tab.querySelector('.tab-context-size')?.addEventListener('change', () => syncContextSizeCustomVisibility(tabId));
    tab.querySelector('.tab-context-size-custom')?.addEventListener('input', () => updateTotal(tabId));

    tab.querySelector('.tab-llama-bin')?.addEventListener('change', () => {
        syncTurboquantPanelVisibility(tabId);
        persistLlamaBinSettings(path, tabId);
    });
    tab.querySelector('.tab-turboquant-preset')?.addEventListener('change', (e) => {
        if (e.target.value !== 'custom') {
            applyTurboquantPreset(tabId, e.target.value);
        }
        persistLlamaBinSettings(path, tabId);
    });
    tab.querySelector('.tab-turbo-cache-k')?.addEventListener('change', () => {
        syncTurboquantToCacheFields(tabId);
        const presetEl = tab.querySelector('.tab-turboquant-preset');
        if (presetEl) {
            const k = tab.querySelector('.tab-turbo-cache-k')?.value;
            const v = tab.querySelector('.tab-turbo-cache-v')?.value;
            presetEl.value = detectTurboquantPreset(k, v);
        }
        persistLlamaBinSettings(path, tabId);
    });
    tab.querySelector('.tab-turbo-cache-v')?.addEventListener('change', () => {
        syncTurboquantToCacheFields(tabId);
        const presetEl = tab.querySelector('.tab-turboquant-preset');
        if (presetEl) {
            const k = tab.querySelector('.tab-turbo-cache-k')?.value;
            const v = tab.querySelector('.tab-turbo-cache-v')?.value;
            presetEl.value = detectTurboquantPreset(k, v);
        }
        persistLlamaBinSettings(path, tabId);
    });

    tab.querySelector('.tab-cache-type-k')?.addEventListener('change', () => {
        if (isTurboquantBin(getSelectedLlamaBin(tabId))) {
            syncMainCacheToTurboFields(tabId);
            persistLlamaBinSettings(path, tabId);
        }
    });
    tab.querySelector('.tab-cache-type-v')?.addEventListener('change', () => {
        if (isTurboquantBin(getSelectedLlamaBin(tabId))) {
            syncMainCacheToTurboFields(tabId);
            persistLlamaBinSettings(path, tabId);
        }
    });
    
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
        tab.querySelector('.tab-log-box').dataset.connecting = '0';
        const sizeEl = tab.querySelector('.tab-log-size');
        if (sizeEl) sizeEl.innerText = '0 KB';
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
        flash_attn: tab.querySelector('.tab-pin-flash-attn')?.checked ?? false,
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
        flash_attn: '.tab-pin-flash-attn',
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
    const normalized = path.replace(/\\/g, '/');

    saveScreenSnapshot(path, tabId);

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
    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-amber-500/40 text-amber-400 bg-amber-500/5';

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
                mtp_model_path: getSelectedMtpForModel(normalized),
                numa_enabled: tab.querySelector('.tab-numa-toggle').checked,
                flash_attn_enabled: tab.querySelector('.tab-flash-attn-toggle').checked,
                split_mode: tab.querySelector('.tab-split-mode').value,
                llama_server_bin: getSelectedLlamaBin(tabId),
                turboquant_preset: getTurboquantPreset(tabId),
                auto_balance: true,
                smart_calibration: true,
                pinned_fields: pinnedFields,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            showToast("Erro na calibração: " + (err.detail || "Falha"), 'error');
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
        showToast("Erro de rede.", 'error');
        state.autoBalancePending = false;
        state.autoBalanceTabId = null;
        state.autoBalanceSeenActive = false;
        hideAutoBalanceProgress(tabId);
        window.updateStatus();
    }
}

export function saveScreenSnapshot(path, tabId) {
    const tab = document.getElementById(tabId);
    const payload = collectStartPayloadFromTab(path, tabId);
    if (tab && payload) {
        tab.dataset.screenSnapshot = JSON.stringify(payload);
    }
}

export function restoreScreenSnapshot(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab?.dataset.screenSnapshot) return;
    applyScreenConfigToTab(tabId, JSON.parse(tab.dataset.screenSnapshot));
}

function applyScreenConfigToTab(tabId, cfg) {
    const tab = document.getElementById(tabId);
    if (!tab || !cfg) return;
    const path = (cfg.path || tab.dataset.path || '').replace(/\\/g, '/');
    if (!path) return;
    window.modelConfigs[path] = { ...(window.modelConfigs[path] || {}), ...cfg };
    applyModelConfig(path, tabId);
    applyPinnedFieldsToTab(tabId, cfg.pinned_fields);
}

function getScreenBaselineConfig(tabId) {
    const tab = document.getElementById(tabId);
    if (tab?.dataset.screenSnapshot) {
        return JSON.parse(tab.dataset.screenSnapshot);
    }
    const path = (tab?.dataset.path || '').replace(/\\/g, '/');
    return collectStartPayloadFromTab(path, tabId) || window.modelConfigs[path] || {};
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
                <span class="text-ui-caption text-slate-500 uppercase">${label}</span>
                <div class="flex items-center gap-2 mt-1">
                    <span class="line-through opacity-40">${oldVal}</span>
                    <i class="fas fa-arrow-right text-ui-caption text-blue-500"></i>
                    <span class="font-bold ${changed ? 'text-emerald-400' : 'text-slate-300'}">${newVal}</span>
                </div>
            </div>
        `;
    };

    const baseline = getScreenBaselineConfig(tabId);
    
    details.innerHTML = `
        ${formatDiff('Contexto', (baseline.context_size || 0)/1024 + 'K', (proposal.context_size || 0)/1024 + 'K')}
        ${formatDiff('Slots', baseline.parallel_slots || 1, proposal.parallel_slots || 1)}
        ${formatDiff('Batch', baseline.batch_size || 2048, proposal.batch_size || 2048)}
        ${formatDiff('Cache', baseline.cache_type_k || 'f16', proposal.cache_type_k || 'f16')}
    `;

    area.classList.remove('hidden');
}

export function hideProposedConfig(tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;
    tab.querySelector('.tab-proposed-config')?.classList.add('hidden');
    delete tab.dataset.proposal;
}

function clearScreenSnapshot(tabId) {
    const tab = document.getElementById(tabId);
    if (tab) delete tab.dataset.screenSnapshot;
}

function collectStartPayloadFromTab(path, tabId, { autoBalanceProfile = false } = {}) {
    const tab = document.getElementById(tabId);
    if (!tab) return null;

    const normalized = path.replace(/\\/g, '/');
    const contextSize = getContextSize(tabId);
    if (contextSize === null) return null;

    const cacheTypes = getEffectiveCacheTypes(tabId);
    const mmprojDisabled = _isMmprojDisabledForModel(normalized);
    const visionEnabled = window.modelConfigs[normalized]?.vision_enabled !== false;

    return {
        path: normalized,
        mmproj_path: getSelectedMmprojForModel(normalized) || null,
        // This flag represents the explicit "Sem visão" selection. The
        // checkbox is sent separately and is combined by /start only for the
        // running llama-server process.
        mmproj_disabled: mmprojDisabled,
        vision_enabled: visionEnabled,
        gpu_weights: collectDeviceWeightsFromUI(tabId),
        context_size: contextSize,
        parallel_slots: parseInt(tab.querySelector('.tab-parallel-slots').value, 10) || 1,
        batch_size: parseInt(tab.querySelector('.tab-batch-size').value, 10) || 2048,
        ubatch_size: parseInt(tab.querySelector('.tab-ubatch-size').value, 10) || 512,
        cache_type_k: cacheTypes.cache_type_k,
        cache_type_v: cacheTypes.cache_type_v,
        numa_enabled: tab.querySelector('.tab-numa-toggle').checked,
        flash_attn_enabled: tab.querySelector('.tab-flash-attn-toggle').checked,
        threads: parseInt(tab.querySelector('.tab-threads').value, 10) || 0,
        threads_batch: parseInt(tab.querySelector('.tab-threads-batch').value, 10) || 0,
        split_mode: tab.querySelector('.tab-split-mode').value,
        llama_server_bin: getSelectedLlamaBin(tabId),
        turboquant_preset: getTurboquantPreset(tabId),
        auto_balance: false,
        auto_balance_profile: autoBalanceProfile,
        // A normal start is an explicit user launch.  OOM recovery must not
        // rewrite the user's GPU percentages and persist a different layout.
        manual_gpu_override: !autoBalanceProfile,
        pinned_fields: collectPinnedFieldsFromTab(tabId),
        thinking_enabled: tab.querySelector('.tab-thinking-toggle').checked,
        mtp_enabled: tab.querySelector('.tab-mtp-toggle').checked,
        mtp_draft_tokens: getMtpDraftTokens(tabId),
        mtp_model_path: getSelectedMtpForModel(normalized),
    };
}

export async function applyProposedConfig(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab?.dataset.proposal) return;

    const normalized = path.replace(/\\/g, '/');
    const proposal = JSON.parse(tab.dataset.proposal);

    applyScreenConfigToTab(tabId, proposal);

    const payload = collectStartPayloadFromTab(normalized, tabId, {
        autoBalanceProfile: true,
    });
    if (!payload) { showToast('Contexto inválido', 'error'); return; }

    const weightValidation = validateDeviceWeights(payload.gpu_weights);
    if (!weightValidation.ok) { showToast(weightValidation.message, 'error'); return; }

    window.modelConfigs[normalized] = {
        ...(window.modelConfigs[normalized] || {}),
        ...payload,
    };

    hideProposedConfig(tabId);
    clearScreenSnapshot(tabId);

    const statusBadge = tab.querySelector('.tab-status-badge');
    statusBadge.innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2"></i> SALVANDO...';
    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-amber-500/40 text-amber-400';

    try {
        const res = await apiFetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            showToast('Erro ao salvar/iniciar: ' + (err.detail || 'Falha'), 'error');
            window.updateStatus();
            return;
        }

        const startData = await res.json();
        if (startData.port) {
            state.currentActivePort = startData.port;
            attachTabLogs(tabId, startData.port, {
                force: true,
                sessionKey: `${startData.port}:${startData.start_time ?? Date.now()}`,
            });
        }

        await window.updateModels?.();
        await window.updateStatus();
        const openTab = state.activeTabs.find(t => t.kind === 'platform' && t.backendId === backendId);
        if (openTab) {
            await loadPlatformTabDetails(openTab.id, backendId);
        }
    } catch (e) {
        showToast('Erro de rede ao salvar configuração.', 'error');
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
        attachTabLogs(tabId);
        syncTabLogPanelHeight(tabId);
    }
}

export function closeTab(tabId) {
    unbindTabLogPanelHeightSync(tabId);

    const btn = document.getElementById(`btn-${tabId}`);
    const content = document.getElementById(tabId);
    const closedPath = content?.dataset.path?.replace(/\\/g, '/');

    if (btn) btn.remove();
    if (content) content.remove();
    
    state.activeTabs = state.activeTabs.filter(t => t.id !== tabId);

    if (closedPath) refreshTabLabelsForPath(closedPath);
    
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
    // Preserve both the persisted flag and the legacy sentinel.
    if (cfg.mmproj_disabled || saved === '__no_vision__') return null;
    if (saved && candidates.includes(saved)) return saved;
    return candidates[0];
}

export function buildModelVisionControlsHtml(model, modelJs, visionEnabled = true) {
    const candidates = model.mmproj_candidates || [];
    const safePath = modelJs.replace(/'/g, "\\'");
    const safeModelPath = jsString(modelJs);
    const importBtn = `<button type="button" onclick="event.stopPropagation(); openVisionImportModal('${safePath}')" class="vision-import-btn w-8 h-8 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-violet-400 hover:bg-violet-500/20 transition-all" title="Importar projetor de visão" aria-label="Importar projetor de visão"><i class="fas fa-eye text-ui-label"></i></button>`;
    const visionCheckbox = `<label class="flex items-center gap-1 cursor-pointer shrink-0" onclick="event.stopPropagation()" title="Permitir visão neste modelo local"><span class="text-ui-label font-black text-slate-600 uppercase">Vision</span><input type="checkbox" class="model-vision-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-cyan-600" ${visionEnabled ? 'checked' : ''} onclick="setLocalVisionEnabled(this, '${safeModelPath}')"></label>`;
    if (!candidates.length) return `${visionCheckbox}${importBtn}`;

    const selected = resolveMmprojPath(model);
    let options = '';
    // "Sem visão" option at the top
    const noVisionSelected = selected === null ? ' selected' : '';
    options += `<option value="__no_vision__"${noVisionSelected}>Sem visão</option>`;
    const optionsList = candidates.map((candidate) => {
        const name = escapeHtml(candidate.split('/').pop());
        const value = escapeHtml(candidate);
        const selectedAttr = candidate === selected ? ' selected' : '';
        return `<option value="${value}" class="bg-slate-900"${selectedAttr}>${name}</option>`;
    }).join('');
    options += optionsList;

    const selectAttrs = MMproj_SELECT_ATTRS.replace('__PATH__', safePath);
    const hiddenClass = visionEnabled ? '' : ' hidden';
    const selectWrap = `<span class="model-mmproj-control${hiddenClass}"><select data-mmproj-for="${escapeHtml(modelJs)}" class="model-mmproj-select bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2 py-1 text-ui-label font-bold focus:ring-2 focus:ring-violet-500/50 outline-none transition-all cursor-pointer min-w-[7rem] max-w-[11rem]" ${selectAttrs} title="Projetor de visão para este modelo" aria-label="Projetor de visão para este modelo">${options}</select></span>`;
    return `${visionCheckbox}${importBtn}${selectWrap}`;
}

export function getSelectedMmprojForModel(modelPath) {
    const normalized = modelPath.replace(/\\/g, '/');
    const selects = document.querySelectorAll('select[data-mmproj-for]');
    for (const select of selects) {
        if (select.getAttribute('data-mmproj-for') === normalized && select.value) {
            if (select.value === '__no_vision__') return null;
            return select.value;
        }
    }
    const cfg = window.modelConfigs[normalized];
    if (cfg?.mmproj_disabled || cfg?.mmproj_path === '__no_vision__') return null;
    return cfg?.mmproj_path || null;
}

export function resolveMtpModelPath(model) {
    const candidates = model.mtp_candidates || [];
    const modelPath = model.path.replace(/\\/g, '/');
    const cfg = window.modelConfigs[modelPath] || model.last_config || {};
    if (cfg.mtp_model_path && candidates.includes(cfg.mtp_model_path)) {
        return cfg.mtp_model_path;
    }
    return candidates[0] || null;
}

export function buildModelMtpControlsHtml(model, modelPath) {
    const candidates = model.mtp_candidates || [];
    const safePath = modelPath.replace(/'/g, "\\'");
    const importBtn = `<button type="button" onclick="event.stopPropagation(); openMtpImportModal('${safePath}')" class="mtp-import-btn w-8 h-8 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-amber-400 hover:bg-amber-500/20 transition-all" title="Importar modelo draft MTP" aria-label="Importar modelo draft MTP"><i class="fas fa-bolt text-ui-label"></i></button>`;
    if (!candidates.length) return importBtn;

    const selected = resolveMtpModelPath(model);
    const options = [
        '<option value="">Sem MTP externo</option>',
        ...candidates.map(candidate => {
            const selectedAttr = candidate === selected ? ' selected' : '';
            return `<option value="${escapeHtml(candidate)}" class="bg-slate-900"${selectedAttr}>${escapeHtml(candidate.split('/').pop())}</option>`;
        }),
    ].join('');
    return `${importBtn}<select data-mtp-for="${escapeHtml(modelPath)}" class="model-mtp-select bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2 py-1 text-ui-label font-bold focus:ring-2 focus:ring-amber-500/50 outline-none transition-all cursor-pointer min-w-[7rem] max-w-[11rem]" onmousedown="event.stopPropagation()" onpointerdown="event.stopPropagation()" onclick="event.stopPropagation()" onchange="onMtpModelChange('${safePath}', this)" title="Draft MTP deste modelo" aria-label="Draft MTP deste modelo">${options}</select>`;
}

export function getSelectedMtpForModel(modelPath) {
    const normalized = modelPath.replace(/\\/g, '/');
    for (const select of document.querySelectorAll('select[data-mtp-for]')) {
        if (select.getAttribute('data-mtp-for') === normalized) return select.value || null;
    }
    return window.modelConfigs[normalized]?.mtp_model_path || null;
}

function _isMmprojDisabledForModel(modelPath) {
    const cfg = window.modelConfigs[modelPath];
    return Boolean(cfg?.vision_enabled === false || cfg?.mmproj_disabled || cfg?.mmproj_path === '__no_vision__');
}

function onVisionModalKeydown(event) {
    if (event.key === 'Escape') closeVisionImportModal();
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
    document.addEventListener('keydown', onVisionModalKeydown);
}

export function closeVisionImportModal() {
    const modal = document.getElementById('vision-import-modal');
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    document.removeEventListener('keydown', onVisionModalKeydown);
}

export async function submitVisionImport(event) {
    event.preventDefault();
    const modelPath = document.getElementById('vision-import-model-path')?.value.trim();
    const url = document.getElementById('vision-import-url')?.value.trim();
    if (!modelPath || !url) return;
    try {
        const res = await apiFetch('/downloads', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url, model_path: modelPath}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast('Erro ao iniciar download: ' + (err.detail || 'Erro desconhecido'), 'error');
            return;
        }
        closeVisionImportModal();
        window.updateDownloads();
        window.updateModels();
    } catch (e) {
        showToast('Erro de rede ao iniciar download do projetor.', 'error');
    }
}

function onMtpModalKeydown(event) {
    if (event.key === 'Escape') closeMtpImportModal();
}

export function openMtpImportModal(modelPath) {
    const modal = document.getElementById('mtp-import-modal');
    const pathInput = document.getElementById('mtp-import-model-path');
    const urlInput = document.getElementById('mtp-import-url');
    if (!modal || !pathInput || !urlInput) return;
    pathInput.value = modelPath;
    urlInput.value = '';
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    urlInput.focus();
    document.addEventListener('keydown', onMtpModalKeydown);
}

export function closeMtpImportModal() {
    const modal = document.getElementById('mtp-import-modal');
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    document.removeEventListener('keydown', onMtpModalKeydown);
}

export async function submitMtpImport(event) {
    event.preventDefault();
    const modelPath = document.getElementById('mtp-import-model-path')?.value.trim();
    const url = document.getElementById('mtp-import-url')?.value.trim();
    if (!modelPath || !url) return;
    try {
        const res = await apiFetch('/downloads', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url, model_path: modelPath, asset_type: 'mtp'}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast('Erro ao iniciar download MTP: ' + (err.detail || 'Erro desconhecido'), 'error');
            return;
        }
        closeMtpImportModal();
        window.updateDownloads();
        window.updateModels();
    } catch (e) {
        showToast('Erro de rede ao iniciar download MTP.', 'error');
    }
}

function mergeModelConfigFromServer(modelPath, lastConfig) {
    if (!lastConfig) return;
    const local = window.modelConfigs[modelPath] || {};
    const merged = { ...lastConfig, ...local };
    // A running instance can report the projector it was started with before
    // the persisted preference is loaded. Explicit Vision opt-outs from the
    // server must win that transient runtime value after a page reload.
    if (lastConfig.mmproj_disabled || lastConfig.mmproj_path === '__no_vision__') {
        merged.mmproj_path = lastConfig.mmproj_path;
        merged.mmproj_disabled = lastConfig.mmproj_disabled;
    }
    if (lastConfig.vision_enabled === false) {
        merged.vision_enabled = false;
    }
    window.modelConfigs[modelPath] = merged;
}

function isMmprojSelectFocused() {
    // Também evita re-render enquanto o usuário edita o limite de paralelismo
    return document.activeElement?.matches?.('.model-mmproj-select, .model-mtp-select, .proxy-max-parallel');
}

function patchPlatformListItems(platforms, cfg) {
    for (const raw of platforms || []) {
        const platform = platformDisplayState(raw);
        const backendId = platform.backend_id;
        const el = document.getElementById(platformDomId(backendId));
        if (!el) continue;
        const isRunning = platform.status === 'running';
        el.classList.toggle('border-emerald-500/50', isRunning);
        el.classList.toggle('bg-emerald-500/5', isRunning);
        el.classList.toggle('border-slate-700/50', !isRunning);
        el.classList.toggle('bg-slate-800/40', !isRunning);

        const pCfg = (cfg.platform_configs || window.platformConfigs || {})[backendId] || {};
        const primaryCb = el.querySelector('.proxy-primary-checkbox');
        if (primaryCb && document.activeElement !== primaryCb) {
            primaryCb.checked = (cfg.smart_proxy || {}).primary_backend_id === backendId;
        }
        const eligibleCb = el.querySelector('.proxy-eligible-checkbox');
        if (eligibleCb && document.activeElement !== eligibleCb) {
            eligibleCb.checked = pCfg.proxy_eligible === true;
        }
        const visionCb = el.querySelector('.platform-vision-checkbox');
        if (visionCb && document.activeElement !== visionCb) {
            visionCb.checked = pCfg.vision_enabled !== false;
        }
        const autoStartCb = el.querySelector('.platform-autostart-checkbox');
        if (autoStartCb && document.activeElement !== autoStartCb) {
            autoStartCb.checked = pCfg.auto_start === true;
        }
        const parallelInput = el.querySelector('.proxy-max-parallel');
        if (parallelInput && document.activeElement !== parallelInput) {
            parallelInput.value = pCfg.max_parallel_requests || platform.max_parallel_requests || 1;
        }
    }
}

function patchModelListItems(models, cfg) {
    for (const model of models || []) {
        const mJs = model.path.replace(/\\/g, '/');
        const el = document.getElementById(`lib-${model.id}`);
        if (!el) continue;

        const isDefault = (cfg.default_models || []).includes(mJs) || cfg.default_model === mJs;
        const status = (state.activeInstances || []).find(
            i => (i.model_path || '').replace(/\\/g, '/') === mJs
        );
        el.classList.remove(
            'border-emerald-500/50', 'bg-emerald-500/5',
            'border-slate-700/50', 'bg-slate-800/40',
            'active-selection',
        );
        // classList.add rejeita tokens com espaço (InvalidCharacterError) — separar.
        if (status) {
            el.classList.add('border-emerald-500/50', 'bg-emerald-500/5');
        } else {
            el.classList.add('border-slate-700/50', 'bg-slate-800/40');
        }
        if (state.currentSelectedModel === mJs) {
            el.classList.add('active-selection');
        }

        const autoStart = el.querySelector('.autostart-checkbox');
        if (autoStart) autoStart.checked = isDefault;

        // Mantém os controles do proxy em sincronia no patch incremental
        const primaryCb = el.querySelector('.proxy-primary-checkbox');
        if (primaryCb && document.activeElement !== primaryCb) {
            primaryCb.checked = (cfg.smart_proxy || {}).primary_model_path === mJs;
        }
        const mCfg = (cfg.model_configs || {})[mJs] || {};
        const effectiveCfg = window.modelConfigs[mJs] || mCfg;
        const visionCb = el.querySelector('.model-vision-checkbox');
        const visionEnabled = effectiveCfg.vision_enabled !== false;
        if (visionCb && document.activeElement !== visionCb) {
            visionCb.checked = visionEnabled;
        }
        const mmprojControl = el.querySelector('.model-mmproj-control');
        if (mmprojControl) mmprojControl.classList.toggle('hidden', !visionEnabled);
        const eligibleCb = el.querySelector('.proxy-eligible-checkbox');
        if (eligibleCb && document.activeElement !== eligibleCb) {
            eligibleCb.checked = mCfg.proxy_eligible !== false;
        }
        const parallelInput = el.querySelector('.proxy-max-parallel');
        if (parallelInput && document.activeElement !== parallelInput) {
            parallelInput.value = mCfg.max_parallel_requests || 1;
        }
    }
    patchPlatformListItems(state.lastPlatformList || [], cfg);
}

function buildPlatformCardHtml(rawPlatform, cfg) {
    const platform = platformDisplayState(rawPlatform);
    if (platform.provider === 'generic-openai' && !platform.account_id) {
        return buildGenericOpenAICardsHtml(platform);
    }
    const backendId = platform.backend_id;
    const safeBackendId = jsString(backendId);
    const status = platform.status || (platform.detected ? 'detected' : 'missing');
    const isRunning = status === 'running';
    const isBusy = platformActions.has(backendId);
    const canStart = !isBusy && !isRunning && platform.detected && status !== 'not_ready' && status !== 'missing';
    const canStop = !isBusy && isRunning;
    const statusClass = platformStatusClass(status);
    const statusLabel = PLATFORM_STATUS_LABELS[status] || String(status || '').toUpperCase();
    const reason = platform.last_error || platform.reason || (
        platform.detected ? '' : 'Aplicativo nao encontrado nesta maquina.'
    );
    const pCfg = (cfg.platform_configs || window.platformConfigs || {})[backendId] || {};
    const platformAutoStart = pCfg.auto_start === true;
    const isProxyPrimary = (cfg.smart_proxy || {}).primary_backend_id === backendId;
    const isProxyEligible = pCfg.proxy_eligible === true;
    const isVisionEnabled = pCfg.vision_enabled !== false;
    const proxyMaxParallel = pCfg.max_parallel_requests || platform.max_parallel_requests || 1;
    const runningClass = isRunning ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-slate-700/50 bg-slate-800/40';
    const authSummary = platformAuthSummary(platform);
    const actionHtml = isRunning
        ? `<button type="button" ${canStop ? '' : 'disabled'} onclick="event.stopPropagation(); stopPlatform('${safeBackendId}')" title="Parar integração" aria-label="Parar integração" class="w-8 h-8 flex items-center justify-center rounded bg-red-600/10 text-red-400 hover:bg-red-600/20 disabled:opacity-40 disabled:pointer-events-none"><i class="fas fa-stop text-ui-label"></i></button>`
        : `<button type="button" ${canStart ? '' : 'disabled'} onclick="event.stopPropagation(); startPlatform('${safeBackendId}')" title="Iniciar integração" aria-label="Iniciar integração" class="w-8 h-8 flex items-center justify-center rounded bg-blue-600/10 text-blue-300 hover:bg-blue-600/20 disabled:opacity-40 disabled:pointer-events-none"><i class="fas fa-bolt text-ui-label"></i></button>`;
    const authButtonHtml = buildPlatformAuthButton(platform, safeBackendId);

    return `
            <div id="${platformDomId(backendId)}" class="model-item-container platform-card group p-3 rounded-xl border transition-all cursor-pointer ${runningClass}"
                 data-backend-id="${escapeHtml(backendId)}" data-backend-type="platform"
                 title="Clique para abrir · Ctrl+clique para nova aba"
                 onclick="selectPlatformFromEvent(event, '${safeBackendId}')"
                 onauxclick="selectPlatformFromEvent(event, '${safeBackendId}')">
                <div class="flex items-start justify-between gap-2 overflow-hidden">
                    <div class="flex-1 min-w-0">
                        <p class="model-name text-ui-body font-bold text-slate-100 truncate">${escapeHtml(platform.display_name || platform.name || backendId)}</p>
                        <p class="text-ui-label text-slate-400 font-mono uppercase truncate mt-0.5">${escapeHtml(platform.provider || 'platform')}</p>
                    </div>
                    ${isRunning ? '<div class="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981]"></div>' : ''}
                </div>
                <div class="flex items-center justify-between gap-2 mt-3">
                    <span class="px-2 py-1 rounded border text-ui-label font-black uppercase tracking-widest ${statusClass}">${escapeHtml(statusLabel)}</span>
                    <div class="flex items-center gap-1">${authButtonHtml}${isBusy ? '<i class="fas fa-sync animate-spin text-blue-300 text-ui-label"></i>' : ''}${actionHtml}</div>
                </div>
                ${authSummary ? `<p class="mt-2 text-ui-label leading-snug"><span class="inline-flex px-2 py-0.5 rounded border ${platformAuthClass(platform)}">${escapeHtml(authSummary)}</span></p>` : ''}
                ${reason ? `<p class="mt-2 text-ui-label text-slate-500 leading-snug">${escapeHtml(reason)}</p>` : ''}
                <div class="flex items-center justify-end mt-3 pt-2 border-t border-slate-700/30" onclick="event.stopPropagation()">
                    <label class="flex items-center gap-1.5 cursor-pointer">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Auto-Start</span>
                        <input type="checkbox" class="platform-autostart-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-blue-600" ${platformAutoStart ? 'checked' : ''} onclick="setPlatformAutoStart(this, '${safeBackendId}')">
                    </label>
                </div>
                <div class="proxy-model-controls flex flex-wrap items-center gap-x-3 gap-y-1.5 mt-2 pt-2 border-t border-slate-700/30 min-w-0" onclick="event.stopPropagation()">
                    <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Backend principal exposto pela API no Modo Proxy Inteligente">
                        <span class="text-ui-label font-black text-violet-400/80 uppercase">Principal</span>
                        <input type="checkbox" class="proxy-primary-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-violet-600" data-backend-id="${escapeHtml(backendId)}" ${isProxyPrimary ? 'checked' : ''} onclick="setProxyPrimary(this, null, '${safeBackendId}')">
                    </label>
                    <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Usar como backend no proxy inteligente">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Proxy</span>
                        <input type="checkbox" class="proxy-eligible-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-violet-600" ${isProxyEligible ? 'checked' : ''} onclick="setProxyEligible(this, null, '${safeBackendId}')">
                    </label>
                    <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Permitir requisições com imagens nesta plataforma">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Vision</span>
                        <input type="checkbox" class="platform-vision-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-cyan-600" ${isVisionEnabled ? 'checked' : ''} onclick="setPlatformVisionEnabled(this, '${safeBackendId}')">
                    </label>
                    <label class="flex items-center gap-1 shrink-0 ml-auto" title="Capacidade paralela inicial; cresce automaticamente sob pressão">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Paralelo</span>
                        <input type="number" min="1" max="16" value="${proxyMaxParallel}" class="proxy-max-parallel w-9 px-0.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-ui-label text-slate-300 text-center outline-none" onchange="setProxyMaxParallel(this, null, '${safeBackendId}')">
                    </label>
                </div>
            </div>`;
}

function buildGenericOpenAICardsHtml(platform) {
    const creatorCard = `
        <div class="model-item-container platform-card group p-3 rounded-xl border border-dashed border-amber-500/40 bg-amber-500/5 transition-all cursor-pointer hover:bg-amber-500/10"
             data-generic-openai-creator="true"
             title="Cadastrar uma API compatível com OpenAI"
             onclick="manageGenericOpenAIAuth('platform:generic-openai', 'Cadastrar API Genérica')">
            <div class="flex items-center justify-between gap-3">
                <div class="min-w-0">
                    <p class="model-name text-ui-body font-bold text-slate-100 truncate">Cadastrar API Genérica</p>
                    <p class="text-ui-label text-slate-400 mt-0.5">Adicione um endpoint compatível com OpenAI</p>
                </div>
                <span class="w-9 h-9 shrink-0 rounded-lg bg-amber-500/15 text-amber-300 flex items-center justify-center">
                    <i class="fas fa-plus"></i>
                </span>
            </div>
        </div>`;

    return creatorCard;
}

function buildModelListHtml(models, cfg, platforms = []) {
    const localHtml = models.map(m => {
        const m_js = m.path.replace(/\\/g, '/');
        if (m.last_config && !tabHasPendingProposal(m_js)) {
            mergeModelConfigFromServer(m_js, m.last_config);
        }

        const isDefault = (cfg.default_models || []).includes(m_js) || cfg.default_model === m_js;
        const status = (state.activeInstances || []).find(i => (i.model_path || '').replace(/\\/g, '/') === m_js);
        const runningClass = status ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-slate-700/50 bg-slate-800/40';
        const selectedClass = state.currentSelectedModel === m_js ? 'active-selection' : '';

        const mCfg = (cfg.model_configs || {})[m_js] || {};
        const localAlias = Object.entries(cfg.model_aliases || {}).find(([, target]) => {
            const normalizedTarget = String(target || '').replace(/\\/g, '/');
            return normalizedTarget === m_js
                || normalizedTarget.split('/').pop() === m_js.split('/').pop();
        })?.[0] || '';
        const isProxyPrimary = (cfg.smart_proxy || {}).primary_model_path === m_js;
        const isProxyEligible = mCfg.proxy_eligible !== false;
        const proxyMaxParallel = mCfg.max_parallel_requests || 1;

        const safePath = m_js.replace(/'/g, "\\'");
        const visionControls = buildModelVisionControlsHtml(
            m,
            m_js,
            (window.modelConfigs[m_js] || mCfg).vision_enabled !== false,
        );
        const mtpControls = buildModelMtpControlsHtml(m, m_js);
        return `
            <div id="lib-${m.id}" class="model-item-container group p-3 rounded-xl border transition-all cursor-pointer ${runningClass} ${selectedClass}" 
                 title="Clique para abrir · Ctrl+clique para nova aba"
                 onclick="selectModelFromEvent(event, '${safePath}', '${m.id}')"
                 onauxclick="selectModelFromEvent(event, '${safePath}', '${m.id}')">
                <div class="flex items-start justify-between gap-2 overflow-hidden">
                    <div class="flex-1 min-w-0">
                        <p class="model-name text-ui-body font-bold text-slate-100 truncate">${escapeHtml(m.name)}</p>
                        <p class="text-ui-label text-slate-400 font-mono uppercase truncate mt-0.5">${escapeHtml(m.dir)}</p>
                        ${localAlias ? `<p class="text-ui-label text-cyan-400/80 font-mono truncate mt-1">Alias: ${escapeHtml(localAlias)}</p>` : ''}
                    </div>
                    ${status ? '<div class="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981]"></div>' : ''}
                </div>
                <div class="flex items-center justify-between mt-3 pt-2 border-t border-slate-700/30">
                    <div class="flex items-center gap-1 flex-wrap">
                        <button onclick="event.stopPropagation(); renameModel('${safePath}')" title="Renomear modelo" aria-label="Renomear modelo" class="rename-btn w-8 h-8 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-blue-400"><i class="fas fa-edit text-ui-label"></i></button>
                        <button onclick="event.stopPropagation(); configureLocalModelAlias('${safePath}')" title="${localAlias ? 'Editar alias' : 'Adicionar alias'}" aria-label="${localAlias ? 'Editar alias' : 'Adicionar alias'}" class="rename-btn w-8 h-8 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-cyan-400"><i class="fas fa-tag text-ui-label"></i></button>
                        <button onclick="event.stopPropagation(); deleteModel('${safePath}')" title="Excluir modelo" aria-label="Excluir modelo" class="w-8 h-8 flex items-center justify-center rounded bg-slate-800/50 text-slate-500 hover:text-red-400"><i class="fas fa-trash-alt text-ui-label"></i></button>
                        ${visionControls}
                        ${mtpControls}
                    </div>
                    <label class="flex items-center gap-1.5 cursor-pointer" onclick="event.stopPropagation()">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Auto-Start</span>
                        <input type="checkbox" class="autostart-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-blue-600" ${isDefault ? 'checked' : ''} onclick="setDefaultModel(this, '${safePath}')">
                    </label>
                </div>
                <div class="proxy-model-controls flex flex-wrap items-center gap-x-3 gap-y-1.5 mt-2 pt-2 border-t border-slate-700/30 min-w-0" onclick="event.stopPropagation()">
                    <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Modelo principal exposto pela API no Modo Proxy Inteligente (apenas um por vez)">
                        <span class="text-ui-label font-black text-violet-400/80 uppercase">Principal</span>
                        <input type="checkbox" class="proxy-primary-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-violet-600" data-path="${escapeHtml(m_js)}" ${isProxyPrimary ? 'checked' : ''} onclick="setProxyPrimary(this, '${safePath}')">
                    </label>
                    <label class="flex items-center gap-1 cursor-pointer shrink-0" title="Usar como backend secundário no proxy inteligente">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Proxy</span>
                        <input type="checkbox" class="proxy-eligible-checkbox w-3 h-3 bg-slate-900 border-slate-700 rounded text-violet-600" ${isProxyEligible ? 'checked' : ''} onclick="setProxyEligible(this, '${safePath}')">
                    </label>
                    <label class="flex items-center gap-1 shrink-0 ml-auto" title="Capacidade paralela inicial; cresce automaticamente sob pressão">
                        <span class="text-ui-label font-black text-slate-600 uppercase">Paralelo</span>
                        <input type="number" min="1" max="16" value="${proxyMaxParallel}" class="proxy-max-parallel w-9 px-0.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-ui-label text-slate-300 text-center outline-none" onchange="setProxyMaxParallel(this, '${safePath}')">
                    </label>
                </div>
            </div>`;
    }).join('');
    const orderedPlatforms = [...(platforms || [])].sort((a, b) => {
        const aIsGenericCreator = a.provider === 'generic-openai' && !a.account_id;
        const bIsGenericCreator = b.provider === 'generic-openai' && !b.account_id;
        return Number(aIsGenericCreator) - Number(bIsGenericCreator);
    });
    const platformHtml = orderedPlatforms.map(p => buildPlatformCardHtml(p, cfg)).join('');
    return localHtml + platformHtml;
}

async function renderModelList(container, models, cfg, platforms = []) {
    container.innerHTML = buildModelListHtml(models, cfg, platforms);
}

export async function persistMmprojSelection(modelPath, mmprojPath, { silent = false } = {}) {
    const normalized = (modelPath || '').replace(/\\/g, '/');
    if (!normalized) return;
    const mmproj = mmprojPath ? mmprojPath.replace(/\\/g, '/') : null;
    if (!window.modelConfigs[normalized]) window.modelConfigs[normalized] = {};
    window.modelConfigs[normalized].mmproj_path = mmproj;
    window.modelConfigs[normalized].mmproj_disabled = mmproj === '__no_vision__';
    try {
        await apiFetch('/models/mmproj', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model_path: normalized,
                mmproj_path: mmproj,
                user_initiated: true,
            }),
        });
    } catch (e) {
        if (!silent) showToast('Erro ao salvar projetor de visão.', 'error');
    }
}

export async function onMmprojChange(modelPath, selectEl) {
    const val = selectEl?.value;
    // Keep __no_vision__ as a sentinel so it persists in config
    await persistMmprojSelection(modelPath, val || null);
}

export async function onMtpModelChange(modelPath, selectEl) {
    const normalized = (modelPath || '').replace(/\\/g, '/');
    const mtpPath = selectEl?.value?.replace(/\\/g, '/') || null;
    if (!window.modelConfigs[normalized]) window.modelConfigs[normalized] = {};
    window.modelConfigs[normalized].mtp_model_path = mtpPath;
    try {
        const res = await apiFetch('/models/mtp', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_path: normalized, mtp_model_path: mtpPath}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Falha ao salvar draft MTP');
        }
    } catch (e) {
        showToast(e.message || 'Erro ao salvar draft MTP.', 'error');
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

export async function persistLlamaBinSettings(modelPath, tabId, { silent = false } = {}) {
    const normalized = (modelPath || '').replace(/\\/g, '/');
    if (!normalized) return;

    const bin = getSelectedLlamaBin(tabId);
    if (!bin) return;

    const tab = document.getElementById(tabId);
    const payload = {
        model_path: normalized,
        llama_server_bin: bin,
    };

    if (isTurboquantBin(bin)) {
        const cacheTypes = getEffectiveCacheTypes(tabId);
        payload.cache_type_k = cacheTypes.cache_type_k;
        payload.cache_type_v = cacheTypes.cache_type_v;
        const preset = getTurboquantPreset(tabId);
        if (preset) payload.turboquant_preset = preset;
    }

    if (!window.modelConfigs[normalized]) window.modelConfigs[normalized] = {};
    Object.assign(window.modelConfigs[normalized], payload);

    try {
        const res = await apiFetch('/models/llama-bin', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        if (!res.ok && !silent) {
            showToast('Erro ao salvar versão do llama.cpp.', 'error');
        }
    } catch (e) {
        if (!silent) showToast('Erro ao salvar versão do llama.cpp.', 'error');
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
            if (inst.config) {
                // Preserve escolhas persistidas que ainda não fazem parte do
                // processo atual (por exemplo, desativar Vision no próximo start).
                window.modelConfigs[path] = {
                    ...inst.config,
                    ...(window.modelConfigs[path] || {}),
                };
            }
            const existing = state.activeTabs.find(t => t.path === path);
            const tabId = existing?.id || createModelTab(path, name, id, false);
            if (!firstTabId) firstTabId = tabId;
        }

        if (firstTabId) switchTab(firstTabId);
        state.initialTabsSynced = true;
    } catch (e) {
        console.error('syncRunningModelTabsOnLoad error:', e);
    }
}

export async function initDashboard() {
    await fetchLlamaBins();
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
            <a href="/ui/${port}/" target="_blank" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-ui-body-sm font-black rounded-xl flex items-center gap-2 uppercase tracking-widest shadow-xl shadow-blue-600/20 transition-all active:scale-95">
                <i class="fas fa-comments"></i> Chat
            </a>
            <button onclick="stopModel(${port})" class="px-5 py-2.5 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/20 text-ui-body-sm font-black rounded-xl transition-all uppercase tracking-widest active:scale-95">
                Encerrar
            </button>
        `;
    }
    return `
        <button onclick="startModel('${path}', '${tabId}')" class="px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white text-ui-body-sm font-black rounded-2xl active:scale-95 flex items-center gap-3 uppercase tracking-[0.2em] shadow-2xl shadow-blue-600/30 transition-all">
            <i class="fas fa-bolt"></i> Iniciar Instância
        </button>
    `;
}

export function selectModel(path, elementId, forceNew = false) {
    const container = document.getElementById(`lib-${elementId}`);
    const name = container?.querySelector('.model-name')?.innerText
        || path.split('/').pop()?.replace(/\.gguf$/i, '')
        || 'Modelo';
    createModelTab(path, name, elementId, true, forceNew);
    
    // Visual feedback in sidebar
    document.querySelectorAll('.model-item-container').forEach(el => el.classList.remove('active-selection'));
    if (container) container.classList.add('active-selection');
}

export function selectModelFromEvent(event, path, elementId) {
    // onauxclick também dispara no botão direito (button 2): ignorar para não
    // abrir/trocar de aba junto com o menu de contexto. Só o do meio (1) força nova aba.
    if (event?.type === 'auxclick' && event.button !== 1) return;
    const forceNew = !!(event?.ctrlKey || event?.metaKey || event?.button === 1);
    if (forceNew) event?.preventDefault?.();
    selectModel(path, elementId, forceNew);
}

export function selectPlatform(backendId, forceNew = false) {
    const container = document.getElementById(platformDomId(backendId));
    const catalog = (state.lastPlatformList || []).find(p => p.backend_id === backendId) || {};
    const platform = platformDisplayState(catalog);
    const name = platform.display_name || platform.name || backendId;
    createPlatformTab(backendId, name, forceNew);

    document.querySelectorAll('.model-item-container').forEach(el => el.classList.remove('active-selection'));
    if (container) container.classList.add('active-selection');
}

export function selectPlatformFromEvent(event, backendId) {
    if (event?.type === 'auxclick' && event.button !== 1) return;
    const forceNew = !!(event?.ctrlKey || event?.metaKey || event?.button === 1);
    if (forceNew) event?.preventDefault?.();
    selectPlatform(backendId, forceNew);
}

export function applyModelConfig(path, tabId) {
    const cfg = window.modelConfigs[path];
    const tab = document.getElementById(tabId);
    if (!cfg || !tab) return;
    
    if (cfg.context_size) setContextSize(cfg.context_size, tabId);
    if (cfg.parallel_slots) tab.querySelector('.tab-parallel-slots').value = cfg.parallel_slots;
    if (cfg.batch_size) tab.querySelector('.tab-batch-size').value = cfg.batch_size;
    if (cfg.ubatch_size) tab.querySelector('.tab-ubatch-size').value = cfg.ubatch_size;
    if (cfg.numa_enabled !== undefined) tab.querySelector('.tab-numa-toggle').checked = !!cfg.numa_enabled;
    const flashAttnToggle = tab.querySelector('.tab-flash-attn-toggle');
    if (flashAttnToggle) {
        flashAttnToggle.checked = cfg.flash_attn_enabled !== false;
    }
    if (cfg.threads !== undefined) tab.querySelector('.tab-threads').value = String(cfg.threads);
    if (cfg.threads_batch !== undefined) tab.querySelector('.tab-threads-batch').value = String(cfg.threads_batch);
    if (cfg.split_mode) tab.querySelector('.tab-split-mode').value = cfg.split_mode;
    populateLlamaBinSelect(tabId, cfg.llama_server_bin || null);
    applySavedCacheTypes(tabId, cfg);
    syncTurboquantPanelVisibility(tabId, { autoPreset: false });
    if (!cfg.llama_server_bin && getSelectedLlamaBin(tabId)) {
        persistLlamaBinSettings(path, tabId, { silent: true });
    }
    
    const thinkingToggle = tab.querySelector('.tab-thinking-toggle');
    if (thinkingToggle) thinkingToggle.checked = cfg.thinking_enabled !== false;
    
    const mtpToggle = tab.querySelector('.tab-mtp-toggle');
    if (mtpToggle) mtpToggle.checked = !!cfg.mtp_enabled;
    const mtpDraft = tab.querySelector('.tab-mtp-draft-tokens');
    if (mtpDraft) {
        const tokens = cfg.mtp_draft_tokens ?? window.__constants?.DEFAULT_MTP_DRAFT_TOKENS ?? 3;
        mtpDraft.value = String(tokens);
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
        await apiFetch('/set_default', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path, add: checkbox.checked }),
        });
    } catch (e) {
        showToast("Erro ao salvar configuração.", 'error');
    }
}

export async function setPlatformAutoStart(checkbox, backendId) {
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                backend_id: backendId,
                auto_start: checkbox.checked,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Falha ao salvar Auto-Start');
        }
        window.platformConfigs = window.platformConfigs || {};
        window.platformConfigs[backendId] = {
            ...(window.platformConfigs[backendId] || {}),
            auto_start: checkbox.checked,
        };
    } catch (e) {
        checkbox.checked = !checkbox.checked;
        showToast(e.message || "Erro ao salvar Auto-Start.", 'error');
    }
}

export async function setPlatformVisionEnabled(checkbox, backendId) {
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                backend_id: backendId,
                vision_enabled: checkbox.checked,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Falha ao salvar Vision');
        }
        window.platformConfigs = window.platformConfigs || {};
        window.platformConfigs[backendId] = {
            ...(window.platformConfigs[backendId] || {}),
            vision_enabled: checkbox.checked,
        };
    } catch (e) {
        checkbox.checked = !checkbox.checked;
        showToast(e.message || 'Erro ao salvar Vision.', 'error');
    }
}

export async function setLocalVisionEnabled(checkbox, modelPath) {
    const normalized = (modelPath || '').replace(/\\/g, '/');
    if (!normalized || !checkbox) return;
    const enabled = checkbox.checked;
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model_path: normalized,
                vision_enabled: enabled,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Falha ao salvar Vision');
        }
        if (!window.modelConfigs[normalized]) window.modelConfigs[normalized] = {};
        window.modelConfigs[normalized].vision_enabled = enabled;
        document.querySelectorAll('select[data-mmproj-for]').forEach((select) => {
            if (select.getAttribute('data-mmproj-for') !== normalized) return;
            select.closest('.model-mmproj-control')?.classList.toggle('hidden', !enabled);
        });
    } catch (e) {
        checkbox.checked = !enabled;
        showToast(e.message || 'Erro ao salvar Vision.', 'error');
    }
}

export async function setPlatformDefaultModel(select, backendId) {
    const value = select.value || '';
    const previous = select.dataset.prevValue || '';
    try {
        const res = await apiFetch('/models/proxy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                backend_id: backendId,
                default_model: value,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Falha ao salvar modelo padrão');
        }
        select.dataset.prevValue = value;
        window.platformConfigs = window.platformConfigs || {};
        window.platformConfigs[backendId] = {
            ...(window.platformConfigs[backendId] || {}),
            default_model: value || null,
        };
        showToast(
            value ? 'Modelo padrão do proxy salvo' : 'Modelo padrão removido',
            'success'
        );
    } catch (e) {
        select.value = previous;
        showToast(e.message || 'Erro ao salvar modelo padrão.', 'error');
    }
}

export async function downloadModel() {
    const url = document.getElementById('download-url').value.trim();
    if (!url) return;
    try {
        const res = await apiFetch('/downloads', {
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
            showToast('Erro: ' + (err.detail || 'Inacessível'), 'error');
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
        state.lastPlatformList = data.platforms || [];
        state.platforms = data.platforms || state.platforms || [];
        state.lastConfig = cfg;
        window.platformConfigs = cfg.platform_configs || {};

        if (cfg.model_configs) {
            for (const [rawPath, settings] of Object.entries(cfg.model_configs)) {
                if (!settings || typeof settings !== 'object') continue;
                const norm = rawPath.replace(/\\/g, '/');
                if (!tabHasPendingProposal(norm)) {
                    mergeModelConfigFromServer(norm, settings);
                }
            }
        }

        renderNoTabShortcuts(data.models, cfg);

        const container = document.getElementById('model-list-container');
        if (!container) return;

        document.getElementById('model-count').innerText = (data.models || []).length + (data.platforms || []).length;
        document.getElementById('repo-storage').innerText = formatRepoStorageLabel(data.storage);
        
        const dirInput = document.getElementById('models-dir-input');
        if (dirInput && data.storage?.path && document.activeElement !== dirInput) {
            dirInput.value = data.storage.path;
        }

        const renderList = () => renderModelList(container, data.models, cfg, data.platforms || []);

        if (isMmprojSelectFocused()) {
            deferredModelListUpdate = renderList;
            patchModelListItems(data.models, cfg);
        } else {
            deferredModelListUpdate = null;
            await renderList();
        }
    } catch (e) {
        console.error('updateModels error:', e);
    }
}

async function platformAction(backendId, action) {
    if (!backendId || platformActions.has(backendId)) return;
    platformActions.add(backendId);
    await updateModels();
    try {
        const res = await apiFetch(`/platforms/${encodeURIComponent(backendId)}/${action}`, {
            method: 'POST',
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || 'platform');
        showToast(
            action === 'start' ? 'Integração iniciada' : 'Integração parada',
            'success',
        );
        await window.updateStatus?.();
        await updateModels();
    } catch (e) {
        await window.updateStatus?.();
        showToast(
            action === 'start'
                ? 'Falha ao iniciar integração'
                : 'Falha ao parar integração',
            'error',
        );
    } finally {
        platformActions.delete(backendId);
        await updateModels();
        const openTab = state.activeTabs.find(t => t.kind === 'platform' && t.backendId === backendId);
        if (openTab) {
            await loadPlatformTabDetails(openTab.id, backendId);
        }
    }
}

export async function startPlatform(backendId) {
    await platformAction(backendId, 'start');
}

export async function stopPlatform(backendId) {
    await platformAction(backendId, 'stop');
}

function cliproxyAuthElements() {
    return {
        modal: document.getElementById('cliproxy-auth-modal'),
        title: document.getElementById('cliproxy-auth-title'),
        subtitle: document.getElementById('cliproxy-auth-subtitle'),
        status: document.getElementById('cliproxy-auth-status'),
        deviceBox: document.getElementById('cliproxy-auth-device'),
        deviceCode: document.getElementById('cliproxy-auth-device-code'),
        deviceUrl: document.getElementById('cliproxy-auth-device-url'),
        oauthBox: document.getElementById('cliproxy-auth-oauth'),
        oauthUrl: document.getElementById('cliproxy-auth-oauth-url'),
        instructions: document.getElementById('cliproxy-auth-instructions'),
        callbackBox: document.getElementById('cliproxy-auth-callback-box'),
        callbackInput: document.getElementById('cliproxy-auth-callback-input'),
        callbackBtn: document.getElementById('cliproxy-auth-callback-btn'),
        log: document.getElementById('cliproxy-auth-log'),
        cancelBtn: document.getElementById('cliproxy-auth-cancel-btn'),
    };
}

function resetCliproxyAuthModal() {
    const els = cliproxyAuthElements();
    if (!els.modal) return;
    els.status.textContent = 'Preparando autenticacao...';
    els.deviceBox.classList.add('hidden');
    els.oauthBox.classList.add('hidden');
    if (els.callbackBox) els.callbackBox.classList.add('hidden');
    els.log.classList.add('hidden');
    els.log.textContent = '';
    els.cancelBtn.classList.add('hidden');
    if (els.callbackInput) els.callbackInput.value = '';
    if (els.callbackBtn) els.callbackBtn.disabled = false;
    els.deviceCode.textContent = '';
    els.deviceUrl.textContent = '';
    els.deviceUrl.href = '#';
    els.oauthUrl.textContent = '';
    els.oauthUrl.href = '#';
    els.instructions.textContent = '';
}

export function closeCliproxyAuthModal() {
    if (cliproxyAuthPollTimer) {
        clearInterval(cliproxyAuthPollTimer);
        cliproxyAuthPollTimer = null;
    }
    cliproxyAuthSessionId = null;
    const els = cliproxyAuthElements();
    if (els.modal) {
        els.modal.classList.add('hidden');
        els.modal.classList.remove('flex');
    }
}

function renderCliproxyAuthSession(session) {
    const els = cliproxyAuthElements();
    if (!els.modal || !session) return;

    const statusText = {
        pending: 'Gerando instrucoes de autenticacao...',
        waiting: 'Aguardando voce concluir o login no navegador.',
        waiting_callback: 'Abra o link, faca login e cole a URL de callback abaixo.',
        completed: 'Autenticacao concluida com sucesso.',
        failed: 'Falha na autenticacao.',
        cancelled: 'Autenticacao cancelada.',
    };
    els.status.textContent = session.error || statusText[session.status] || session.status_message || session.callback_hint || 'Processando...';

    if (session.device_code) {
        els.deviceBox.classList.remove('hidden');
        els.deviceCode.textContent = session.device_code;
        if (session.auth_url) {
            els.deviceUrl.href = session.auth_url;
            els.deviceUrl.textContent = session.auth_url;
        }
    } else if (session.auth_url) {
        els.oauthBox.classList.remove('hidden');
        els.oauthUrl.href = session.auth_url;
        els.oauthUrl.textContent = session.auth_url;
        if ((session.instructions || []).length) {
            els.instructions.textContent = session.instructions.join('\n');
        }
        if (session.needs_callback && els.callbackBox) {
            els.callbackBox.classList.remove('hidden');
            if (session.callback_submitted && els.callbackBtn) {
                els.callbackBtn.disabled = true;
                els.callbackBtn.textContent = 'Callback enviado';
            }
        }
    }

    if (session.output_tail && session.status === 'failed') {
        els.log.classList.remove('hidden');
        els.log.textContent = session.output_tail;
    } else if (els.log) {
        els.log.classList.add('hidden');
        els.log.textContent = '';
    }

    if (session.status === 'waiting' || session.status === 'waiting_callback' || session.status === 'pending') {
        els.cancelBtn.classList.remove('hidden');
    } else {
        els.cancelBtn.classList.add('hidden');
    }
}

export async function submitCliproxyAuthCallback() {
    const els = cliproxyAuthElements();
    if (!cliproxyAuthSessionId || !els.callbackInput) return;
    const callbackUrl = els.callbackInput.value.trim();
    if (!callbackUrl) {
        showToast('Cole a URL de callback completa.', 'error');
        return;
    }
    if (els.callbackBtn) els.callbackBtn.disabled = true;
    try {
        const res = await apiFetch(`/cliproxy/auth/sessions/${encodeURIComponent(cliproxyAuthSessionId)}/callback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ callback_url: callbackUrl }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(data.detail || 'Nao foi possivel enviar o callback.');
        }
        renderCliproxyAuthSession(data.session);
        showToast('Callback enviado. Aguardando confirmacao...', 'success');
        if (!cliproxyAuthPollTimer) {
            cliproxyAuthPollTimer = setInterval(pollCliproxyAuthSession, 2000);
        }
        pollCliproxyAuthSession();
    } catch (e) {
        if (els.callbackBtn) els.callbackBtn.disabled = false;
        showToast(e.message || 'Erro ao enviar callback.', 'error');
    }
}

async function pollCliproxyAuthSession() {
    if (!cliproxyAuthSessionId) return;
    try {
        const res = await apiFetch(`/cliproxy/auth/sessions/${encodeURIComponent(cliproxyAuthSessionId)}`);
        if (!res.ok) return;
        const data = await res.json();
        const session = data.session;
        renderCliproxyAuthSession(session);
        if (session.status === 'completed') {
            if (cliproxyAuthPollTimer) {
                clearInterval(cliproxyAuthPollTimer);
                cliproxyAuthPollTimer = null;
            }
            await apiFetch('/cliproxy/restart', { method: 'POST' });
            showToast('Autenticacao concluida. Sidecar atualizado.', 'success');
            await updateModels();
            if (window.updateStatus) await window.updateStatus();
            closeCliproxyAuthModal();
        } else if (session.status === 'failed' || session.status === 'cancelled') {
            if (cliproxyAuthPollTimer) {
                clearInterval(cliproxyAuthPollTimer);
                cliproxyAuthPollTimer = null;
            }
            if (session.status === 'failed') {
                showToast(session.error || 'Falha na autenticacao do CLIProxyAPI.', 'error');
            }
        }
    } catch (e) {
        // ignore transient polling errors
    }
}

export async function startCliproxyAuth(_backendId, provider, displayName) {
    const els = cliproxyAuthElements();
    if (!els.modal || !provider) return;

    closeCliproxyAuthModal();
    resetCliproxyAuthModal();
    els.title.textContent = `Autenticar ${displayName || provider}`;
    els.subtitle.textContent = 'Abra o link, faca login e cole a URL de callback que o navegador mostrar em localhost.';
    els.modal.classList.remove('hidden');
    els.modal.classList.add('flex');

    try {
        const res = await apiFetch(`/cliproxy/auth/${encodeURIComponent(provider)}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(data.detail || 'Nao foi possivel iniciar a autenticacao.');
        }
        cliproxyAuthSessionId = data.session?.id || null;
        renderCliproxyAuthSession(data.session);
        if (cliproxyAuthSessionId) {
            cliproxyAuthPollTimer = setInterval(pollCliproxyAuthSession, 2000);
            pollCliproxyAuthSession();
        }
    } catch (e) {
        els.status.textContent = e.message || 'Erro ao iniciar autenticacao.';
        showToast(els.status.textContent, 'error');
    }
}

export async function manageOllamaCloudAuth(_backendId, displayName) {
    const apiKey = await showPrompt(
        `Cole a chave de API do ${displayName || 'Ollama Cloud'}:`,
        '',
        { confirmLabel: 'Salvar e validar' },
    );
    if (!apiKey?.trim()) return;
    try {
        const res = await apiFetch('/platforms/ollama-cloud/accounts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey.trim() }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || 'Não foi possível salvar a chave.');
        const validation = await apiFetch(`/platforms/ollama-cloud/accounts/${encodeURIComponent(data.id)}/validate`, { method: 'POST' });
        const validationData = await validation.json().catch(() => ({}));
        if (!validation.ok || !validationData.valid) {
            throw new Error('A chave foi salva, mas não foi aceita pelo Ollama Cloud.');
        }
        showToast('Ollama Cloud autenticado com sucesso.', 'success');
        await updateModels();
        await window.updateStatus?.();
    } catch (e) {
        showToast(e.message || 'Falha ao autenticar no Ollama Cloud.', 'error');
    }
}

export async function deleteOllamaCloudAccount(accountId, backendId, tabId = null) {
    if (!accountId) return;
    if (!await showConfirm('Apagar esta credencial do Ollama Cloud?', { confirmLabel: 'Apagar' })) return;
    try {
        const res = await apiFetch(`/platforms/ollama-cloud/accounts/${encodeURIComponent(accountId)}`, {
            method: 'DELETE',
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || 'Não foi possível apagar a credencial.');
        showToast('Credencial do Ollama Cloud apagada.', 'success');
        await updateModels();
        await window.updateStatus?.();
        if (tabId && backendId && document.getElementById(tabId)) {
            await loadPlatformTabDetails(tabId, backendId);
        }
    } catch (e) {
        showToast(e.message || 'Falha ao apagar a credencial do Ollama Cloud.', 'error');
    }
}

export async function cancelCliproxyAuth() {
    if (!cliproxyAuthSessionId) {
        closeCliproxyAuthModal();
        return;
    }
    try {
        await apiFetch(`/cliproxy/auth/sessions/${encodeURIComponent(cliproxyAuthSessionId)}`, {
            method: 'DELETE',
        });
    } catch (e) {}
    closeCliproxyAuthModal();
}

export async function renameModel(path) {
    const normalized = path.replace(/\\/g, '/');
    const currentName = normalized.split('/').pop().replace(/\.gguf$/i, '');
    const newName = await showPrompt('Novo nome do modelo:', currentName, { confirmLabel: 'Renomear' });
    if (!newName || newName === currentName) return;
    try {
        const res = await apiFetch('/rename', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: normalized, new_name: newName }),
        });
        if (sessionExpiredHandled || !res.ok) {
            if (!sessionExpiredHandled) {
                const err = await res.json().catch(() => ({}));
                showToast('Erro ao renomear: ' + (err.detail || 'Falha desconhecida'), 'error');
            }
            return;
        }
        const data = await res.json().catch(() => ({}));
        const newPath = (data.new_path || normalized).replace(/\\/g, '/');

        state.activeTabs.forEach(tab => {
            if (tab.path.replace(/\\/g, '/') === normalized) {
                tab.path = newPath;
                const el = document.getElementById(tab.id);
                if (el) el.dataset.path = newPath;
            }
        });
        refreshTabLabelsForPath(newPath);

        if (window.modelConfigs[normalized]) {
            window.modelConfigs[newPath] = window.modelConfigs[normalized];
            delete window.modelConfigs[normalized];
        }
        if (state.currentSelectedModel === normalized) {
            state.currentSelectedModel = newPath;
        }

        await updateModels();
    } catch (e) {
        showToast('Erro de rede ao renomear modelo.', 'error');
    }
}

export async function deleteModel(path) {
    if (!await showConfirm('Excluir este modelo permanentemente? O arquivo .gguf será removido do disco.', { confirmLabel: 'Excluir' })) return;
    const normalized = path.replace(/\\/g, '/');
    try {
        const res = await apiFetch('/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast('Erro: ' + (err.detail || 'Falha ao excluir'), 'error');
            return;
        }

        [...state.activeTabs.filter(t => t.path === normalized)].forEach(t => closeTab(t.id));
        delete window.modelConfigs[normalized];
        await updateModels();
    } catch (e) {
        showToast('Erro de rede ao excluir modelo.', 'error');
    }
}

export async function startModel(path, tabId) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    // Evita duplo clique disparando dois POST /start (poderia subir duas instâncias).
    if (tab.dataset.starting === '1') return;

    const payload = collectStartPayloadFromTab(path, tabId, { autoBalanceProfile: false });
    if (!payload) { showToast('Contexto inválido', 'error'); return; }

    const weightValidation = validateDeviceWeights(payload.gpu_weights);
    if (!weightValidation.ok) { showToast(weightValidation.message, 'error'); return; }

    tab.dataset.starting = '1';
    hideProposedConfig(tabId);
    clearScreenSnapshot(tabId);

    const statusBadge = tab.querySelector('.tab-status-badge');
    statusBadge.innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2"></i> INICIANDO...';
    statusBadge.className = 'tab-status-badge px-5 py-2.5 rounded-xl text-ui-body-sm font-black tracking-[0.2em] uppercase glass border-blue-500/50 text-blue-400';

    try {
        const res = await apiFetch('/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        
        if (!res.ok) {
            const err = await res.json();
            showToast("Erro: " + (err.detail || "Falha ao iniciar"), 'error');
            window.updateStatus();
            return;
        }
        
        const startData = await res.json();
        if (startData.port) {
            state.currentActivePort = startData.port;
            attachTabLogs(tabId, startData.port, {
                force: true,
                sessionKey: `${startData.port}:${startData.start_time ?? Date.now()}`,
            });
        }

        await window.updateStatus();
        const inst = (state.activeInstances || []).find(
            i => (i.model_path || '').replace(/\\/g, '/') === path.replace(/\\/g, '/')
        );
        if (!inst || inst.status !== 'running') {
            showToast('O servidor encerrou logo após iniciar. Verifique os logs abaixo.', 'error');
        }
        if (!startData.probing && startData.mtp_applied !== undefined) {
            if (!startData.mtp_applied && startData.mtp_reason) {
                window.showMtpWarning(startData.mtp_reason);
            } else {
                window.hideMtpWarning();
            }
        }
    } catch (e) {
        showToast("Erro de rede.", 'error');
        window.updateStatus();
    } finally {
        delete tab.dataset.starting;
    }
    setTimeout(window.updateStatus, 2000);
}

export async function stopModel(port = null) {
    if (!await showConfirm("Encerrar esta instância?", { confirmLabel: 'Encerrar' })) return;
    try {
        detachTabLogs();
        const url = port !== null ? `/stop?port=${port}` : '/stop';
        const res = await apiFetch(url, {method: 'POST'});
        if (res.ok) {
            setTimeout(window.updateStatus, 1000);
        }
    } catch (e) {}
}


// --- Generic OpenAI Auth ---

let currentGenericOpenAIBackendId = null;

export async function manageGenericOpenAIAuth(backendId, displayName) {
    currentGenericOpenAIBackendId = backendId;
    const modal = document.getElementById('generic-openai-auth-modal');
    if (!modal) return;

    document.getElementById('generic-openai-auth-title').textContent = displayName || 'Plataformas Genéricas';

    await loadGenericOpenAIAccounts();

    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

window.manageGenericOpenAIAuth = manageGenericOpenAIAuth;

window.manageGenericOpenAIAccount = async function(id, name, baseUrl) {
    await manageGenericOpenAIAuth('platform:generic-openai', name || 'API Genérica');
    editGenericOpenAIAccount(id, name, baseUrl);
};

export function closeGenericOpenAIAuthModal() {
    const modal = document.getElementById('generic-openai-auth-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
    cancelGenericOpenAIForm();
}

window.closeGenericOpenAIAuthModal = closeGenericOpenAIAuthModal;

async function loadGenericOpenAIAccounts() {
    const listContainer = document.getElementById('generic-openai-accounts-list');
    if (!listContainer) return;
    listContainer.innerHTML = '<div class="text-center text-slate-400 text-sm py-4"><i class="fas fa-spinner fa-spin mr-2"></i>Carregando contas...</div>';

    try {
        const resp = await apiFetch('/platforms/generic-openai/accounts');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const accounts = Array.isArray(data) ? data : (Array.isArray(data?.accounts) ? data.accounts : []);

        if (accounts.length === 0) {
            listContainer.innerHTML = '<div class="text-center text-slate-500 text-sm py-6 bg-slate-900/50 rounded-xl border border-slate-800">Nenhuma conta cadastrada.</div>';
            return;
        }

        let html = '';
        accounts.forEach(acc => {
            html += `
                <div class="flex items-center justify-between p-3 rounded-xl border border-slate-700 bg-slate-800/50 hover:border-amber-500/30 transition-colors group">
                    <div class="flex-1 min-w-0 pr-4">
                        <div class="font-bold text-white text-sm truncate">${escapeHtml(acc.name)}</div>
                        <div class="text-xs text-slate-400 truncate mt-0.5">${escapeHtml(acc.base_url || 'Endpoint não configurado')}</div>
                        ${acc.is_valid ?
                            '<div class="text-[10px] text-emerald-400 font-bold uppercase mt-1"><i class="fas fa-check-circle mr-1"></i>Válida</div>' :
                            '<div class="text-[10px] text-amber-400 font-bold uppercase mt-1"><i class="fas fa-exclamation-triangle mr-1"></i>Não validada</div>'}
                    </div>
                    <div class="flex items-center gap-2 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                        <button type="button" data-generic-openai-action="validate" data-account-id="${escapeHtml(acc.id)}" class="w-8 h-8 rounded-lg bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 flex items-center justify-center transition-colors" title="Validar Conta">
                            <i class="fas fa-sync-alt text-xs"></i>
                        </button>
                        <button type="button" data-generic-openai-action="edit" data-account-id="${escapeHtml(acc.id)}" class="w-8 h-8 rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600 flex items-center justify-center transition-colors" title="Editar">
                            <i class="fas fa-pen text-xs"></i>
                        </button>
                        <button type="button" data-generic-openai-action="delete" data-account-id="${escapeHtml(acc.id)}" class="w-8 h-8 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 flex items-center justify-center transition-colors" title="Excluir">
                            <i class="fas fa-trash text-xs"></i>
                        </button>
                    </div>
                </div>
            `;
        });

        listContainer.innerHTML = html;
        listContainer.querySelectorAll('[data-generic-openai-action]').forEach((button) => {
            button.addEventListener('click', () => {
                const account = accounts.find(acc => String(acc.id) === button.dataset.accountId);
                if (!account) return;
                if (button.dataset.genericOpenaiAction === 'validate') {
                    validateGenericOpenAIAccount(account.id);
                } else if (button.dataset.genericOpenaiAction === 'edit') {
                    editGenericOpenAIAccount(account.id, account.name, account.base_url || '');
                } else if (button.dataset.genericOpenaiAction === 'delete') {
                    deleteGenericOpenAIAccount(account.id, account.name);
                }
            });
        });
    } catch (e) {
        console.error('Failed to load generic openai accounts:', e);
        listContainer.innerHTML = `<div class="text-center text-red-400 text-sm py-4 bg-red-500/10 rounded-xl border border-red-500/20"><i class="fas fa-exclamation-circle mr-2"></i>Erro ao carregar contas: ${escapeHtml(e.message)}</div>`;
    }
}

window.showGenericOpenAIForm = function() {
    document.getElementById('generic-openai-id').value = '';
    document.getElementById('generic-openai-name').value = '';
    document.getElementById('generic-openai-baseurl').value = 'https://api.openai.com/v1';
    document.getElementById('generic-openai-apikey').value = '';
    document.getElementById('generic-openai-apikey').required = true;
    document.getElementById('generic-openai-apikey-hint').classList.add('hidden');
    document.getElementById('generic-openai-form-title').textContent = 'Adicionar Nova Conta';

    document.getElementById('generic-openai-add-btn-container').classList.add('hidden');
    document.getElementById('generic-openai-form-container').classList.remove('hidden');
};

window.cancelGenericOpenAIForm = function() {
    document.getElementById('generic-openai-form-container').classList.add('hidden');
    document.getElementById('generic-openai-add-btn-container').classList.remove('hidden');
};

window.editGenericOpenAIAccount = function(id, name, baseUrl) {
    document.getElementById('generic-openai-id').value = id;
    document.getElementById('generic-openai-name').value = name;
    document.getElementById('generic-openai-baseurl').value = baseUrl;
    document.getElementById('generic-openai-apikey').value = '';
    document.getElementById('generic-openai-apikey').required = false;
    document.getElementById('generic-openai-apikey-hint').classList.remove('hidden');
    document.getElementById('generic-openai-form-title').textContent = 'Editar Conta';

    document.getElementById('generic-openai-add-btn-container').classList.add('hidden');
    document.getElementById('generic-openai-form-container').classList.remove('hidden');
};

window.saveGenericOpenAIAccount = async function(event) {
    event.preventDefault();

    const id = document.getElementById('generic-openai-id').value;
    const name = document.getElementById('generic-openai-name').value;
    const baseUrl = document.getElementById('generic-openai-baseurl').value;
    const apiKey = document.getElementById('generic-openai-apikey').value;

    const payload = {
        name: name,
        base_url: baseUrl.trim() || 'https://api.openai.com/v1'
    };

    if (apiKey) {
        payload.api_key = apiKey;
    }

    const method = id ? 'PATCH' : 'POST';
    const url = id ? `/platforms/generic-openai/accounts/${encodeURIComponent(id)}` : '/platforms/generic-openai/accounts';

    const btn = event.target.querySelector('button[type="submit"]');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    btn.disabled = true;

    try {
        const resp = await apiFetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Erro HTTP ${resp.status}`);
        }

        cancelGenericOpenAIForm();
        await loadGenericOpenAIAccounts();
        await updateModels();
        await window.updateStatus?.();
        showToast('Conta salva com sucesso.', 'success');
    } catch (e) {
        showToast(`Erro ao salvar conta: ${e.message}`, 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
};

window.deleteGenericOpenAIAccount = async function(id, name, tabId = '') {
    if (!await showConfirm(`Tem certeza que deseja excluir a conta "${name}"?`, { confirmLabel: 'Excluir' })) return;

    try {
        const resp = await apiFetch(`/platforms/generic-openai/accounts/${encodeURIComponent(id)}`, {
            method: 'DELETE'
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Erro HTTP ${resp.status}`);
        }

        await loadGenericOpenAIAccounts();
        if (tabId) closeTab(tabId);
        await updateModels();
        await window.updateStatus?.();
        showToast('Conta excluída com sucesso.', 'success');
    } catch (e) {
        showToast(`Erro ao excluir conta: ${e.message}`, 'error');
    }
};

window.validateGenericOpenAIAccount = async function(id) {
    try {
        const resp = await apiFetch(`/platforms/generic-openai/accounts/${encodeURIComponent(id)}/validate`, {
            method: 'POST'
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Erro HTTP ${resp.status}`);
        }
        const data = await resp.json();
        if (data.valid === true) {
            showToast('Conta validada com sucesso! Conexão estabelecida.', 'success');
        } else {
            showToast(`Falha ao validar conta: ${data.message || 'Erro desconhecido'}`, 'error');
        }

        await loadGenericOpenAIAccounts();
        await updateModels();
        await window.updateStatus?.();
    } catch (e) {
        showToast(`Erro na validação: ${e.message}`, 'error');
    }
};
