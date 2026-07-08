import { state } from './state.js?v=4.1.0';
import { apiFetch } from './auth.js?v=4.1.0';

const CPU_INDEX = -1;

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

export async function fetchLlamaBins() {
    try {
        const res = await apiFetch('/llama-bins');
        if (!res.ok) return [];
        const data = await res.json();
        window.llamaBins = Array.isArray(data.bins) ? data.bins : [];
        window.defaultLlamaBin = data.default || window.llamaBins[0]?.path || null;
        return window.llamaBins;
    } catch (e) {
        window.llamaBins = window.llamaBins || [];
        return window.llamaBins;
    }
}

export function populateLlamaBinSelect(tabId, selectedPath = null) {
    const tab = document.getElementById(tabId);
    const select = tab?.querySelector('.tab-llama-bin');
    if (!select) return;

    const bins = window.llamaBins || [];
    const defaultPath = window.defaultLlamaBin || bins[0]?.path || '';
    const preferred = selectedPath || select.value || defaultPath;

    if (!bins.length) {
        select.innerHTML = '<option value="" class="bg-slate-900">Nenhum binário detectado</option>';
        select.disabled = true;
        select.classList.add('opacity-60', 'cursor-not-allowed');
        return;
    }

    select.innerHTML = bins.map((bin) => {
        const value = escapeHtml(bin.path);
        const label = escapeHtml(bin.label || bin.path);
        return `<option value="${value}" class="bg-slate-900">${label}</option>`;
    }).join('');

    select.value = bins.some((bin) => bin.path === preferred) ? preferred : defaultPath;

    const disabled = bins.length <= 1;
    select.disabled = disabled;
    select.classList.toggle('opacity-60', disabled);
    select.classList.toggle('cursor-not-allowed', disabled);
}

export function getSelectedLlamaBin(tabId = null) {
    const tab = tabId ? document.getElementById(tabId) : null;
    const select = tab?.querySelector('.tab-llama-bin');
    const value = select?.value?.trim();
    if (value) return value;
    return window.defaultLlamaBin || window.llamaBins?.[0]?.path || null;
}

export function isTurboquantBin(binPath) {
    if (!binPath) return false;
    const bins = window.llamaBins || [];
    const match = bins.find((bin) => bin.path === binPath);
    if (match && match.is_turboquant != null) return !!match.is_turboquant;
    const normalized = String(binPath).toLowerCase();
    return normalized.includes('turboquant');
}

function getTurboquantPresets() {
    return window.__constants?.TURBOQUANT_PRESETS || [];
}

function getTurboquantPresetById(presetId) {
    return getTurboquantPresets().find((preset) => preset.id === presetId) || null;
}

export function detectTurboquantPreset(cacheK, cacheV) {
    const presets = getTurboquantPresets();
    return presets.find(
        (preset) => preset.cache_type_k === cacheK && preset.cache_type_v === cacheV,
    )?.id || 'custom';
}

function addSelectOptionIfMissing(select, value) {
    if (!select || !value) return;
    if ([...select.options].some((opt) => opt.value === value)) return;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    opt.className = 'bg-slate-900';
    select.appendChild(opt);
}

function removeSelectOption(select, value) {
    if (!select) return;
    const option = [...select.options].find((opt) => opt.value === value);
    if (option) option.remove();
}

export function ensureMainCacheOptions(tabId = null, isTurbo = null) {
    const scope = getTabScope(tabId);
    const mainK = scope.querySelector('.tab-cache-type-k');
    const mainV = scope.querySelector('.tab-cache-type-v');
    if (!mainK || !mainV) return;

    const turboActive = isTurbo ?? isTurboquantBin(getSelectedLlamaBin(tabId));
    const turboKTypes = window.__constants?.TURBOQUANT_CACHE_K_PRESETS || ['f16', 'q8_0'];
    const turboVTypes = window.__constants?.TURBOQUANT_CACHE_V_PRESETS || ['turbo4', 'turbo3', 'turbo2'];

    for (const value of turboKTypes) addSelectOptionIfMissing(mainK, value);

    if (turboActive) {
        for (const value of turboVTypes) addSelectOptionIfMissing(mainV, value);
        return;
    }

    for (const value of turboVTypes) removeSelectOption(mainV, value);
    if (turboVTypes.includes(mainV.value)) {
        mainV.value = window.__constants?.DEFAULT_CACHE_TYPE || 'f16';
    }
}

export function getEffectiveCacheTypes(tabId = null) {
    const scope = getTabScope(tabId);
    const isTurbo = isTurboquantBin(getSelectedLlamaBin(tabId));
    ensureMainCacheOptions(tabId, isTurbo);

    if (isTurbo) {
        syncTurboquantToCacheFields(tabId);
    }

    const defaultK = window.__constants?.DEFAULT_CACHE_TYPE || 'f16';
    const defaultV = window.__constants?.DEFAULT_CACHE_TYPE || 'f16';
    const turboDefaultK = window.__constants?.TURBOQUANT_DEFAULT_CACHE_K || 'q8_0';
    const turboDefaultV = window.__constants?.TURBOQUANT_DEFAULT_CACHE_V || 'turbo3';

    const mainK = scope.querySelector('.tab-cache-type-k');
    const mainV = scope.querySelector('.tab-cache-type-v');
    let cacheK = mainK?.value?.trim() || '';
    let cacheV = mainV?.value?.trim() || '';

    if (isTurbo) {
        const turboK = scope.querySelector('.tab-turbo-cache-k')?.value?.trim();
        const turboV = scope.querySelector('.tab-turbo-cache-v')?.value?.trim();
        if (turboK) cacheK = turboK;
        if (turboV) cacheV = turboV;
        if (!cacheK) cacheK = turboDefaultK;
        if (!cacheV) cacheV = turboDefaultV;
    } else {
        if (!cacheK) cacheK = defaultK;
        if (!cacheV) cacheV = defaultV;
    }

    if (mainK && cacheK) mainK.value = cacheK;
    if (mainV && cacheV) mainV.value = cacheV;

    return { cache_type_k: cacheK, cache_type_v: cacheV };
}

export function syncTurboquantToCacheFields(tabId = null) {
    const scope = getTabScope(tabId);
    const cacheK = scope.querySelector('.tab-turbo-cache-k');
    const cacheV = scope.querySelector('.tab-turbo-cache-v');
    const mainK = scope.querySelector('.tab-cache-type-k');
    const mainV = scope.querySelector('.tab-cache-type-v');
    if (!cacheK || !cacheV || !mainK || !mainV) return;

    ensureMainCacheOptions(tabId, true);
    if (cacheK.value) mainK.value = cacheK.value;
    if (cacheV.value) mainV.value = cacheV.value;
}

export function syncMainCacheToTurboFields(tabId = null) {
    const scope = getTabScope(tabId);
    if (!isTurboquantBin(getSelectedLlamaBin(tabId))) return;

    const mainK = scope.querySelector('.tab-cache-type-k')?.value;
    const mainV = scope.querySelector('.tab-cache-type-v')?.value;
    const turboK = scope.querySelector('.tab-turbo-cache-k');
    const turboV = scope.querySelector('.tab-turbo-cache-v');
    const presetEl = scope.querySelector('.tab-turboquant-preset');

    if (turboK && mainK) turboK.value = mainK;
    if (turboV && mainV) turboV.value = mainV;
    if (presetEl && mainK && mainV) {
        presetEl.value = detectTurboquantPreset(mainK, mainV);
    }
}

export function applyTurboquantPreset(tabId, presetId) {
    const scope = getTabScope(tabId);
    const preset = getTurboquantPresetById(presetId);
    if (!preset) return;
    const cacheK = scope.querySelector('.tab-turbo-cache-k');
    const cacheV = scope.querySelector('.tab-turbo-cache-v');
    const presetSel = scope.querySelector('.tab-turboquant-preset');
    if (cacheK) cacheK.value = preset.cache_type_k;
    if (cacheV) cacheV.value = preset.cache_type_v;
    if (presetSel) presetSel.value = presetId;
    syncTurboquantToCacheFields(tabId);
}

export function populateTurboquantSelects(tabId) {
    const scope = getTabScope(tabId);
    const kPresets = window.__constants?.TURBOQUANT_CACHE_K_PRESETS || ['f16', 'q8_0'];
    const vPresets = window.__constants?.TURBOQUANT_CACHE_V_PRESETS || ['turbo4', 'turbo3', 'turbo2'];
    const presetSel = scope.querySelector('.tab-turboquant-preset');
    const cacheK = scope.querySelector('.tab-turbo-cache-k');
    const cacheV = scope.querySelector('.tab-turbo-cache-v');

    if (cacheK) {
        cacheK.innerHTML = kPresets.map(
            (value) => `<option value="${escapeHtml(value)}" class="bg-slate-900">${escapeHtml(value)}</option>`,
        ).join('');
    }
    if (cacheV) {
        cacheV.innerHTML = vPresets.map(
            (value) => `<option value="${escapeHtml(value)}" class="bg-slate-900">${escapeHtml(value)}</option>`,
        ).join('');
    }
    if (presetSel) {
        const presets = getTurboquantPresets();
        presetSel.innerHTML = [
            ...presets.map(
                (preset) => `<option value="${escapeHtml(preset.id)}" class="bg-slate-900">${escapeHtml(preset.label)}</option>`,
            ),
            '<option value="custom" class="bg-slate-900">Personalizado</option>',
        ].join('');
    }
}

export function applyTurboquantConfig(tabId, cfg = {}) {
    populateTurboquantSelects(tabId);
    const scope = getTabScope(tabId);
    const defaultK = window.__constants?.TURBOQUANT_DEFAULT_CACHE_K || 'q8_0';
    const defaultV = window.__constants?.TURBOQUANT_DEFAULT_CACHE_V || 'turbo3';
    const cacheK = cfg.cache_type_k || defaultK;
    const cacheV = cfg.cache_type_v || defaultV;
    const presetId = cfg.turboquant_preset || detectTurboquantPreset(cacheK, cacheV);

    const kEl = scope.querySelector('.tab-turbo-cache-k');
    const vEl = scope.querySelector('.tab-turbo-cache-v');
    const presetEl = scope.querySelector('.tab-turboquant-preset');
    if (kEl) kEl.value = cacheK;
    if (vEl) vEl.value = cacheV;
    if (presetEl) presetEl.value = presetId;
    syncTurboquantToCacheFields(tabId);
}

export function applySavedCacheTypes(tabId, cfg = {}) {
    const scope = getTabScope(tabId);
    const isTurbo = isTurboquantBin(getSelectedLlamaBin(tabId));
    ensureMainCacheOptions(tabId, isTurbo);

    const defaultK = window.__constants?.DEFAULT_CACHE_TYPE || 'f16';
    const defaultV = window.__constants?.DEFAULT_CACHE_TYPE || 'f16';
    const cacheK = cfg.cache_type_k || defaultK;
    const cacheV = cfg.cache_type_v || defaultV;

    const mainK = scope.querySelector('.tab-cache-type-k');
    const mainV = scope.querySelector('.tab-cache-type-v');
    if (mainK) {
        addSelectOptionIfMissing(mainK, cacheK);
        mainK.value = cacheK;
    }
    if (mainV) {
        addSelectOptionIfMissing(mainV, cacheV);
        mainV.value = cacheV;
    }

    if (!isTurbo) return;

    populateTurboquantSelects(tabId);
    const kPresets = window.__constants?.TURBOQUANT_CACHE_K_PRESETS || ['f16', 'q8_0'];
    const vPresets = window.__constants?.TURBOQUANT_CACHE_V_PRESETS || ['turbo4', 'turbo3', 'turbo2'];
    const turboDefaultK = window.__constants?.TURBOQUANT_DEFAULT_CACHE_K || 'q8_0';
    const turboDefaultV = window.__constants?.TURBOQUANT_DEFAULT_CACHE_V || 'turbo3';

    const turboK = kPresets.includes(cacheK) ? cacheK : turboDefaultK;
    const turboV = vPresets.includes(cacheV) ? cacheV : turboDefaultV;
    const presetId = cfg.turboquant_preset
        || (kPresets.includes(cacheK) && vPresets.includes(cacheV)
            ? detectTurboquantPreset(cacheK, cacheV)
            : 'custom');

    const kEl = scope.querySelector('.tab-turbo-cache-k');
    const vEl = scope.querySelector('.tab-turbo-cache-v');
    const presetEl = scope.querySelector('.tab-turboquant-preset');
    if (kEl) kEl.value = turboK;
    if (vEl) vEl.value = turboV;
    if (presetEl) presetEl.value = presetId;

    if (vPresets.includes(cacheV) && kPresets.includes(cacheK)) {
        syncTurboquantToCacheFields(tabId);
    }
}

export function getTurboquantPreset(tabId = null) {
    const scope = getTabScope(tabId);
    const presetEl = scope.querySelector('.tab-turboquant-preset');
    const value = presetEl?.value?.trim();
    return value && value !== 'custom' ? value : null;
}

export function syncTurboquantPanelVisibility(tabId = null, { autoPreset = true } = {}) {
    const scope = getTabScope(tabId);
    const binPath = getSelectedLlamaBin(tabId);
    const isTurbo = isTurboquantBin(binPath);
    const panel = scope.querySelector('.tab-turboquant-panel');

    if (panel) panel.classList.toggle('hidden', !isTurbo);
    ensureMainCacheOptions(tabId, isTurbo);

    if (isTurbo) {
        populateTurboquantSelects(tabId);
        const mainK = scope.querySelector('.tab-cache-type-k')?.value;
        const mainV = scope.querySelector('.tab-cache-type-v')?.value;
        const vPresets = window.__constants?.TURBOQUANT_CACHE_V_PRESETS || [];
        const kPresets = window.__constants?.TURBOQUANT_CACHE_K_PRESETS || [];
        if (mainV && vPresets.includes(mainV) && mainK && kPresets.includes(mainK)) {
            applyTurboquantConfig(tabId, { cache_type_k: mainK, cache_type_v: mainV });
        } else if (autoPreset) {
            applyTurboquantPreset(tabId, 'recommended');
        }
    } else if (autoPreset) {
        syncMainCacheToTurboFields(tabId);
    }
}

function getTabScope(tabId = null) {
    if (!tabId) tabId = state.currentTabId;
    return tabId ? document.getElementById(tabId) : document;
}

function getDeviceRows(tabId = null) {
    const scope = getTabScope(tabId);
    return Array.from(scope.querySelectorAll('.gpu-row, .cpu-row'));
}

function getRowCheckbox(row) {
    return row.querySelector('.gpu-checkbox') || row.querySelector('.cpu-checkbox');
}

function getRowWeightInput(row) {
    return row.querySelector('.gpu-weight') || row.querySelector('.cpu-weight');
}

function getRowPinInput(row) {
    return row.querySelector('.gpu-pin') || row.querySelector('.cpu-pin');
}

function findDeviceRow(weight, tabId = null) {
    const scope = getTabScope(tabId);
    if (weight.device === 'cpu' || weight.index === CPU_INDEX) {
        return scope.querySelector('.cpu-row');
    }
    return scope.querySelector(`.gpu-row[data-index="${weight.index}"]`);
}

export function showAutoBalanceCapacityAlert(recovery, tabId = null) {
    hideAutoBalanceProgress(tabId);
    const scope = getTabScope(tabId);
    const el = scope.querySelector('.tab-auto-balance-alert');
    const msgEl = scope.querySelector('.tab-auto-balance-msg');
    const detailsEl = scope.querySelector('.tab-auto-balance-details');
    if (!el || !msgEl) return;

    const d = recovery.failure_details || {};
    msgEl.textContent = recovery.message || 'O modelo excede a capacidade de memória das GPUs.';

    if (detailsEl) {
        const rows = [];
        if (d.total_vram_gb != null) rows.push(`<li>VRAM total: ~${d.total_vram_gb} GB</li>`);
        if (d.context_size) rows.push(`<li>Contexto: ${d.context_size} tokens</li>`);
        detailsEl.innerHTML = rows.join('');
    }

    el.classList.remove('hidden');
}

export function updateAutoBalanceProfileBadge(hasProfile, tabId = null) {
    const scope = getTabScope(tabId);
    const badge = scope.querySelector('.tab-auto-balance-badge');
    if (!badge) return;
    badge.classList.toggle('hidden', !hasProfile);
}

export function balanceWeights(changedInput, tabId = null) {
    redistributeUnpinnedWeights(changedInput, tabId);
}

export function redistributeUnpinnedWeights(changedInput, tabId = null) {
    const rows = getDeviceRows(tabId).filter(r => getRowCheckbox(r)?.checked);
    const pinnedRows = rows.filter(r => getRowPinInput(r)?.checked);
    const unpinnedRows = rows.filter(r => !getRowPinInput(r)?.checked);

    if (rows.length === 0) {
        updateTotal(tabId);
        return;
    }
    if (rows.length === 1) {
        getRowWeightInput(rows[0]).value = 100;
        updateTotal(tabId);
        return;
    }

    const pinnedSum = pinnedRows.reduce(
        (s, r) => s + (parseInt(getRowWeightInput(r).value, 10) || 0), 0
    );

    if (unpinnedRows.length === 0) {
        updateTotal(tabId);
        return;
    }

    if (unpinnedRows.length === 1) {
        getRowWeightInput(unpinnedRows[0]).value = Math.max(0, 100 - pinnedSum);
        updateTotal(tabId);
        return;
    }

    const changedRow = changedInput?.closest('.gpu-row, .cpu-row');
    const changedIsPinned = changedRow && getRowPinInput(changedRow)?.checked;

    if (changedIsPinned || !changedInput) {
        let remaining = Math.max(0, 100 - pinnedSum);
        for (let i = 0; i < unpinnedRows.length; i++) {
            const input = getRowWeightInput(unpinnedRows[i]);
            if (i === unpinnedRows.length - 1) {
                input.value = remaining;
            } else {
                const share = Math.min(
                    remaining,
                    Math.max(0, Math.round(remaining / (unpinnedRows.length - i)))
                );
                input.value = share;
                remaining -= share;
            }
        }
        updateTotal(tabId);
        return;
    }

    let val = parseInt(changedInput.value, 10) || 0;
    const maxForChanged = Math.max(0, 100 - pinnedSum);
    if (val > maxForChanged) { val = maxForChanged; changedInput.value = val; }
    if (val < 0) { val = 0; changedInput.value = 0; }

    const otherUnpinned = unpinnedRows.filter(r => getRowWeightInput(r) !== changedInput);
    let remaining = maxForChanged - val;
    for (let i = 0; i < otherUnpinned.length; i++) {
        const input = getRowWeightInput(otherUnpinned[i]);
        if (i === otherUnpinned.length - 1) {
            input.value = Math.max(0, remaining);
        } else {
            const share = Math.min(
                remaining,
                Math.max(0, Math.round(remaining / (otherUnpinned.length - i)))
            );
            input.value = share;
            remaining -= share;
        }
    }
    updateTotal(tabId);
}

export function bindGpuManualListeners(tabId = null) {
    const scope = getTabScope(tabId);
    scope.querySelectorAll('.gpu-weight, .cpu-weight').forEach(el => {
        el.addEventListener('input', () => markManualGpuChange(tabId));
    });
    scope.querySelectorAll('.gpu-checkbox, .cpu-checkbox').forEach(el => {
        el.addEventListener('change', () => {
            markManualGpuChange(tabId);
            redistributeUnpinnedWeights(null, tabId);
        });
    });
    scope.querySelectorAll('.gpu-main-radio').forEach(el => {
        el.addEventListener('change', () => markManualGpuChange(tabId));
    });
    scope.querySelectorAll('.gpu-pin, .cpu-pin').forEach(el => {
        el.addEventListener('change', () => onGpuPinToggle(el, tabId));
    });
}

export function applyGpuWeightsToUI(weights, duringAutoBalance, tabId = null) {
    if (!weights || !Array.isArray(weights)) return;
    const scope = getTabScope(tabId);
    weights.forEach(w => {
        const row = findDeviceRow(w, tabId);
        if (!row) return;
        const input = getRowWeightInput(row);
        const cb = getRowCheckbox(row);
        const pin = getRowPinInput(row);
        const radio = row.querySelector('.gpu-main-radio');
        if (duringAutoBalance || document.activeElement !== input) {
            input.value = Math.round(w.weight);
        }
        if (cb) {
            cb.checked = w.active !== undefined ? w.active : (w.weight > 0);
        }
        if (radio && w.is_main) radio.checked = true;
        if (pin && w.pinned !== undefined) {
            pin.checked = !!w.pinned;
            if (w.pinned) input.classList.add('ring-1', 'ring-amber-500/50');
            else input.classList.remove('ring-1', 'ring-amber-500/50');
        }
    });
    updateTotal(tabId);
}

export function getActiveWeightTotal(weights) {
    return (weights || []).filter(w => w.active).reduce((sum, w) => sum + (w.weight || 0), 0);
}

export function collectDeviceWeightsFromUI(tabId = null) {
    updateTotal(tabId);
    const scope = getTabScope(tabId);
    const weights = [];
    scope.querySelectorAll('.gpu-row').forEach(r => {
        const isChecked = r.querySelector('.gpu-checkbox').checked;
        const isMain = r.querySelector('.gpu-main-radio').checked;
        const gpuName = r.querySelector('span.font-black')?.innerText?.trim() || 'GPU';
        weights.push({
            index: parseInt(r.dataset.index, 10),
            weight: isChecked ? parseInt(r.querySelector('.gpu-weight').value || 0, 10) : 0,
            name: gpuName,
            active: isChecked,
            is_main: isMain,
            pinned: r.querySelector('.gpu-pin')?.checked || false,
            device: 'gpu',
        });
    });
    const cpuRow = scope.querySelector('.cpu-row');
    if (cpuRow) {
        const cpuChecked = cpuRow.querySelector('.cpu-checkbox')?.checked ?? false;
        weights.push({
            index: -1,
            weight: cpuChecked ? parseInt(cpuRow.querySelector('.cpu-weight')?.value || 0, 10) : 0,
            name: 'CPU',
            active: cpuChecked,
            is_main: false,
            pinned: cpuRow.querySelector('.cpu-pin')?.checked || false,
            device: 'cpu',
        });
    }
    return weights;
}

export function validateDeviceWeights(weights) {
    const active = weights.filter(w => w.active);
    const hasGpu = active.some(w => w.device === 'gpu');
    const hasCpu = active.some(w => w.device === 'cpu');
    if (!hasGpu && !hasCpu) return { ok: false, message: 'SELECIONE UM DISPOSITIVO' };
    const total = getActiveWeightTotal(weights);
    if (Math.abs(total - 100) > 1) return { ok: false, message: `CARGA TOTAL: ${total}% (DEVE SER 100%)` };
    return { ok: true, message: '' };
}

export function updateTotal(tabId = null) {
    let sum = 0;
    const scope = getTabScope(tabId);
    getDeviceRows(tabId).forEach(row => {
        const input = getRowWeightInput(row);
        const isChecked = getRowCheckbox(row)?.checked;
        if (!input) return;
        if (isChecked) sum += parseInt(input.value || 0, 10);
        else input.value = 0;
    });
    const badge = scope.querySelector('.tab-total-percent');
    if (!badge) return;
    badge.innerText = `CARGA: ${sum}%`;
    badge.className = `tab-total-percent text-ui-body-sm font-black tracking-widest uppercase ${sum === 100 ? 'text-blue-500/80' : 'text-red-500/80'}`;
}

export function onGpuPinToggle(pinCheckbox, tabId = null) {
    markManualGpuChange(tabId);
    const row = pinCheckbox.closest('.gpu-row, .cpu-row');
    const weightInput = row ? getRowWeightInput(row) : null;
    if (weightInput) {
        if (pinCheckbox.checked) weightInput.classList.add('ring-1', 'ring-amber-500/50');
        else weightInput.classList.remove('ring-1', 'ring-amber-500/50');
    }
    redistributeUnpinnedWeights(weightInput, tabId);
}

export function markManualGpuChange(tabId = null) {
    state.manualGpuOverride = true;
}

export function getContextSize(tabId = null) {
    const scope = getTabScope(tabId);
    const sel = scope.querySelector('.tab-context-size');
    if (!sel) return window.__constants.DEFAULT_CONTEXT_SIZE;
    if (sel.value === 'custom') {
        const k = parseFloat(scope.querySelector('.tab-context-size-custom')?.value);
        if (!Number.isFinite(k) || k < 1) return null;
        return Math.round(k * window.__constants.CONTEXT_K_MULTIPLIER);
    }
    return parseInt(sel.value, 10) || window.__constants.DEFAULT_CONTEXT_SIZE;
}

export function setContextSize(value, tabId = null) {
    const scope = getTabScope(tabId);
    const sel = scope.querySelector('.tab-context-size');
    const custom = scope.querySelector('.tab-context-size-custom');
    if (!sel || !custom) return;
    const n = parseInt(value, 10);
    if (!Number.isFinite(n)) return;
    if (window.__constants.CONTEXT_PRESET_VALUES.includes(n)) {
        sel.value = String(n);
    } else {
        sel.value = 'custom';
        custom.value = n / window.__constants.CONTEXT_K_MULTIPLIER;
    }
    syncContextSizeCustomVisibility(tabId);
}

export function syncContextSizeCustomVisibility(tabId = null) {
    const scope = getTabScope(tabId);
    const sel = scope.querySelector('.tab-context-size');
    const wrap = scope.querySelector('.tab-custom-ctx-wrap');
    if (!sel || !wrap) return;
    wrap.classList.toggle('hidden', sel.value !== 'custom');
}

export function isModelHardwareIncapable(path) {
    const cfg = window.modelConfigs[path];
    return !!(cfg && cfg.hardware_incapable);
}

export async function cancelAutoBalance() {
    try {
        await apiFetch('/auto-balance/cancel', { method: 'POST' });
    } catch (e) {}
}

export function showAutoBalanceProgress(recovery, tabId = null) {
    const scope = getTabScope(tabId);
    const el = scope.querySelector('.tab-auto-balance-progress');
    const msgEl = scope.querySelector('.tab-auto-balance-progress-msg');
    const attemptEl = scope.querySelector('.tab-auto-balance-progress-attempt');
    if (!el || !msgEl) return;

    const label = recovery?.smart_calibration ? 'Calibração smart' : 'Auto-balance';
    msgEl.textContent = recovery?.message || `Executando ${label}...`;
    if (attemptEl) {
        const attempt = recovery?.attempt;
        attemptEl.textContent = (attempt != null && attempt > 0)
            ? `Tentativa ${attempt} · aguarde, isso pode levar alguns minutos`
            : 'Aguarde, isso pode levar alguns minutos';
    }

    const wasHidden = el.classList.contains('hidden');
    el.classList.remove('hidden');
    syncAutoBalanceCancelButton(true, tabId);
    // Só rola na primeira exibição: chamado a cada poll (1s) durante a calibração,
    // um scrollIntoView repetido sequestraria a rolagem do usuário.
    if (wasHidden) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

export function hideAutoBalanceProgress(tabId = null) {
    if (tabId) {
        const scope = getTabScope(tabId);
        scope.querySelector('.tab-auto-balance-progress')?.classList.add('hidden');
        syncAutoBalanceCancelButton(false, tabId);
        return;
    }
    document.querySelectorAll('.tab-auto-balance-progress').forEach(el => el.classList.add('hidden'));
    state.activeTabs.forEach(tab => syncAutoBalanceCancelButton(false, tab.id));
}

export function hideAutoBalanceCapacityAlert(tabId = null) {
    const scope = getTabScope(tabId);
    const el = scope.querySelector('.tab-auto-balance-alert');
    if (el) el.classList.add('hidden');
}

export function modelIncapableRowClass(incapable) {
    return incapable ? 'border-red-500/40 bg-red-950/20' : '';
}

export function syncAutoBalanceCancelButton(autoBalancing, tabId = null) {
    const scope = getTabScope(tabId);
    const calibrateBtn = scope.querySelector('.tab-smart-calibrate-btn');
    if (calibrateBtn) {
        calibrateBtn.disabled = autoBalancing;
        calibrateBtn.classList.toggle('opacity-50', autoBalancing);
        calibrateBtn.classList.toggle('pointer-events-none', autoBalancing);
    }
}

export function resetToDefaults(tabId = null) {
    const scope = getTabScope(tabId);
    setContextSize(window.__constants.DEFAULT_CONTEXT_SIZE, tabId);
    scope.querySelector('.tab-parallel-slots').value = String(window.__constants.DEFAULT_PARALLEL_SLOTS);
    scope.querySelector('.tab-batch-size').value = String(window.__constants.DEFAULT_BATCH_SIZE);
    scope.querySelector('.tab-cache-type-k').value = window.__constants.DEFAULT_CACHE_TYPE;
    scope.querySelector('.tab-cache-type-v').value = window.__constants.DEFAULT_CACHE_TYPE;
    scope.querySelector('.tab-ubatch-size').value = String(window.__constants?.DEFAULT_UBATCH_SIZE ?? 512);
    
    scope.querySelector('.tab-thinking-toggle').checked = true;
    scope.querySelector('.tab-mtp-toggle').checked = false;
    const mtpDraft = scope.querySelector('.tab-mtp-draft-tokens');
    if (mtpDraft) {
        mtpDraft.value = String(window.__constants.DEFAULT_MTP_DRAFT_TOKENS ?? 3);
    }
    syncMtpDraftTokensState(tabId);
    scope.querySelector('.tab-numa-toggle').checked = false;
    const flashAttnToggle = scope.querySelector('.tab-flash-attn-toggle');
    if (flashAttnToggle) {
        flashAttnToggle.checked = window.__constants?.DEFAULT_FLASH_ATTN_ENABLED !== false;
    }
    populateLlamaBinSelect(tabId, window.defaultLlamaBin || null);
    syncTurboquantPanelVisibility(tabId);
    
    scope.querySelectorAll('.gpu-row').forEach((row, idx) => {
        row.querySelector('.gpu-checkbox').checked = true;
        row.querySelector('.gpu-weight').value = (idx === 0 ? "100" : "0");
        row.querySelector('.gpu-main-radio').checked = (idx === 0);
        const pin = row.querySelector('.gpu-pin');
        if (pin) pin.checked = false;
        row.querySelector('.gpu-weight')?.classList.remove('ring-1', 'ring-amber-500/50');
    });
    
    const cpuRow = scope.querySelector('.cpu-row');
    if (cpuRow) {
        cpuRow.querySelector('.cpu-checkbox').checked = false;
        const weight = cpuRow.querySelector('.cpu-weight');
        weight.value = '0';
        weight.classList.remove('ring-1', 'ring-amber-500/50');
        cpuRow.querySelector('.cpu-pin').checked = false;
    }
    
    updateTotal(tabId);
}

export function modelIncapableBadgeHtml(incapable) {
    if (!incapable) return '';
    return '<span class="text-ui-label font-black text-red-400 bg-red-500/10 px-1.5 py-0.5 rounded border border-red-500/20 uppercase tracking-tighter">Incapaz</span>';
}

export function updateThinkingBadge(enabled, tabId = null) {}
export function updateMtpBadge(enabled, tabId = null) {}

export function getMtpDraftTokens(tabId = null) {
    const scope = getTabScope(tabId);
    const input = scope.querySelector('.tab-mtp-draft-tokens');
    const fallback = window.__constants?.DEFAULT_MTP_DRAFT_TOKENS ?? 3;
    const raw = parseInt(input?.value, 10);
    return Number.isFinite(raw) ? raw : fallback;
}

export function syncMtpDraftTokensState(tabId = null) {
    const scope = getTabScope(tabId);
    const enabled = !!scope.querySelector('.tab-mtp-toggle')?.checked;
    const input = scope.querySelector('.tab-mtp-draft-tokens');
    if (input) input.disabled = !enabled;
}
export function showMtpWarning(reason, tabId = null) {
    const scope = getTabScope(tabId);
    const el = scope.querySelector('.tab-mtp-warning');
    if (el) {
        el.querySelector('.tab-mtp-warning-msg').textContent = reason;
        el.classList.remove('hidden');
    }
}
export function hideMtpWarning(tabId = null) {
    const scope = getTabScope(tabId);
    const el = scope.querySelector('.tab-mtp-warning');
    if (el) el.classList.add('hidden');
}
