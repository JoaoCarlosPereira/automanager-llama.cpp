import { state } from './state.js';
import { apiFetch } from './auth.js';

const CPU_INDEX = -1;

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
    if (!active.some(w => w.device === 'gpu')) return { ok: false, message: 'SELECIONE UMA GPU' };
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
    badge.className = `tab-total-percent text-[10px] font-black tracking-widest uppercase ${sum === 100 ? 'text-blue-500/80' : 'text-red-500/80'}`;
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

export function hideAutoBalanceCapacityAlert(tabId = null) {
    const scope = getTabScope(tabId);
    const el = scope.querySelector('.tab-auto-balance-alert');
    if (el) el.classList.add('hidden');
}

export function modelIncapableRowClass(incapable) {
    return incapable ? 'border-red-500/40 bg-red-950/20' : '';
}

export function syncAutoBalanceCancelButton(autoBalancing, tabId = null) {
    // Shared or localized as needed
}

export function resetToDefaults(tabId = null) {
    const scope = getTabScope(tabId);
    setContextSize(window.__constants.DEFAULT_CONTEXT_SIZE, tabId);
    scope.querySelector('.tab-parallel-slots').value = String(window.__constants.DEFAULT_PARALLEL_SLOTS);
    scope.querySelector('.tab-batch-size').value = String(window.__constants.DEFAULT_BATCH_SIZE);
    scope.querySelector('.tab-cache-type-k').value = window.__constants.DEFAULT_CACHE_TYPE;
    scope.querySelector('.tab-cache-type-v').value = window.__constants.DEFAULT_CACHE_TYPE;
    scope.querySelector('.tab-ubatch-size').value = '512';
    
    scope.querySelector('.tab-thinking-toggle').checked = true;
    scope.querySelector('.tab-mtp-toggle').checked = false;
    const mtpDraft = scope.querySelector('.tab-mtp-draft-tokens');
    if (mtpDraft) {
        mtpDraft.value = String(window.__constants.DEFAULT_MTP_DRAFT_TOKENS ?? 3);
    }
    syncMtpDraftTokensState(tabId);
    scope.querySelector('.tab-numa-toggle').checked = false;
    
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
    return '<span class="text-[8px] font-black text-red-400 bg-red-500/10 px-1.5 py-0.5 rounded border border-red-500/20 uppercase tracking-tighter">Incapaz</span>';
}

export function updateThinkingBadge(enabled, tabId = null) {}
export function updateMtpBadge(enabled, tabId = null) {}

export function getMtpDraftTokens(tabId = null) {
    const scope = getTabScope(tabId);
    const input = scope.querySelector('.tab-mtp-draft-tokens');
    const fallback = window.__constants?.DEFAULT_MTP_DRAFT_TOKENS ?? 3;
    const raw = parseInt(input?.value, 10);
    const n = Number.isFinite(raw) ? raw : fallback;
    return Math.max(1, Math.min(4, n));
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
