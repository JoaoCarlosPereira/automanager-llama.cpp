import { state } from './state.js';
import { apiFetch } from './auth.js';

const CPU_DATA_INDEX = 'cpu';
const CPU_INDEX = -1;

function getDeviceRows() {
    return Array.from(document.querySelectorAll('.gpu-row, .cpu-row'));
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

function isCpuRow(row) {
    return row.classList.contains('cpu-row');
}

function findDeviceRow(weight) {
    if (weight.device === 'cpu' || weight.index === CPU_INDEX) {
        return document.querySelector('.cpu-row');
    }
    return document.querySelector(`.gpu-row[data-index="${weight.index}"]`);
}

export function showAutoBalanceCapacityAlert(recovery) {
    const el = document.getElementById('auto-balance-capacity-alert');
    const msgEl = document.getElementById('auto-balance-capacity-msg');
    const detailsEl = document.getElementById('auto-balance-capacity-details');
    const suggEl = document.getElementById('auto-balance-capacity-suggestions');
    if (!el || !msgEl) return;

    const d = recovery.failure_details || {};
    msgEl.textContent = recovery.message || (
        'O modelo excede a capacidade de memória das GPUs disponíveis.'
    );

    if (detailsEl) {
        const rows = [];
        if (d.model) rows.push(`<li><span class="text-slate-300">Modelo:</span> ${d.model}</li>`);
        if (d.total_vram_gb != null) rows.push(
            `<li><span class="text-slate-300">VRAM total (GPUs ativas):</span> ~${d.total_vram_gb} GB`
        );
        if (d.context_size) rows.push(
            `<li><span class="text-slate-300">Contexto / slot:</span> ${d.context_size} tokens`
        );
        if (d.parallel_slots) rows.push(
            `<li><span class="text-slate-300">Slots paralelos:</span> ${d.parallel_slots}</li>`
        );
        if (Array.isArray(d.gpus) && d.gpus.length) {
            const gpuTxt = d.gpus.map(g =>
                `GPU ${g.index} (${g.name}): ${g.vram_mb} MB`
            ).join(' · ');
            rows.push(`<li><span class="text-slate-300">GPUs testadas:</span> ${gpuTxt}</li>`);
        }
        detailsEl.innerHTML = rows.join('');
    }

    if (suggEl) {
        const tips = Array.isArray(d.suggestions) ? d.suggestions : [
            'Reduza contexto ou use quantização menor',
            'Escolha um modelo menor',
        ];
        suggEl.innerHTML = tips.map(t => `<li>${t}</li>`).join('');
    }

    el.classList.remove('hidden');
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

export function updateAutoBalanceProfileBadge(hasProfile) {
    const badge = document.getElementById('auto-balance-badge');
    if (!badge) return;
    let show = !!hasProfile;
    if (hasProfile === undefined && state.currentSelectedModel) {
        const cfg = window.modelConfigs[state.currentSelectedModel];
        show = !!(cfg && cfg.auto_balance_profile);
    }
    badge.classList.toggle('hidden', !show);
}

export function balanceWeights(changedInput) {
    redistributeUnpinnedWeights(changedInput);
}

export function redistributeUnpinnedWeights(changedInput) {
    const rows = getDeviceRows()
        .filter(r => getRowCheckbox(r)?.checked);
    const pinnedRows = rows.filter(r => getRowPinInput(r)?.checked);
    const unpinnedRows = rows.filter(r => !getRowPinInput(r)?.checked);

    if (rows.length === 0) {
        updateTotal();
        return;
    }
    if (rows.length === 1) {
        getRowWeightInput(rows[0]).value = 100;
        updateTotal();
        return;
    }

    const pinnedSum = pinnedRows.reduce(
        (s, r) => s + (parseInt(getRowWeightInput(r).value, 10) || 0), 0
    );

    if (unpinnedRows.length === 0) {
        updateTotal();
        return;
    }

    if (unpinnedRows.length === 1) {
        getRowWeightInput(unpinnedRows[0]).value = Math.max(0, 100 - pinnedSum);
        updateTotal();
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
        updateTotal();
        return;
    }

    let val = parseInt(changedInput.value, 10) || 0;
    const maxForChanged = Math.max(0, 100 - pinnedSum);
    if (val > maxForChanged) { val = maxForChanged; changedInput.value = val; }
    if (val < 0) { val = 0; changedInput.value = 0; }

    const otherUnpinned = unpinnedRows.filter(
        r => getRowWeightInput(r) !== changedInput
    );
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
    updateTotal();
}

export function bindGpuManualListeners() {
    document.querySelectorAll('.gpu-weight, .cpu-weight').forEach(el => {
        el.addEventListener('input', markManualGpuChange);
    });
    document.querySelectorAll('.gpu-checkbox, .cpu-checkbox').forEach(el => {
        el.addEventListener('change', () => {
            markManualGpuChange();
            redistributeUnpinnedWeights(null);
        });
    });
    document.querySelectorAll('.gpu-main-radio').forEach(el => {
        el.addEventListener('change', markManualGpuChange);
    });
    document.querySelectorAll('.gpu-pin, .cpu-pin').forEach(el => {
        el.addEventListener('change', () => onGpuPinToggle(el));
    });
    document.querySelectorAll('.gpu-pin:checked, .cpu-pin:checked').forEach(pin => {
        getRowWeightInput(pin.closest('.gpu-row, .cpu-row'))
            ?.classList.add('ring-2', 'ring-amber-500/40');
    });
    const abToggle = document.getElementById('auto-balance-toggle');
    if (abToggle) {
        abToggle.addEventListener('change', () => onAutoBalanceToggle(abToggle));
        onAutoBalanceToggle(abToggle);  // aplica estado inicial
    }
}

export function applyGpuWeightsToUI(weights, duringAutoBalance) {
    if (!weights || !Array.isArray(weights)) return;
    const cpuPayload = weights.find(
        w => w.device === 'cpu' || w.index === CPU_INDEX
    );
    weights.forEach(w => {
        const row = findDeviceRow(w);
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
            if (w.pinned) input.classList.add('ring-2', 'ring-amber-500/40');
            else input.classList.remove('ring-2', 'ring-amber-500/40');
        }
    });
    const cpuRow = document.querySelector('.cpu-row');
    if (cpuRow && !cpuPayload) {
        const input = getRowWeightInput(cpuRow);
        if (input && (duringAutoBalance || document.activeElement !== input)) {
            input.value = '0';
        }
    }
    updateTotal();
}

export function getActiveWeightTotal(weights) {
    if (weights) {
        return weights
            .filter(w => w.active)
            .reduce((sum, w) => sum + (w.weight || 0), 0);
    }
    let sum = 0;
    getDeviceRows().forEach(row => {
        const input = getRowWeightInput(row);
        const isChecked = getRowCheckbox(row)?.checked;
        if (!input) return;
        if (isChecked) sum += parseInt(input.value || 0, 10);
    });
    return sum;
}

/** Collect active device weights from the dashboard table (syncs totals first). */
export function collectDeviceWeightsFromUI() {
    updateTotal();
    const weights = [];
    document.querySelectorAll('.gpu-row').forEach(r => {
        const isChecked = r.querySelector('.gpu-checkbox').checked;
        const isMain = r.querySelector('.gpu-main-radio').checked;
        const gpuName = r.querySelector('.text-sm.font-bold')?.innerText?.trim() || 'GPU';
        weights.push({
            index: parseInt(r.dataset.index, 10),
            weight: isChecked
                ? parseInt(r.querySelector('.gpu-weight').value || 0, 10)
                : 0,
            name: gpuName,
            active: isChecked,
            is_main: isMain,
            pinned: r.querySelector('.gpu-pin')?.checked || false,
            device: 'gpu',
        });
    });
    const cpuRow = document.querySelector('.cpu-row');
    if (cpuRow) {
        const cpuChecked = cpuRow.querySelector('.cpu-checkbox')?.checked ?? false;
        const cpuName = cpuRow.querySelector('.text-sm.font-bold')?.innerText?.trim() || 'CPU';
        weights.push({
            index: -1,
            weight: cpuChecked
                ? parseInt(cpuRow.querySelector('.cpu-weight')?.value || 0, 10)
                : 0,
            name: cpuName,
            active: cpuChecked,
            is_main: false,
            pinned: cpuRow.querySelector('.cpu-pin')?.checked || false,
            device: 'cpu',
        });
    }
    return weights;
}

/** @returns {{ ok: boolean, message: string }} */
export function validateDeviceWeights(weights) {
    const active = weights.filter(w => w.active);

    if (!active.some(w => w.device === 'gpu')) {
        return { ok: false, message: 'SELECIONE PELO MENOS UMA GPU' };
    }

    const total = getActiveWeightTotal(weights);
    if (Math.abs(total - 100) > 1) {
        return {
            ok: false,
            message: `A CARGA TOTAL DEVE SER 100% (atual: ${total}%). Ajuste os pesos antes de iniciar.`,
        };
    }

    // CPU weight has no upper cap — LoadDistributor manages spill-over dynamically

    return { ok: true, message: '' };
}

export function updateTotal() {
    let sum = 0;
    getDeviceRows().forEach(row => {
        const input = getRowWeightInput(row);
        const isChecked = getRowCheckbox(row)?.checked;
        if (!input) return;
        if (isChecked) sum += parseInt(input.value || 0, 10);
        else input.value = 0;
    });
    const badge = document.getElementById('total-percent');
    if (!badge) return;
    badge.innerText = `CARGA TOTAL: ${sum}%`;
    badge.className = sum === 100
        ? 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20'
        : 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-red-500/10 text-red-500 border border-red-500/20';
}

export function onGpuPinToggle(pinCheckbox) {
    markManualGpuChange();
    const row = pinCheckbox.closest('.gpu-row, .cpu-row');
    const weightInput = row ? getRowWeightInput(row) : null;
    if (weightInput) {
        if (pinCheckbox.checked) {
            weightInput.classList.add('ring-2', 'ring-amber-500/40');
        } else {
            weightInput.classList.remove('ring-2', 'ring-amber-500/40');
        }
    }
    redistributeUnpinnedWeights(weightInput);
}

export function clearGpuPins() {
    document.querySelectorAll('.gpu-pin, .cpu-pin').forEach(pin => {
        pin.checked = false;
        const row = pin.closest('.gpu-row, .cpu-row');
        const weightInput = row ? getRowWeightInput(row) : null;
        if (weightInput) weightInput.classList.remove('ring-2', 'ring-amber-500/40');
    });
}

export function onAutoBalanceToggle(toggle) {
    // Sob Auto-Balance, a cascata controla 100% da distribuição: limpa e
    // desabilita os pins (ADR-001). Ao desligar, reabilita os controles.
    const active = !!(toggle && toggle.checked);
    if (active) clearGpuPins();
    document.querySelectorAll('.gpu-pin, .cpu-pin').forEach(pin => {
        pin.disabled = active;
    });
}

export function onContextSizeCustomInput() {
    const sel = document.getElementById('context-size');
    if (sel && sel.value !== 'custom') sel.value = 'custom';
    syncContextSizeCustomVisibility();
}

export function getContextSize() {
    const sel = document.getElementById('context-size');
    if (!sel) return window.__constants.DEFAULT_CONTEXT_SIZE;
    if (sel.value === 'custom') {
        const k = parseFloat(document.getElementById('context-size-custom')?.value);
        if (!Number.isFinite(k) || k < 1) return null;
        return Math.round(k * window.__constants.CONTEXT_K_MULTIPLIER);
    }
    const preset = parseInt(sel.value, 10);
    return Number.isFinite(preset) ? preset : window.__constants.DEFAULT_CONTEXT_SIZE;
}

export function markManualGpuChange() {
    state.manualGpuOverride = true;
    const badge = document.getElementById('auto-balance-badge');
    if (badge) badge.classList.add('hidden');
}

export function onContextSizePresetChange() {
    syncContextSizeCustomVisibility();
}

export function tokensToContextK(tokens) {
    const k = tokens / window.__constants.CONTEXT_K_MULTIPLIER;
    if (Number.isInteger(k)) return String(k);
    const rounded = Math.round(k * 1000) / 1000;
    return Number.isInteger(rounded) ? String(rounded) : String(rounded);
}

export function setContextSize(value) {
    const sel = document.getElementById('context-size');
    const custom = document.getElementById('context-size-custom');
    if (!sel || !custom) return;
    const n = parseInt(value, 10);
    if (!Number.isFinite(n)) return;
    if (window.__constants.CONTEXT_PRESET_VALUES.includes(n)) {
        sel.value = String(n);
    } else {
        sel.value = 'custom';
        custom.value = tokensToContextK(n);
    }
    syncContextSizeCustomVisibility();
}

export function syncContextSizeCustomVisibility() {
    const sel = document.getElementById('context-size');
    const wrap = document.getElementById('context-size-custom-wrap');
    const custom = document.getElementById('context-size-custom');
    if (!sel || !wrap || !custom) return;
    const show = sel.value === 'custom';
    wrap.classList.toggle('hidden', !show);
    if (show && !custom.value) custom.focus();
}

export function isModelHardwareIncapable(path) {
    const cfg = window.modelConfigs[path];
    return !!(cfg && cfg.hardware_incapable);
}

export async function cancelAutoBalance() {
    const btn = document.getElementById('auto-balance-cancel-btn');
    if (btn) {
        btn.disabled = true;
        btn.classList.add('opacity-50');
    }
    try {
        const res = await fetch('/auto-balance/cancel', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(err.detail || 'Nao foi possivel cancelar o auto balance.');
        }
    } catch (e) {
        alert('Erro de rede ao cancelar auto balance.');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.classList.remove('opacity-50');
        }
    }
}

export function hideAutoBalanceCapacityAlert() {
    const el = document.getElementById('auto-balance-capacity-alert');
    if (el) el.classList.add('hidden');
}

export function modelIncapableRowClass(incapable) {
    return incapable ? 'border-red-500/40 bg-red-950/20' : '';
}

export function syncAutoBalanceCancelButton(autoBalancing) {
    const btn = document.getElementById('auto-balance-cancel-btn');
    if (btn) btn.classList.toggle('hidden', !autoBalancing);
}

export function resetToDefaults() {
    setContextSize(window.__constants.DEFAULT_CONTEXT_SIZE);
    document.getElementById('parallel-slots').value = String(window.__constants.DEFAULT_PARALLEL_SLOTS);
    document.getElementById('batch-size').value = String(window.__constants.DEFAULT_BATCH_SIZE);
    document.getElementById('split-mode').value = "layer";
    const toggle = document.getElementById('auto-balance-toggle');
    if (toggle) toggle.checked = false;
    const thinkingToggle = document.getElementById('thinking-toggle');
    if (thinkingToggle) {
        thinkingToggle.checked = true;
        updateThinkingBadge(true);
    }
    const mtpToggle = document.getElementById('mtp-toggle');
    if (mtpToggle) {
        mtpToggle.checked = false;
        updateMtpBadge(false);
    }
    const mtpDraftTokens = document.getElementById('mtp-draft-tokens');
    if (mtpDraftTokens) {
        mtpDraftTokens.value = '3';
    }
    document.querySelectorAll('.gpu-row').forEach((row, idx) => {
        row.querySelector('.gpu-checkbox').checked = true;
        row.querySelector('.gpu-weight').value = (idx === 0 ? "100" : "0");
        row.querySelector('.gpu-main-radio').checked = (idx === 0);
        const pin = row.querySelector('.gpu-pin');
        if (pin) pin.checked = false;
        row.querySelector('.gpu-weight')?.classList.remove('ring-2', 'ring-amber-500/40');
    });
    const cpuRow = document.querySelector('.cpu-row');
    if (cpuRow) {
        const cpuCb = getRowCheckbox(cpuRow);
        const cpuWeight = getRowWeightInput(cpuRow);
        const cpuPin = getRowPinInput(cpuRow);
        if (cpuCb) cpuCb.checked = false;
        if (cpuWeight) {
            cpuWeight.value = '0';
            cpuWeight.classList.remove('ring-2', 'ring-amber-500/40');
        }
        if (cpuPin) cpuPin.checked = false;
    }
    markManualGpuChange();
    updateTotal();
}

export function modelIncapableBadgeHtml(incapable) {
    if (!incapable) return '';
    return '<span class="shrink-0 text-[8px] font-black uppercase tracking-wider text-red-400 bg-red-500/15 px-2 py-0.5 rounded-lg border border-red-500/30" title="Incompativel com o hardware atual (auto balance)">Incapaz</span>';
}

export function updateThinkingBadge(enabled) {
    const badge = document.getElementById('thinking-badge');
    if (!badge) return;
    const isOn = enabled !== false;
    badge.innerText = isOn ? 'ON' : 'OFF';
    badge.className = isOn
        ? 'text-[9px] font-black uppercase tracking-wider text-violet-400'
        : 'text-[9px] font-black uppercase tracking-wider text-slate-500';
}

export function updateMtpBadge(enabled) {
    const badge = document.getElementById('mtp-badge');
    if (!badge) return;
    const isOn = !!enabled;
    badge.innerText = isOn ? 'ON' : 'OFF';
    badge.className = isOn
        ? 'text-[9px] font-black uppercase tracking-wider text-amber-400'
        : 'text-[9px] font-black uppercase tracking-wider text-slate-500';
}
