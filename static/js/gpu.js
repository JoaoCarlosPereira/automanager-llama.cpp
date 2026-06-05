import { state } from './state.js';
import { apiFetch } from './auth.js';

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
    const rows = Array.from(document.querySelectorAll('.gpu-row'))
        .filter(r => r.querySelector('.gpu-checkbox').checked);
    const pinnedRows = rows.filter(r => r.querySelector('.gpu-pin').checked);
    const unpinnedRows = rows.filter(r => !r.querySelector('.gpu-pin').checked);

    if (rows.length === 0) {
        updateTotal();
        return;
    }
    if (rows.length === 1) {
        rows[0].querySelector('.gpu-weight').value = 100;
        updateTotal();
        return;
    }

    const pinnedSum = pinnedRows.reduce(
        (s, r) => s + (parseInt(r.querySelector('.gpu-weight').value, 10) || 0), 0
    );

    if (unpinnedRows.length === 0) {
        updateTotal();
        return;
    }

    if (unpinnedRows.length === 1) {
        unpinnedRows[0].querySelector('.gpu-weight').value = Math.max(0, 100 - pinnedSum);
        updateTotal();
        return;
    }

    const changedRow = changedInput?.closest('.gpu-row');
    const changedIsPinned = changedRow && changedRow.querySelector('.gpu-pin').checked;

    if (changedIsPinned || !changedInput) {
        let remaining = Math.max(0, 100 - pinnedSum);
        for (let i = 0; i < unpinnedRows.length; i++) {
            const input = unpinnedRows[i].querySelector('.gpu-weight');
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
        r => r.querySelector('.gpu-weight') !== changedInput
    );
    let remaining = maxForChanged - val;
    for (let i = 0; i < otherUnpinned.length; i++) {
        const input = otherUnpinned[i].querySelector('.gpu-weight');
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
    document.querySelectorAll('.gpu-weight').forEach(el => {
        el.addEventListener('input', markManualGpuChange);
    });
    document.querySelectorAll('.gpu-checkbox').forEach(el => {
        el.addEventListener('change', () => {
            markManualGpuChange();
            redistributeUnpinnedWeights(null);
        });
    });
    document.querySelectorAll('.gpu-main-radio').forEach(el => {
        el.addEventListener('change', markManualGpuChange);
    });
    document.querySelectorAll('.gpu-pin').forEach(el => {
        el.addEventListener('change', () => onGpuPinToggle(el));
    });
    document.querySelectorAll('.gpu-pin:checked').forEach(pin => {
        pin.closest('.gpu-row')?.querySelector('.gpu-weight')
            ?.classList.add('ring-2', 'ring-amber-500/40');
    });
}

export function applyGpuWeightsToUI(weights, duringAutoBalance) {
    if (!weights || !Array.isArray(weights)) return;
    weights.forEach(w => {
        const row = document.querySelector(`.gpu-row[data-index="${w.index}"]`);
        if (!row) return;
        const input = row.querySelector('.gpu-weight');
        const cb = row.querySelector('.gpu-checkbox');
        const pin = row.querySelector('.gpu-pin');
        const radio = row.querySelector('.gpu-main-radio');
        if (duringAutoBalance || document.activeElement !== input) {
            input.value = Math.round(w.weight);
        }
        if (w.active !== undefined) cb.checked = w.active;
        if (w.is_main) radio.checked = true;
        if (pin && w.pinned !== undefined) {
            pin.checked = !!w.pinned;
            if (w.pinned) input.classList.add('ring-2', 'ring-amber-500/40');
            else input.classList.remove('ring-2', 'ring-amber-500/40');
        }
    });
    updateTotal();
}

export function updateTotal() {
    let sum = 0;
    document.querySelectorAll('.gpu-weight').forEach(i => {
        const isChecked = i.closest('.gpu-row').querySelector('.gpu-checkbox').checked;
        if (isChecked) sum += parseInt(i.value || 0);
        else i.value = 0;
    });
    const badge = document.getElementById('total-percent');
    badge.innerText = `CARGA TOTAL: ${sum}%`;
    badge.className = sum === 100
        ? 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20'
        : 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-red-500/10 text-red-500 border border-red-500/20';
}

export function onGpuPinToggle(pinCheckbox) {
    markManualGpuChange();
    const row = pinCheckbox.closest('.gpu-row');
    const weightInput = row?.querySelector('.gpu-weight');
    if (weightInput) {
        if (pinCheckbox.checked) {
            weightInput.classList.add('ring-2', 'ring-amber-500/40');
        } else {
            weightInput.classList.remove('ring-2', 'ring-amber-500/40');
        }
    }
    redistributeUnpinnedWeights(weightInput);
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
    document.getElementById('mmproj-path').value = "";
    document.getElementById('split-mode').value = "layer";
    const toggle = document.getElementById('auto-balance-toggle');
    if (toggle) toggle.checked = false;
    const thinkingToggle = document.getElementById('thinking-toggle');
    if (thinkingToggle) {
        thinkingToggle.checked = true;
        updateThinkingBadge(true);
    }
    document.querySelectorAll('.gpu-row').forEach((row, idx) => {
        row.querySelector('.gpu-checkbox').checked = true;
        row.querySelector('.gpu-weight').value = (idx === 0 ? "100" : "0");
        row.querySelector('.gpu-main-radio').checked = (idx === 0);
        const pin = row.querySelector('.gpu-pin');
        if (pin) pin.checked = false;
        row.querySelector('.gpu-weight')?.classList.remove('ring-2', 'ring-amber-500/40');
    });
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
