import { jest, test, expect, beforeEach, describe } from '@jest/globals';
import * as gpu from './gpu.js';
import { state } from './state.js';

const {
    getContextSize,
    setContextSize,
    tokensToContextK,
    syncContextSizeCustomVisibility,
    onContextSizePresetChange,
    onContextSizeCustomInput,
    updateTotal,
    balanceWeights,
    redistributeUnpinnedWeights,
    onGpuPinToggle,
    bindGpuManualListeners,
    applyGpuWeightsToUI,
    markManualGpuChange,
    hideAutoBalanceCapacityAlert,
    showAutoBalanceCapacityAlert,
    modelIncapableBadgeHtml,
    modelIncapableRowClass,
    isModelHardwareIncapable,
    updateAutoBalanceProfileBadge,
    resetToDefaults,
    cancelAutoBalance,
    syncAutoBalanceCancelButton,
    validateDeviceWeights,
    collectDeviceWeightsFromUI,
    getActiveWeightTotal,
    updateMtpBadge,
} = gpu;

const PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

function setupConstants() {
    window.__constants = {
        CONTEXT_PRESET_VALUES: PRESETS,
        DEFAULT_CONTEXT_SIZE: 65536,
        CONTEXT_K_MULTIPLIER: 1000,
        DEFAULT_PARALLEL_SLOTS: 1,
        DEFAULT_BATCH_SIZE: 2048,
    };
}

function gpuRowHtml(index, { weight = '0', checked = true, pinned = false, main = false } = {}) {
    return `
    <div class="gpu-row" data-index="${index}">
        <input type="checkbox" class="gpu-checkbox" ${checked ? 'checked' : ''}/>
        <input type="number" class="gpu-weight" value="${weight}"/>
        <input type="checkbox" class="gpu-pin" ${pinned ? 'checked' : ''}/>
        <input type="radio" name="gpu-main" class="gpu-main-radio" ${main ? 'checked' : ''}/>
    </div>`;
}

function cpuRowHtml({ weight = '0', checked = true, pinned = false } = {}) {
    return `
    <div class="cpu-row" data-index="cpu">
        <input type="checkbox" class="cpu-checkbox" ${checked ? 'checked' : ''}/>
        <input type="number" class="cpu-weight" value="${weight}"/>
        <input type="checkbox" class="cpu-pin" ${pinned ? 'checked' : ''}/>
    </div>`;
}

function setupGpuDom({ gpuCount = 2, withCpu = false, cpuChecked = true } = {}) {
    const presetOptions = PRESETS.map(v => `<option value="${v}">${v}</option>`).join('');
    const rows = Array.from({ length: gpuCount }, (_, i) =>
        gpuRowHtml(i, { weight: i === 0 ? '100' : '0', main: i === 0 })
    ).join('');

    document.body.innerHTML = `
        <select id="context-size">
            <option value="65536">64K</option>
            ${presetOptions}
            <option value="custom">Custom</option>
        </select>
        <div id="context-size-custom-wrap" class="hidden">
            <input id="context-size-custom" value=""/>
        </div>
        <input id="parallel-slots" value="2"/>
        <input id="batch-size" value="512"/>
        <input id="mmproj-path" value="/old/path"/>
        <select id="split-mode"><option value="row">row</option><option value="layer">layer</option></select>
        <input type="checkbox" id="auto-balance-toggle" checked/>
        <input type="checkbox" id="thinking-toggle" checked/>
        <span id="thinking-badge"></span>
        <input type="checkbox" id="mtp-toggle"/>
        <span id="mtp-badge"></span>
        <input id="mtp-draft-tokens" value="3"/>
        <span id="total-percent"></span>
        <span id="auto-balance-badge" class=""></span>
        <button id="auto-balance-cancel-btn" class="hidden"></button>
        <div id="auto-balance-capacity-alert" class="hidden">
            <p id="auto-balance-capacity-msg"></p>
            <ul id="auto-balance-capacity-details"></ul>
            <ul id="auto-balance-capacity-suggestions"></ul>
        </div>
        ${rows}
        ${withCpu ? cpuRowHtml({ weight: '0', checked: cpuChecked }) : ''}
    `;
}

beforeEach(() => {
    setupConstants();
    setupGpuDom();
    state.manualGpuOverride = false;
    state.currentSelectedModel = null;
    window.modelConfigs = {};
    global.fetch = jest.fn();
    global.alert = jest.fn();
    Element.prototype.scrollIntoView = jest.fn();
});

describe('context size', () => {
    test('getContextSize com preset retorna valor do select', () => {
        document.getElementById('context-size').value = '8192';
        expect(getContextSize()).toBe(8192);
    });

    test('getContextSize com custom valido retorna k * 1000', () => {
        const sel = document.getElementById('context-size');
        sel.value = 'custom';
        document.getElementById('context-size-custom').value = '100';
        expect(getContextSize()).toBe(100000);
    });

    test('getContextSize com custom invalido retorna null', () => {
        document.getElementById('context-size').value = 'custom';
        document.getElementById('context-size-custom').value = '0';
        expect(getContextSize()).toBeNull();
    });

    test('getContextSize sem select usa DEFAULT_CONTEXT_SIZE', () => {
        document.getElementById('context-size').remove();
        expect(getContextSize()).toBe(65536);
    });

    test('getContextSize com preset nao numerico usa default', () => {
        document.getElementById('context-size').value = 'invalid';
        expect(getContextSize()).toBe(65536);
    });

    test('setContextSize com preset valido define select', () => {
        setContextSize(4096);
        expect(document.getElementById('context-size').value).toBe('4096');
        expect(document.getElementById('context-size-custom-wrap').classList.contains('hidden')).toBe(true);
    });

    test('setContextSize com valor custom define select e input', () => {
        setContextSize(100000);
        expect(document.getElementById('context-size').value).toBe('custom');
        expect(document.getElementById('context-size-custom').value).toBe('100');
    });

    test('setContextSize com valor nao finito nao altera DOM', () => {
        document.getElementById('context-size').value = '8192';
        setContextSize('abc');
        expect(document.getElementById('context-size').value).toBe('8192');
    });

    test('tokensToContextK converte tokens para K', () => {
        expect(tokensToContextK(100000)).toBe('100');
        expect(tokensToContextK(150000)).toBe('150');
    });

    test('tokensToContextK com valor fracionario', () => {
        expect(tokensToContextK(1500)).toBe('1.5');
    });

    test('syncContextSizeCustomVisibility mostra wrap em custom', () => {
        document.getElementById('context-size').value = 'custom';
        syncContextSizeCustomVisibility();
        expect(document.getElementById('context-size-custom-wrap').classList.contains('hidden')).toBe(false);
    });

    test('onContextSizePresetChange sincroniza visibilidade', () => {
        document.getElementById('context-size').value = 'custom';
        onContextSizePresetChange();
        expect(document.getElementById('context-size-custom-wrap').classList.contains('hidden')).toBe(false);
    });

    test('onContextSizeCustomInput forca modo custom', () => {
        document.getElementById('context-size').value = '8192';
        document.getElementById('context-size-custom').value = '50';
        onContextSizeCustomInput();
        expect(document.getElementById('context-size').value).toBe('custom');
    });
});

describe('updateTotal e pesos GPU', () => {
    test('updateTotal com soma 100% usa badge azul', () => {
        const rows = document.querySelectorAll('.gpu-row');
        rows[0].querySelector('.gpu-weight').value = '60';
        rows[1].querySelector('.gpu-weight').value = '40';
        updateTotal();
        const badge = document.getElementById('total-percent');
        expect(badge.innerText).toBe('CARGA TOTAL: 100%');
        expect(badge.className).toContain('text-blue-400');
    });

    test('updateTotal com soma diferente de 100% usa badge vermelho', () => {
        document.querySelector('.gpu-weight').value = '90';
        updateTotal();
        const badge = document.getElementById('total-percent');
        expect(badge.innerText).toBe('CARGA TOTAL: 90%');
        expect(badge.className).toContain('text-red-500');
    });

    test('updateTotal zera peso de GPU desmarcada', () => {
        const row = document.querySelector('.gpu-row');
        row.querySelector('.gpu-checkbox').checked = false;
        row.querySelector('.gpu-weight').value = '50';
        updateTotal();
        expect(row.querySelector('.gpu-weight').value).toBe('0');
    });

    test('balanceWeights delega para redistributeUnpinnedWeights', () => {
        const input = document.querySelectorAll('.gpu-weight')[1];
        input.value = '30';
        balanceWeights(input);
        updateTotal();
        const sum = [...document.querySelectorAll('.gpu-weight')]
            .filter(w => w.closest('.gpu-row').querySelector('.gpu-checkbox').checked)
            .reduce((s, w) => s + parseInt(w.value, 10), 0);
        expect(sum).toBe(100);
    });

    test('redistributeUnpinnedWeights com uma GPU ativa atribui 100%', () => {
        document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-checkbox').checked = false;
        redistributeUnpinnedWeights(null);
        expect(document.querySelector('.gpu-weight').value).toBe('100');
    });

    test('redistributeUnpinnedWeights com duas GPUs distribui igualmente', () => {
        document.querySelectorAll('.gpu-weight').forEach(w => { w.value = '0'; });
        redistributeUnpinnedWeights(null);
        const weights = [...document.querySelectorAll('.gpu-weight')].map(w => parseInt(w.value, 10));
        expect(weights.reduce((a, b) => a + b, 0)).toBe(100);
    });

    test('redistributeUnpinnedWeights com pinned respeita soma pinned', () => {
        const rows = document.querySelectorAll('.gpu-row');
        rows[0].querySelector('.gpu-pin').checked = true;
        rows[0].querySelector('.gpu-weight').value = '40';
        rows[1].querySelector('.gpu-weight').value = '0';
        redistributeUnpinnedWeights(null);
        expect(parseInt(rows[1].querySelector('.gpu-weight').value, 10)).toBe(60);
    });

    test('redistributeUnpinnedWeights com unpinned alterado redistribui demais', () => {
        const rows = document.querySelectorAll('.gpu-row');
        const changed = rows[0].querySelector('.gpu-weight');
        changed.value = '70';
        redistributeUnpinnedWeights(changed);
        expect(parseInt(rows[1].querySelector('.gpu-weight').value, 10)).toBe(30);
    });

    test('redistributeUnpinnedWeights limita valor acima do maximo', () => {
        const rows = document.querySelectorAll('.gpu-row');
        rows[0].querySelector('.gpu-pin').checked = true;
        rows[0].querySelector('.gpu-weight').value = '80';
        const changed = rows[1].querySelector('.gpu-weight');
        changed.value = '50';
        redistributeUnpinnedWeights(changed);
        expect(parseInt(changed.value, 10)).toBeLessThanOrEqual(20);
    });

    test('redistributeUnpinnedWeights sem GPUs ativas apenas atualiza total', () => {
        document.querySelectorAll('.gpu-checkbox').forEach(cb => { cb.checked = false; });
        redistributeUnpinnedWeights(null);
        expect(document.getElementById('total-percent').innerText).toContain('0%');
    });

    test('redistributeUnpinnedWeights so pinned nao altera pesos', () => {
        document.querySelectorAll('.gpu-pin').forEach(p => { p.checked = true; });
        document.querySelectorAll('.gpu-weight')[0].value = '60';
        document.querySelectorAll('.gpu-weight')[1].value = '40';
        redistributeUnpinnedWeights(null);
        expect(document.querySelectorAll('.gpu-weight')[0].value).toBe('60');
    });

    test('redistributeUnpinnedWeights com input pinned alterado redistribui unpinned', () => {
        const rows = document.querySelectorAll('.gpu-row');
        rows[0].querySelector('.gpu-pin').checked = true;
        rows[0].querySelector('.gpu-weight').value = '30';
        rows[1].querySelector('.gpu-weight').value = '0';
        const pinnedWeight = rows[0].querySelector('.gpu-weight');
        redistributeUnpinnedWeights(pinnedWeight);
        expect(parseInt(rows[1].querySelector('.gpu-weight').value, 10)).toBe(70);
    });

    test('redistributeUnpinnedWeights com valor negativo no input alterado zera', () => {
        const rows = document.querySelectorAll('.gpu-row');
        const changed = rows[1].querySelector('.gpu-weight');
        changed.value = '-10';
        redistributeUnpinnedWeights(changed);
        expect(changed.value).toBe('0');
    });

    test('redistributeUnpinnedWeights com tres GPUs unpinned distribui restante', () => {
        setupGpuDom({ gpuCount: 3 });
        const weights = document.querySelectorAll('.gpu-weight');
        weights.forEach(w => { w.value = '0'; });
        weights[0].value = '50';
        redistributeUnpinnedWeights(weights[0]);
        const vals = [...weights].map(w => parseInt(w.value, 10));
        expect(vals.reduce((a, b) => a + b, 0)).toBe(100);
        expect(vals[0]).toBe(50);
    });
});

describe('pin, listeners e applyGpuWeightsToUI', () => {
    test('onGpuPinToggle adiciona ring ao fixar', () => {
        const pin = document.querySelector('.gpu-pin');
        pin.checked = true;
        onGpuPinToggle(pin);
        expect(document.querySelector('.gpu-weight').classList.contains('ring-2')).toBe(true);
        expect(state.manualGpuOverride).toBe(true);
    });

    test('onGpuPinToggle remove ring ao desfixar', () => {
        const pin = document.querySelector('.gpu-pin');
        const weight = pin.closest('.gpu-row').querySelector('.gpu-weight');
        weight.classList.add('ring-2', 'ring-amber-500/40');
        pin.checked = false;
        onGpuPinToggle(pin);
        expect(weight.classList.contains('ring-2')).toBe(false);
    });

    test('markManualGpuChange seta flag e esconde badge', () => {
        const badge = document.getElementById('auto-balance-badge');
        badge.classList.remove('hidden');
        markManualGpuChange();
        expect(state.manualGpuOverride).toBe(true);
        expect(badge.classList.contains('hidden')).toBe(true);
    });

    test('bindGpuManualListeners dispara markManualGpuChange em input', () => {
        bindGpuManualListeners();
        state.manualGpuOverride = false;
        document.querySelector('.gpu-weight').dispatchEvent(new Event('input'));
        expect(state.manualGpuOverride).toBe(true);
    });

    test('bindGpuManualListeners recalcula ao mudar checkbox', () => {
        bindGpuManualListeners();
        const cb = document.querySelectorAll('.gpu-checkbox')[1];
        cb.checked = true;
        cb.dispatchEvent(new Event('change'));
        updateTotal();
        const sum = [...document.querySelectorAll('.gpu-weight')]
            .filter(w => w.closest('.gpu-row').querySelector('.gpu-checkbox').checked)
            .reduce((s, w) => s + parseInt(w.value, 10), 0);
        expect(sum).toBe(100);
    });

    test('bindGpuManualListeners aplica ring em pins ja marcados', () => {
        document.querySelector('.gpu-pin').checked = true;
        bindGpuManualListeners();
        expect(document.querySelector('.gpu-weight').classList.contains('ring-2')).toBe(true);
    });

    test('bindGpuManualListeners dispara onGpuPinToggle ao mudar pin', () => {
        bindGpuManualListeners();
        const pin = document.querySelector('.gpu-pin');
        pin.checked = true;
        pin.dispatchEvent(new Event('change'));
        expect(document.querySelector('.gpu-weight').classList.contains('ring-2')).toBe(true);
    });

    test('bindGpuManualListeners dispara markManualGpuChange no radio main', () => {
        bindGpuManualListeners();
        state.manualGpuOverride = false;
        document.querySelector('.gpu-main-radio').dispatchEvent(new Event('change'));
        expect(state.manualGpuOverride).toBe(true);
    });

    test('applyGpuWeightsToUI atualiza inputs', () => {
        applyGpuWeightsToUI([
            { index: 0, weight: 55, active: true, is_main: true, pinned: true },
            { index: 1, weight: 45, active: true, pinned: false },
        ]);
        const rows = document.querySelectorAll('.gpu-row');
        expect(rows[0].querySelector('.gpu-weight').value).toBe('55');
        expect(rows[1].querySelector('.gpu-weight').value).toBe('45');
        expect(rows[0].querySelector('.gpu-pin').checked).toBe(true);
    });

    test('applyGpuWeightsToUI ignora weights invalidos', () => {
        applyGpuWeightsToUI(null);
        applyGpuWeightsToUI('bad');
        expect(document.querySelector('.gpu-weight').value).toBe('100');
    });

    test('applyGpuWeightsToUI nao sobrescreve input com foco sem duringAutoBalance', () => {
        const row = document.querySelector('.gpu-row');
        const input = row.querySelector('.gpu-weight');
        input.value = '99';
        input.focus();
        applyGpuWeightsToUI([{ index: 0, weight: 10 }], false);
        expect(input.value).toBe('99');
    });

    test('applyGpuWeightsToUI sobrescreve com duringAutoBalance', () => {
        const input = document.querySelector('.gpu-weight');
        input.value = '99';
        input.focus();
        applyGpuWeightsToUI([{ index: 0, weight: 25 }], true);
        expect(input.value).toBe('25');
    });

    test('applyGpuWeightsToUI ignora indice inexistente', () => {
        applyGpuWeightsToUI([{ index: 99, weight: 50 }]);
        expect(document.querySelector('.gpu-weight').value).toBe('100');
    });

    test('applyGpuWeightsToUI desmarca pin remove ring', () => {
        const row = document.querySelector('.gpu-row');
        row.querySelector('.gpu-pin').checked = true;
        row.querySelector('.gpu-weight').classList.add('ring-2', 'ring-amber-500/40');
        applyGpuWeightsToUI([{ index: 0, weight: 50, pinned: false }]);
        expect(row.querySelector('.gpu-weight').classList.contains('ring-2')).toBe(false);
    });
});

describe('auto balance alerts e badge', () => {
    test('showAutoBalanceCapacityAlert preenche e exibe alert', () => {
        showAutoBalanceCapacityAlert({
            message: 'VRAM insuficiente',
            failure_details: {
                model: 'test.gguf',
                total_vram_gb: 24,
                context_size: 65536,
                parallel_slots: 2,
                gpus: [{ index: 0, name: 'RTX', vram_mb: 12000 }],
                suggestions: ['Reduza contexto'],
            },
        });
        const el = document.getElementById('auto-balance-capacity-alert');
        expect(el.classList.contains('hidden')).toBe(false);
        expect(document.getElementById('auto-balance-capacity-msg').textContent).toBe('VRAM insuficiente');
        expect(document.getElementById('auto-balance-capacity-details').innerHTML).toContain('test.gguf');
        expect(document.getElementById('auto-balance-capacity-suggestions').innerHTML).toContain('Reduza contexto');
        expect(el.scrollIntoView).toHaveBeenCalled();
    });

    test('showAutoBalanceCapacityAlert usa defaults sem failure_details', () => {
        showAutoBalanceCapacityAlert({});
        expect(document.getElementById('auto-balance-capacity-msg').textContent)
            .toContain('capacidade de memória');
        expect(document.getElementById('auto-balance-capacity-suggestions').innerHTML)
            .toContain('Reduza contexto');
    });

    test('showAutoBalanceCapacityAlert retorna cedo sem elementos', () => {
        document.getElementById('auto-balance-capacity-alert').remove();
        expect(() => showAutoBalanceCapacityAlert({})).not.toThrow();
    });

    test('showAutoBalanceCapacityAlert sem details nem suggestions', () => {
        document.getElementById('auto-balance-capacity-details').remove();
        document.getElementById('auto-balance-capacity-suggestions').remove();
        showAutoBalanceCapacityAlert({ message: 'Aviso' });
        expect(document.getElementById('auto-balance-capacity-msg').textContent).toBe('Aviso');
    });

    test('hideAutoBalanceCapacityAlert adiciona hidden', () => {
        const el = document.getElementById('auto-balance-capacity-alert');
        el.classList.remove('hidden');
        hideAutoBalanceCapacityAlert();
        expect(el.classList.contains('hidden')).toBe(true);
    });

    test('updateAutoBalanceProfileBadge mostra com hasProfile true', () => {
        updateAutoBalanceProfileBadge(true);
        expect(document.getElementById('auto-balance-badge').classList.contains('hidden')).toBe(false);
    });

    test('updateAutoBalanceProfileBadge esconde com hasProfile false', () => {
        const badge = document.getElementById('auto-balance-badge');
        badge.classList.remove('hidden');
        updateAutoBalanceProfileBadge(false);
        expect(badge.classList.contains('hidden')).toBe(true);
    });

    test('updateAutoBalanceProfileBadge infere de modelConfigs', () => {
        state.currentSelectedModel = '/models/a.gguf';
        window.modelConfigs = {
            '/models/a.gguf': { auto_balance_profile: { weights: [] } },
        };
        updateAutoBalanceProfileBadge();
        expect(document.getElementById('auto-balance-badge').classList.contains('hidden')).toBe(false);
    });

    test('updateAutoBalanceProfileBadge sem elemento badge nao lanca', () => {
        document.getElementById('auto-balance-badge').remove();
        expect(() => updateAutoBalanceProfileBadge(true)).not.toThrow();
    });

    test('markManualGpuChange sem badge no DOM', () => {
        document.getElementById('auto-balance-badge').remove();
        markManualGpuChange();
        expect(state.manualGpuOverride).toBe(true);
    });

    test('syncAutoBalanceCancelButton alterna visibilidade', () => {
        const btn = document.getElementById('auto-balance-cancel-btn');
        syncAutoBalanceCancelButton(true);
        expect(btn.classList.contains('hidden')).toBe(false);
        syncAutoBalanceCancelButton(false);
        expect(btn.classList.contains('hidden')).toBe(true);
    });
});

describe('cancelAutoBalance e model incapable', () => {
    test('cancelAutoBalance com sucesso', async () => {
        fetch.mockResolvedValue({ ok: true });
        await cancelAutoBalance();
        expect(fetch).toHaveBeenCalledWith('/auto-balance/cancel', { method: 'POST' });
        expect(document.getElementById('auto-balance-cancel-btn').disabled).toBe(false);
    });

    test('cancelAutoBalance com erro API', async () => {
        fetch.mockResolvedValue({
            ok: false,
            json: async () => ({ detail: 'Em execucao' }),
        });
        await cancelAutoBalance();
        expect(alert).toHaveBeenCalledWith('Em execucao');
    });

    test('cancelAutoBalance com erro API sem detail', async () => {
        fetch.mockResolvedValue({
            ok: false,
            json: async () => ({}),
        });
        await cancelAutoBalance();
        expect(alert).toHaveBeenCalledWith('Nao foi possivel cancelar o auto balance.');
    });

    test('cancelAutoBalance com erro de rede', async () => {
        fetch.mockRejectedValue(new Error('network'));
        await cancelAutoBalance();
        expect(alert).toHaveBeenCalledWith('Erro de rede ao cancelar auto balance.');
    });

    test('cancelAutoBalance sem botao no DOM', async () => {
        document.getElementById('auto-balance-cancel-btn').remove();
        fetch.mockResolvedValue({ ok: true });
        await expect(cancelAutoBalance()).resolves.toBeUndefined();
    });

    test('isModelHardwareIncapable retorna true quando configurado', () => {
        window.modelConfigs = { '/m/x.gguf': { hardware_incapable: true } };
        expect(isModelHardwareIncapable('/m/x.gguf')).toBe(true);
        expect(isModelHardwareIncapable('/m/other.gguf')).toBe(false);
    });

    test('modelIncapableBadgeHtml', () => {
        expect(modelIncapableBadgeHtml(false)).toBe('');
        expect(modelIncapableBadgeHtml(true)).toContain('Incapaz');
    });

    test('modelIncapableRowClass', () => {
        expect(modelIncapableRowClass(false)).toBe('');
        expect(modelIncapableRowClass(true)).toContain('border-red-500');
    });
});

describe('resetToDefaults', () => {
    test('setContextSize sem elementos de contexto retorna cedo', () => {
        document.getElementById('context-size').remove();
        document.getElementById('context-size-custom').remove();
        expect(() => setContextSize(4096)).not.toThrow();
    });

    test('syncContextSizeCustomVisibility sem elementos retorna cedo', () => {
        document.getElementById('context-size').remove();
        expect(() => syncContextSizeCustomVisibility()).not.toThrow();
    });

    test('hideAutoBalanceCapacityAlert sem elemento', () => {
        document.getElementById('auto-balance-capacity-alert').remove();
        expect(() => hideAutoBalanceCapacityAlert()).not.toThrow();
    });

    test('syncAutoBalanceCancelButton sem botao', () => {
        document.getElementById('auto-balance-cancel-btn').remove();
        expect(() => syncAutoBalanceCancelButton(true)).not.toThrow();
    });

    test('resetToDefaults restaura valores padrao', () => {
        document.getElementById('context-size').value = '8192';
        document.getElementById('parallel-slots').value = '8';
        document.getElementById('batch-size').value = '4096';
        document.getElementById('mmproj-path').value = '/tmp/mmproj';
        document.getElementById('split-mode').value = 'row';
        document.getElementById('auto-balance-toggle').checked = true;

        resetToDefaults();

        expect(getContextSize()).toBe(65536);
        expect(document.getElementById('parallel-slots').value).toBe('1');
        expect(document.getElementById('batch-size').value).toBe('2048');
        expect(document.getElementById('mmproj-path').value).toBe('');
        expect(document.getElementById('split-mode').value).toBe('layer');
        expect(document.getElementById('auto-balance-toggle').checked).toBe(false);
        expect(document.getElementById('mtp-toggle').checked).toBe(false);
        expect(document.getElementById('mtp-draft-tokens').value).toBe('3');
        const rows = document.querySelectorAll('.gpu-row');
        expect(rows[0].querySelector('.gpu-weight').value).toBe('100');
        expect(rows[1].querySelector('.gpu-weight').value).toBe('0');
        expect(state.manualGpuOverride).toBe(true);
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
    });

    test('resetToDefaults sem auto-balance-toggle', () => {
        document.getElementById('auto-balance-toggle').remove();
        expect(() => resetToDefaults()).not.toThrow();
    });

    test('resetToDefaults zera peso da CPU e desmarca checkbox', () => {
        setupGpuDom({ gpuCount: 2, withCpu: true });
        document.querySelector('.cpu-weight').value = '40';
        document.querySelector('.cpu-checkbox').checked = true;
        resetToDefaults();
        expect(document.querySelector('.cpu-weight').value).toBe('0');
        expect(document.querySelector('.cpu-checkbox').checked).toBe(false);
    });
});

describe('validateDeviceWeights', () => {
    test('rejeita soma diferente de 100%', () => {
        const result = validateDeviceWeights([
            { index: 0, weight: 50, active: true, device: 'gpu' },
            { index: -1, weight: 30, active: true, device: 'cpu' },
        ]);
        expect(result.ok).toBe(false);
        expect(result.message).toContain('100%');
    });

    test('aceita GPU-only com soma 100%', () => {
        const result = validateDeviceWeights([
            { index: 0, weight: 100, active: true, device: 'gpu' },
        ]);
        expect(result.ok).toBe(true);
    });

    test('aceita CPU inativa com peso 0', () => {
        const result = validateDeviceWeights([
            { index: 0, weight: 100, active: true, device: 'gpu' },
            { index: -1, weight: 0, active: false, device: 'cpu' },
        ]);
        expect(result.ok).toBe(true);
    });

    test('aceita qualquer peso de CPU (sem cap de 70%)', () => {
        const result = validateDeviceWeights([
            { index: 0, weight: 20, active: true, device: 'gpu' },
            { index: -1, weight: 80, active: true, device: 'cpu' },
        ]);
        expect(result.ok).toBe(true);
    });

    test('rejeita offload apenas em CPU', () => {
        const result = validateDeviceWeights([
            { index: -1, weight: 100, active: true, device: 'cpu' },
        ]);
        expect(result.ok).toBe(false);
        expect(result.message).toContain('GPU');
    });

    test('getActiveWeightTotal soma apenas dispositivos ativos', () => {
        expect(getActiveWeightTotal([
            { weight: 70, active: true },
            { weight: 30, active: false },
        ])).toBe(70);
    });
});

describe('CPU offload na tabela de dispositivos', () => {
    beforeEach(() => {
        setupGpuDom({ gpuCount: 2, withCpu: true });
    });

    test('updateTotal inclui peso da CPU ativa', () => {
        const rows = document.querySelectorAll('.gpu-row');
        rows[0].querySelector('.gpu-weight').value = '70';
        rows[1].querySelector('.gpu-weight').value = '20';
        document.querySelector('.cpu-weight').value = '10';
        updateTotal();
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
    });

    test('redistributeUnpinnedWeights redistribui entre GPU e CPU', () => {
        const cpuWeight = document.querySelector('.cpu-weight');
        cpuWeight.value = '30';
        redistributeUnpinnedWeights(cpuWeight);
        const gpuSum = [...document.querySelectorAll('.gpu-weight')]
            .reduce((s, w) => s + parseInt(w.value, 10), 0);
        expect(gpuSum + parseInt(cpuWeight.value, 10)).toBe(100);
    });

    test('applyGpuWeightsToUI aplica peso da CPU por device', () => {
        applyGpuWeightsToUI([
            { index: 0, weight: 60, active: true, is_main: true, device: 'gpu' },
            { index: 1, weight: 10, active: true, device: 'gpu' },
            { index: -1, weight: 30, active: true, device: 'cpu' },
        ]);
        expect(document.querySelector('.cpu-weight').value).toBe('30');
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
    });

    test('bindGpuManualListeners dispara em cpu-weight', () => {
        bindGpuManualListeners();
        state.manualGpuOverride = false;
        document.querySelector('.cpu-weight').dispatchEvent(new Event('input'));
        expect(state.manualGpuOverride).toBe(true);
    });

    test('onGpuPinToggle funciona na linha da CPU', () => {
        const pin = document.querySelector('.cpu-pin');
        pin.checked = true;
        onGpuPinToggle(pin);
        expect(document.querySelector('.cpu-weight').classList.contains('ring-2')).toBe(true);
    });
});

describe('CPU desmarcada — contrato de offload', () => {
    beforeEach(() => {
        setupGpuDom({ gpuCount: 2, withCpu: true, cpuChecked: false });
    });

    test('updateTotal ignora peso da CPU desmarcada', () => {
        document.querySelectorAll('.gpu-row')[0].querySelector('.gpu-weight').value = '60';
        document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-weight').value = '40';
        document.querySelector('.cpu-weight').value = '30';
        document.querySelector('.cpu-checkbox').checked = false;
        updateTotal();
        expect(document.querySelector('.cpu-weight').value).toBe('0');
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
    });

    test('redistributeUnpinnedWeights exclui CPU desmarcada', () => {
        document.querySelector('.cpu-checkbox').checked = false;
        document.querySelector('.cpu-weight').value = '0';
        document.querySelectorAll('.gpu-row')[0].querySelector('.gpu-weight').value = '70';
        const changed = document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-weight');
        changed.value = '20';
        redistributeUnpinnedWeights(changed);
        const gpuSum = [...document.querySelectorAll('.gpu-weight')]
            .reduce((s, w) => s + parseInt(w.value, 10), 0);
        expect(gpuSum).toBe(100);
        expect(document.querySelector('.cpu-weight').value).toBe('0');
    });

    test('validateDeviceWeights aceita somente GPUs quando CPU desmarcada', () => {
        const result = validateDeviceWeights([
            { index: 0, weight: 60, active: true, device: 'gpu' },
            { index: 1, weight: 40, active: true, device: 'gpu' },
            { index: -1, weight: 30, active: false, device: 'cpu' },
        ]);
        expect(result.ok).toBe(true);
    });

    test('validateDeviceWeights rejeita GPUs parciais com CPU desmarcada', () => {
        const result = validateDeviceWeights([
            { index: 0, weight: 70, active: true, device: 'gpu' },
            { index: -1, weight: 0, active: false, device: 'cpu' },
        ]);
        expect(result.ok).toBe(false);
        expect(result.message).toContain('100%');
    });

    test('applyGpuWeightsToUI desmarca CPU conforme payload', () => {
        applyGpuWeightsToUI([
            { index: 0, weight: 100, active: true, is_main: true, device: 'gpu' },
            { index: -1, weight: 0, active: false, device: 'cpu' },
        ]);
        expect(document.querySelector('.cpu-checkbox').checked).toBe(false);
        expect(document.querySelector('.cpu-weight').value).toBe('0');
    });
});

describe('auto-balance budget UI', () => {
    beforeEach(() => {
        setupGpuDom({ gpuCount: 3, withCpu: true, cpuChecked: true });
    });

    test('applyGpuWeightsToUI zera CPU stale quando payload omitido', () => {
        document.querySelector('.cpu-checkbox').checked = true;
        document.querySelector('.cpu-weight').value = '10';

        applyGpuWeightsToUI([
            { index: 0, weight: 43, active: true, device: 'gpu' },
            { index: 1, weight: 47, active: true, device: 'gpu' },
            { index: 2, weight: 10, active: true, is_main: true, device: 'gpu' },
        ]);

        expect(document.querySelector('.cpu-weight').value).toBe('0');
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
    });

    test('estado valido sem offload GPU 43/47/10 CPU 0 exibe 100%', () => {
        applyGpuWeightsToUI([
            { index: 0, weight: 43, active: true, device: 'gpu' },
            { index: 1, weight: 47, active: true, device: 'gpu' },
            { index: 2, weight: 10, active: true, is_main: true, device: 'gpu' },
            { index: -1, weight: 0, active: true, device: 'cpu' },
        ]);
        expect(document.querySelector('.gpu-row[data-index="0"] .gpu-weight').value).toBe('43');
        expect(document.querySelector('.gpu-row[data-index="1"] .gpu-weight').value).toBe('47');
        expect(document.querySelector('.gpu-row[data-index="2"] .gpu-weight').value).toBe('10');
        expect(document.querySelector('.cpu-weight').value).toBe('0');
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
        expect(validateDeviceWeights(collectDeviceWeightsFromUI()).ok).toBe(true);
    });

    test('estado valido com offload CPU 10 GPU 39/42/9 exibe 100%', () => {
        applyGpuWeightsToUI([
            { index: 0, weight: 39, active: true, device: 'gpu' },
            { index: 1, weight: 42, active: true, device: 'gpu' },
            { index: 2, weight: 9, active: true, is_main: true, device: 'gpu' },
            { index: -1, weight: 10, active: true, device: 'cpu' },
        ]);
        expect(document.querySelector('.gpu-row[data-index="0"] .gpu-weight').value).toBe('39');
        expect(document.querySelector('.gpu-row[data-index="1"] .gpu-weight').value).toBe('42');
        expect(document.querySelector('.gpu-row[data-index="2"] .gpu-weight').value).toBe('9');
        expect(document.querySelector('.cpu-weight').value).toBe('10');
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
        expect(validateDeviceWeights(collectDeviceWeightsFromUI()).ok).toBe(true);
    });

    test('applyGpuWeightsToUI cpu valve ativa com peso zero mantem 100%', () => {
        applyGpuWeightsToUI([
            { index: 0, weight: 43, active: true, device: 'gpu' },
            { index: 1, weight: 47, active: true, device: 'gpu' },
            { index: 2, weight: 10, active: true, is_main: true, device: 'gpu' },
            { index: -1, weight: 0, active: true, device: 'cpu' },
        ]);
        expect(document.querySelector('.cpu-checkbox').checked).toBe(true);
        expect(document.querySelector('.cpu-weight').value).toBe('0');
        expect(document.getElementById('total-percent').innerText).toBe('CARGA TOTAL: 100%');
    });
});

describe('collectDeviceWeightsFromUI — distribuição percentual', () => {
    beforeEach(() => {
        setupGpuDom({ gpuCount: 2, withCpu: true, cpuChecked: true });
    });

    test('envia pesos apenas de dispositivos marcados', () => {
        document.querySelectorAll('.gpu-row')[0].querySelector('.gpu-weight').value = '50';
        document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-weight').value = '20';
        document.querySelector('.cpu-weight').value = '30';
        document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-checkbox').checked = false;

        const weights = collectDeviceWeightsFromUI();

        expect(weights).toEqual(expect.arrayContaining([
            expect.objectContaining({ index: 0, weight: 50, active: true, device: 'gpu' }),
            expect.objectContaining({ index: 1, weight: 0, active: false, device: 'gpu' }),
            expect.objectContaining({ index: -1, weight: 30, active: true, device: 'cpu' }),
        ]));
    });

    test('CPU desmarcada envia active false e weight 0', () => {
        document.querySelectorAll('.gpu-row')[0].querySelector('.gpu-weight').value = '100';
        document.querySelector('.cpu-checkbox').checked = false;
        document.querySelector('.cpu-weight').value = '25';

        const weights = collectDeviceWeightsFromUI();
        const cpu = weights.find(w => w.device === 'cpu');

        expect(cpu).toEqual(expect.objectContaining({
            active: false,
            weight: 0,
        }));
    });

    test('soma dos ativos reflete distribuição 60/40 GPU', () => {
        document.querySelector('.cpu-checkbox').checked = false;
        document.querySelectorAll('.gpu-row')[0].querySelector('.gpu-weight').value = '60';
        document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-checkbox').checked = true;
        document.querySelectorAll('.gpu-row')[1].querySelector('.gpu-weight').value = '40';

        const weights = collectDeviceWeightsFromUI();
        const activeTotal = weights.filter(w => w.active).reduce((s, w) => s + w.weight, 0);

        expect(activeTotal).toBe(100);
    });
});

describe('updateMtpBadge', () => {
    test('updateMtpBadge ON aplica classe amber', () => {
        updateMtpBadge(true);
        const badge = document.getElementById('mtp-badge');
        expect(badge.innerText).toBe('ON');
        expect(badge.className).toContain('text-amber-400');
    });

    test('updateMtpBadge OFF aplica classe slate', () => {
        updateMtpBadge(false);
        const badge = document.getElementById('mtp-badge');
        expect(badge.innerText).toBe('OFF');
        expect(badge.className).toContain('text-slate-500');
    });
});
