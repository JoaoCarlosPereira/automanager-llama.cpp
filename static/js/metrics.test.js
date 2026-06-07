import { jest, test, expect, beforeEach, afterEach, describe } from '@jest/globals';
import { TextEncoder, TextDecoder } from 'util';

const mockApplyGpuWeightsToUI = jest.fn();
const mockSetContextSize = jest.fn();
const mockHideAutoBalanceCapacityAlert = jest.fn();
const mockShowAutoBalanceCapacityAlert = jest.fn();
const mockUpdateAutoBalanceProfileBadge = jest.fn();
const mockSyncAutoBalanceCancelButton = jest.fn();
const mockUpdateThinkingBadge = jest.fn();

jest.unstable_mockModule('./gpu.js', () => ({
    applyGpuWeightsToUI: mockApplyGpuWeightsToUI,
    getContextSize: jest.fn(() => 8192),
    setContextSize: mockSetContextSize,
    hideAutoBalanceCapacityAlert: mockHideAutoBalanceCapacityAlert,
    showAutoBalanceCapacityAlert: mockShowAutoBalanceCapacityAlert,
    updateAutoBalanceProfileBadge: mockUpdateAutoBalanceProfileBadge,
    syncAutoBalanceCancelButton: mockSyncAutoBalanceCancelButton,
    updateThinkingBadge: mockUpdateThinkingBadge,
}));

const auth = await import('./auth.js');
const { state } = await import('./state.js');
const {
    updateMetrics,
    updateStatus,
    updateUptime,
    startLogs,
    stopDashboardPolling,
    startDashboardPolling,
    ensureStatusPolling,
    updateDownloads,
    renewToken,
} = await import('./metrics.js');

function resetState() {
    state.logStream = null;
    state.startTime = null;
    state.currentSelectedModel = null;
    state.currentRunningModelPath = null;
    state.manualGpuOverride = false;
    state.autoBalancePending = false;
    state.metricsTimer = null;
    state.downloadsTimer = null;
    state.modelsTimer = null;
    state.statusPollIntervalMs = 3000;
    state.statusPollTimer = null;
}

function setupMetricsDom(extra = '') {
    document.body.innerHTML = `
        <div id="status-badge"></div>
        <div id="active-card" class="hidden">
            <span id="active-model-name"></span>
        </div>
        <a id="chat-link" href="#"></a>
        <div id="metrics-panel"></div>
        <span id="cpu-val"></span>
        <div id="cpu-bar" style="width:0"></div>
        <span id="ram-val"></span>
        <div id="ram-bar" style="width:0"></div>
        <div class="gpu-row" data-index="0">
            <span class="gpu-util-val"></span>
            <div class="gpu-util-bar" style="width:0"></div>
            <span class="gpu-temp-val"></span>
            <span class="gpu-power-val"></span>
            <span class="gpu-vram-text"></span>
            <div class="gpu-vram-bar" style="width:0"></div>
        </div>
        <span id="uptime-val"></span>
        <div id="log-box"></div>
        <div id="download-status"></div>
        <input id="context-size" value="8192"/>
        <input id="context-size-custom" value=""/>
        <input id="parallel-slots" value="1"/>
        <input id="batch-size" value="512"/>
        <input id="mmproj-path" value=""/>
        <input id="auto-balance-toggle" type="checkbox"/>
        <button id="auto-balance-cancel-btn" class="hidden"></button>
        <div id="auto-balance-capacity-alert" class="hidden">
            <p id="auto-balance-capacity-msg"></p>
            <ul id="auto-balance-capacity-details"></ul>
            <ul id="auto-balance-capacity-suggestions"></ul>
        </div>
        <span id="api-token"></span>
        ${extra}
    `;
}

function text(el) {
    return el ? (el.textContent ?? '') : '';
}

function mockFetchStatus(data, { withLogs = false } = {}) {
    fetch.mockImplementation((url) => {
        if (withLogs && url === '/logs') {
            return Promise.resolve({
                body: { getReader: () => ({ read: async () => ({ done: true }) }) },
            });
        }
        if (url === '/status' || url === '/metrics' || url === '/downloads') {
            return Promise.resolve({ ok: true, json: async () => data });
        }
        return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
}

function modelItemHtml(path, id = 'item-1') {
    return `
        <div class="model-item-container" data-path="${path}" id="${id}">
            <div class="action-btn-container">old</div>
            <button class="rename-btn"></button>
            <button class="delete-btn"></button>
        </div>
    `;
}

let intervalId = 0;
const intervalCallbacks = new Map();

beforeEach(() => {
    global.TextEncoder = TextEncoder;
    global.TextDecoder = TextDecoder;
    resetState();
    auth.resetAuthSessionFlags();
    setupMetricsDom();
    window.modelConfigs = {};
    window.updateModels = jest.fn();
    window.getModelButtonsHtml = jest.fn(() => '<button>go</button>');
    global.fetch = jest.fn();
    global.alert = jest.fn();
    global.confirm = jest.fn(() => true);
    console.error = jest.fn();

    intervalId = 0;
    intervalCallbacks.clear();
    jest.spyOn(global, 'setInterval').mockImplementation((fn, ms) => {
        const id = ++intervalId;
        intervalCallbacks.set(id, { fn, ms });
        return id;
    });
    jest.spyOn(global, 'clearInterval').mockImplementation((id) => {
        intervalCallbacks.delete(id);
    });

    mockApplyGpuWeightsToUI.mockClear();
    mockSetContextSize.mockClear();
    mockHideAutoBalanceCapacityAlert.mockClear();
    mockShowAutoBalanceCapacityAlert.mockClear();
    mockUpdateAutoBalanceProfileBadge.mockClear();
    mockSyncAutoBalanceCancelButton.mockClear();
    window.stopDashboardPolling = jest.fn();
});

afterEach(() => {
    jest.restoreAllMocks();
    stopDashboardPolling();
});

describe('updateMetrics', () => {
    test('ignora gpu sem linha correspondente no DOM', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                cpu: 1,
                ram: 2,
                gpus: [{ index: 9, util: 10, mem_used: 1, mem_total: 2, vram_pct: 50 }],
            }),
        });

        await updateMetrics();

        expect(document.getElementById('cpu-val').innerText).toBe('1%');
    });

    test('funciona sem metrics-panel no DOM', async () => {
        document.getElementById('metrics-panel').remove();
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({ cpu: 5, ram: 6, gpus: [] }),
        });

        await expect(updateMetrics()).resolves.toBeUndefined();
        expect(document.getElementById('cpu-val').innerText).toBe('5%');
    });

    test('atualiza CPU, RAM e GPU com dados validos', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                cpu: 42,
                ram: 67,
                gpus: [{
                    index: 0,
                    util: 85,
                    temp: 72,
                    power: 240,
                    mem_used: 12000,
                    mem_total: 24564,
                    vram_pct: 48,
                }],
            }),
        });
        state.currentRunningModelPath = '/models/a.gguf';

        await updateMetrics();

        expect(document.getElementById('cpu-val').innerText).toBe('42%');
        expect(document.getElementById('cpu-bar').style.width).toBe('42%');
        expect(document.getElementById('ram-val').innerText).toBe('67%');
        expect(document.getElementById('metrics-panel').classList.contains('metric-dimmed')).toBe(false);
        const row = document.querySelector('.gpu-row[data-index="0"]');
        expect(row.querySelector('.gpu-util-val').innerText).toBe('85%');
        expect(row.querySelector('.gpu-vram-text').innerText).toBe('12000 / 24564 MB');
    });

    test('aplica metric-dimmed sem modelo em execucao', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({ cpu: 10, ram: 20, gpus: [] }),
        });

        await updateMetrics();

        expect(document.getElementById('metrics-panel').classList.contains('metric-dimmed')).toBe(true);
    });

    test('ignora quando sessao expirada', async () => {
        auth.handleSessionExpired();
        fetch.mockResolvedValue({ ok: true, json: async () => ({ cpu: 99, ram: 99, gpus: [] }) });

        await updateMetrics();

        expect(text(document.getElementById('cpu-val'))).toBe('');
    });

    test('ignora resposta nao ok', async () => {
        fetch.mockResolvedValue({ ok: false });

        await updateMetrics();

        expect(text(document.getElementById('cpu-val'))).toBe('');
    });

    test('catch de rede nao propaga erro', async () => {
        fetch.mockRejectedValue(new Error('network'));

        await expect(updateMetrics()).resolves.toBeUndefined();
    });

    test('atualiza linha da CPU com usage e RAM em MB', async () => {
        setupMetricsDom(`
            <div class="cpu-row" data-index="cpu">
                <span class="cpu-util-val"></span>
                <div class="cpu-util-bar" style="width:0"></div>
                <span class="cpu-ram-val"></span>
                <span class="cpu-ram-text"></span>
                <div class="cpu-ram-bar" style="width:0"></div>
            </div>
        `);
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                cpu: 55,
                ram: 40,
                ram_used_mb: 25600,
                ram_total_mb: 64000,
                gpus: [],
            }),
        });

        await updateMetrics();

        const cpuRow = document.querySelector('.cpu-row');
        expect(cpuRow.querySelector('.cpu-util-val').innerText).toBe('55%');
        expect(cpuRow.querySelector('.cpu-util-bar').style.width).toBe('55%');
        expect(cpuRow.querySelector('.cpu-ram-val').innerText).toBe('25600 / 64000 MB');
        expect(cpuRow.querySelector('.cpu-ram-text').innerText).toBe('25600 / 64000 MB');
        expect(cpuRow.querySelector('.cpu-ram-bar').style.width).toBe('40%');
    });
});

describe('updateUptime', () => {
    test('calcula uptime a partir de serverStartTime', () => {
        const now = Math.floor(Date.now() / 1000);
        updateUptime(now - 3661);

        expect(document.getElementById('uptime-val').innerText).toMatch(/1h 1m 1s/);
    });

    test('usa state.startTime quando serverStartTime ausente', () => {
        state.startTime = new Date(Date.now() - 125000);
        updateUptime(null);

        expect(document.getElementById('uptime-val').innerText).toMatch(/\d+h \d+m \d+s/);
    });

    test('retorna cedo sem serverStartTime nem state.startTime', () => {
        updateUptime(null);
        expect(text(document.getElementById('uptime-val'))).toBe('');
    });
});

describe('updateStatus', () => {
    test('running=true mostra ONLINE e active card', async () => {
        mockFetchStatus({
            running: true,
            model: 'Llama-3',
            model_path: 'C:/models/a.gguf',
            start_time: Math.floor(Date.now() / 1000) - 60,
            config: { path: 'C:/models/a.gguf', gpu_weights: [100, 0] },
        }, { withLogs: true });

        await updateStatus();

        const badge = document.getElementById('status-badge');
        expect(badge.innerHTML).toContain('ONLINE');
        expect(document.getElementById('active-card').classList.contains('hidden')).toBe(false);
        expect(document.getElementById('active-model-name').innerText).toBe('Llama-3');
        const chatLink = document.getElementById('chat-link');
        expect(chatLink.classList.contains('opacity-40')).toBe(false);
    });

    test('running=false mostra OFFLINE e oculta active card', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({ running: false }),
        });
        state.logStream = { abort: jest.fn() };

        await updateStatus();

        const badge = document.getElementById('status-badge');
        expect(badge.innerHTML).toContain('OFFLINE');
        expect(document.getElementById('active-card').classList.contains('hidden')).toBe(true);
        expect(state.logStream).toBeNull();
        expect(document.getElementById('chat-link').getAttribute('aria-disabled')).toBe('true');
    });

    test('recovery auto_balance mostra badge AUTO BALANCE', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                running: false,
                recovery: {
                    active: true,
                    auto_balance: true,
                    message: 'calibrando',
                    gpu_weights: [50, 50],
                },
            }),
        });

        await updateStatus();

        expect(document.getElementById('status-badge').innerHTML).toContain('AUTO BALANCE');
        expect(mockSyncAutoBalanceCancelButton).toHaveBeenCalledWith(true);
        expect(mockApplyGpuWeightsToUI).toHaveBeenCalledWith([50, 50], true);
    });

    test('recovery ativa sem auto_balance mostra REALOCANDO', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                running: false,
                recovery: { active: true, auto_balance: false },
            }),
        });

        await updateStatus();

        expect(document.getElementById('status-badge').innerHTML).toContain('REALOCANDO');
    });

    test('recovery failed com hardware_capacity_exceeded', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                running: false,
                recovery: {
                    failed: true,
                    hardware_capacity_exceeded: true,
                    message: 'VRAM insuficiente',
                },
            }),
        });

        await updateStatus();

        expect(document.getElementById('status-badge').innerHTML).toContain('HARDWARE INSUFICIENTE');
        expect(mockShowAutoBalanceCapacityAlert).toHaveBeenCalled();
        expect(window.updateModels).toHaveBeenCalled();
    });

    test('recovery failed generico', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                running: false,
                recovery: { failed: true, message: 'erro gpu' },
            }),
        });

        await updateStatus();

        expect(document.getElementById('status-badge').innerHTML).toContain('FALHA: ERRO GPU');
    });

    test('autoBalancePending cancelado atualiza badge', async () => {
        state.autoBalancePending = true;
        document.getElementById('auto-balance-toggle').checked = true;
        mockFetchStatus({
            running: false,
            recovery: { active: false, cancelled: true },
        });

        await updateStatus();

        expect(document.getElementById('status-badge').innerHTML).toContain('AUTO BALANCE CANCELADO');
        expect(mockHideAutoBalanceCapacityAlert).toHaveBeenCalled();
        expect(state.autoBalancePending).toBe(false);
    });

    test('autoBalancePending sucesso aplica pesos finais', async () => {
        state.autoBalancePending = true;
        state.currentSelectedModel = '/models/x.gguf';
        mockFetchStatus({
            running: true,
            model: 'X',
            model_path: '/models/x.gguf',
            config: { path: '/models/x.gguf', gpu_weights: [60, 40] },
            recovery: { active: false, failed: false, gpu_weights: [70, 30] },
        }, { withLogs: true });

        await updateStatus();

        expect(mockApplyGpuWeightsToUI).toHaveBeenCalledWith([70, 30], false);
        expect(window.updateModels).toHaveBeenCalled();
        expect(window.modelConfigs['/models/x.gguf'].auto_balance).toBe(false);
    });

    test('autoBalancePending hardware incapaz alerta e marca config', async () => {
        state.autoBalancePending = true;
        state.currentSelectedModel = '/models/big.gguf';
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                running: false,
                recovery: {
                    active: false,
                    failed: true,
                    hardware_capacity_exceeded: true,
                    message: 'Modelo grande',
                },
            }),
        });

        await updateStatus();

        expect(alert).toHaveBeenCalledWith('Modelo grande');
        expect(window.modelConfigs['/models/big.gguf'].hardware_incapable).toBe(true);
    });

    test('atualiza botoes de modelos em execucao', async () => {
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('C:/models/a.gguf'));
        mockFetchStatus({
            running: true,
            model: 'A',
            model_path: 'C:/models/a.gguf',
            start_time: Math.floor(Date.now() / 1000),
            config: { path: 'C:/models/a.gguf' },
        }, { withLogs: true });

        await updateStatus();

        const item = document.querySelector('.model-item-container');
        expect(item.classList.contains('running-now')).toBe(true);
        expect(item.querySelector('.rename-btn').classList.contains('hidden')).toBe(true);
        expect(window.getModelButtonsHtml).toHaveBeenCalled();
    });

    test('retorna cedo com sessao expirada ou resposta nao ok', async () => {
        auth.handleSessionExpired();
        await updateStatus();
        expect(fetch).toHaveBeenCalled();

        auth.resetAuthSessionFlags();
        fetch.mockResolvedValue({ ok: false });
        await updateStatus();
        expect(document.getElementById('status-badge').innerHTML).toBe('');
    });

    test('catch loga erro sem propagar', async () => {
        fetch.mockRejectedValue(new Error('boom'));
        await expect(updateStatus()).resolves.toBeUndefined();
        expect(console.error).toHaveBeenCalled();
    });

    test('merge modelConfigs quando running com config.path', async () => {
        state.currentSelectedModel = '/models/a.gguf';
        mockFetchStatus({
            running: true,
            model: 'A',
            model_path: '/models/a.gguf',
            start_time: Math.floor(Date.now() / 1000),
            config: {
                path: '/models/a.gguf',
                auto_balance_profile: true,
                gpu_weights: [100, 0],
            },
        }, { withLogs: true });

        await updateStatus();

        expect(mockUpdateAutoBalanceProfileBadge).toHaveBeenCalledWith(true);
        expect(window.modelConfigs['/models/a.gguf'].auto_balance_profile).toBe(true);
    });

    test('running sem modelo selecionado sincroniza slots e batch', async () => {
        mockFetchStatus({
            running: true,
            model: 'A',
            model_path: '/models/a.gguf',
            start_time: Math.floor(Date.now() / 1000),
            config: {
                path: '/models/a.gguf',
                gpu_weights: [100, 0],
                context_size: 16384,
                parallel_slots: 4,
                batch_size: 256,
                mmproj_path: '/mmproj.bin',
            },
        }, { withLogs: true });

        await updateStatus();

        expect(mockSetContextSize).toHaveBeenCalledWith(16384);
        expect(document.getElementById('parallel-slots').value).toBe('4');
        expect(document.getElementById('batch-size').value).toBe('256');
        expect(document.getElementById('mmproj-path').value).toBe('/mmproj.bin');
    });

    test('marca active-selection no modelo selecionado', async () => {
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('/models/sel.gguf'));
        state.currentSelectedModel = '/models/sel.gguf';
        mockFetchStatus({ running: false });

        await updateStatus();

        expect(document.querySelector('.model-item-container').classList.contains('active-selection')).toBe(true);
    });

    test('remove active-selection de modelo nao selecionado', async () => {
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('/models/a.gguf', 'item-a'));
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('/models/b.gguf', 'item-b'));
        document.getElementById('item-b').classList.add('active-selection');
        state.currentSelectedModel = '/models/a.gguf';
        mockFetchStatus({ running: false });

        await updateStatus();

        expect(document.getElementById('item-a').classList.contains('active-selection')).toBe(true);
        expect(document.getElementById('item-b').classList.contains('active-selection')).toBe(false);
    });

    test('atualiza html dos botoes quando conteudo difere', async () => {
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('/models/x.gguf'));
        window.getModelButtonsHtml.mockReturnValue('<button class="new-action">x</button>');
        mockFetchStatus({ running: false });

        await updateStatus();

        expect(document.querySelector('.action-btn-container').innerHTML).toContain('new-action');
    });

    test('modelo parado exibe botoes rename e delete', async () => {
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('/models/off.gguf'));
        mockFetchStatus({ running: false });

        await updateStatus();

        const item = document.querySelector('.model-item-container');
        expect(item.classList.contains('running-now')).toBe(false);
        expect(item.querySelector('.rename-btn').classList.contains('hidden')).toBe(false);
    });

    test('recovery auto_balance sem mensagem usa texto padrao', async () => {
        mockFetchStatus({
            running: false,
            recovery: { active: true, auto_balance: true },
        });

        await updateStatus();

        expect(document.getElementById('status-badge').innerHTML).toContain('CALIBRANDO GPUS...');
    });

    test('nao altera botoes quando html ja coincide', async () => {
        document.body.insertAdjacentHTML('beforeend', modelItemHtml('/models/same.gguf'));
        const container = document.querySelector('.action-btn-container');
        container.innerHTML = '<button>go</button>';
        const htmlBefore = container.innerHTML;
        window.getModelButtonsHtml.mockReturnValue('<button>go</button>');
        mockFetchStatus({ running: false });

        await updateStatus();

        expect(container.innerHTML).toBe(htmlBefore);
    });

    test('running sem chat-link nao lanca erro', async () => {
        document.getElementById('chat-link').remove();
        mockFetchStatus({
            running: true,
            model: 'M',
            model_path: '/m.gguf',
            start_time: Math.floor(Date.now() / 1000),
        }, { withLogs: true });

        await expect(updateStatus()).resolves.toBeUndefined();
    });

    test('offline sem chat-link nao lanca erro', async () => {
        document.getElementById('chat-link').remove();
        mockFetchStatus({ running: false });

        await expect(updateStatus()).resolves.toBeUndefined();
    });

    test('autoBalance sucesso usa gpu_weights do config sem recovery weights', async () => {
        state.autoBalancePending = true;
        mockFetchStatus({
            running: false,
            config: { gpu_weights: [33, 67] },
            recovery: { active: false, failed: false },
        });

        await updateStatus();

        expect(mockApplyGpuWeightsToUI).toHaveBeenCalledWith([33, 67], false);
    });

    test('nao inicia logs se logStream ja existe com servidor online', async () => {
        state.logStream = { abort: jest.fn(), signal: 'keep' };
        const logsFetch = jest.fn();
        fetch.mockImplementation((url) => {
            if (url === '/logs') return logsFetch();
            return Promise.resolve({
                ok: true,
                json: async () => ({
                    running: true,
                    model: 'M',
                    model_path: '/m.gguf',
                    start_time: Math.floor(Date.now() / 1000),
                }),
            });
        });

        await updateStatus();

        expect(logsFetch).not.toHaveBeenCalled();
        expect(state.logStream.signal).toBe('keep');
    });
});

describe('ensureStatusPolling e dashboard polling', () => {
    test('ensureStatusPolling fast usa 1000ms', () => {
        ensureStatusPolling(true);
        expect(state.statusPollIntervalMs).toBe(1000);
        expect(setInterval).toHaveBeenCalled();
        const entry = [...intervalCallbacks.values()].find((v) => v.ms === 1000);
        expect(entry).toBeDefined();
    });

    test('ensureStatusPolling slow usa 3000ms', () => {
        ensureStatusPolling(false);
        expect(state.statusPollIntervalMs).toBe(3000);
        const entry = [...intervalCallbacks.values()].find((v) => v.ms === 3000);
        expect(entry).toBeDefined();
    });

    test('ensureStatusPolling nao recria timer com mesmo intervalo', () => {
        ensureStatusPolling(false);
        const callsBefore = setInterval.mock.calls.length;
        ensureStatusPolling(false);
        expect(setInterval.mock.calls.length).toBe(callsBefore);
    });

    test('ensureStatusPolling troca intervalo e limpa timer anterior', () => {
        state.statusPollTimer = 42;
        state.statusPollIntervalMs = 3000;
        ensureStatusPolling(true);
        expect(clearInterval).toHaveBeenCalledWith(42);
        expect(state.statusPollIntervalMs).toBe(1000);
    });

    test('startDashboardPolling cria quatro timers', () => {
        startDashboardPolling();
        expect(setInterval).toHaveBeenCalledTimes(4);
        const intervals = [...intervalCallbacks.values()].map((v) => v.ms).sort((a, b) => a - b);
        expect(intervals).toEqual([2000, 3000, 3000, 5000]);
    });

    test('stopDashboardPolling limpa timers e logStream', () => {
        state.statusPollTimer = 1;
        state.metricsTimer = 2;
        state.downloadsTimer = 3;
        state.modelsTimer = 4;
        state.logStream = { abort: jest.fn() };

        stopDashboardPolling();

        expect(clearInterval).toHaveBeenCalledTimes(4);
        expect(state.statusPollTimer).toBeNull();
        expect(state.metricsTimer).toBeNull();
        expect(state.downloadsTimer).toBeNull();
        expect(state.modelsTimer).toBeNull();
        expect(state.logStream).toBeNull();
        expect(mockSyncAutoBalanceCancelButton).toHaveBeenCalledWith(false);
    });
});

describe('startLogs', () => {
    test('formata error warn info e limita linhas', async () => {
        const payload = new TextEncoder().encode('error warn info\n');
        let sent = false;
        fetch.mockResolvedValue({
            body: {
                getReader: () => ({
                    read: async () => {
                        if (sent) return { done: true };
                        sent = true;
                        return { value: payload, done: false };
                    },
                }),
            },
        });

        await startLogs();

        const box = document.getElementById('log-box');
        expect(box.innerHTML).toContain('ERRO');
        expect(box.innerHTML).toContain('AVISO');
        expect(box.childNodes.length).toBe(1);
    });

    test('remove linhas antigas acima de 500', async () => {
        let reads = 0;
        fetch.mockResolvedValue({
            body: {
                getReader: () => ({
                    read: async () => {
                        if (reads++ >= 501) return { done: true };
                        return { value: new TextEncoder().encode('x\n'), done: false };
                    },
                }),
            },
        });

        await startLogs();

        expect(document.getElementById('log-box').childNodes.length).toBeLessThanOrEqual(500);
    });

    test('aborta stream anterior antes de novo', async () => {
        const abort = jest.fn();
        state.logStream = { abort, signal: 'old' };
        fetch.mockResolvedValue({
            body: { getReader: () => ({ read: async () => ({ done: true }) }) },
        });

        await startLogs();

        expect(abort).toHaveBeenCalled();
    });

    test('catch de rede nao propaga', async () => {
        fetch.mockRejectedValue(new Error('sse fail'));
        await expect(startLogs()).resolves.toBeUndefined();
    });
});

describe('updateDownloads', () => {
    test('limpa container quando sem downloads', async () => {
        fetch.mockResolvedValue({ ok: true, json: async () => ({}) });
        document.getElementById('download-status').innerHTML = 'pending';

        await updateDownloads();

        expect(document.getElementById('download-status').innerHTML).toBe('');
    });

    test('renderiza download com status failed', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                d1: { filename: 'bad.gguf', status: 'failed', progress: 0 },
            }),
        });

        await updateDownloads();

        expect(document.getElementById('download-status').innerHTML).toContain('Falhou');
        expect(window.updateModels).not.toHaveBeenCalled();
    });

    test('renderiza entradas e chama updateModels ao concluir', async () => {
        fetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                d1: { filename: 'model.gguf', status: 'completed', progress: 100 },
                d2: { filename: 'other.gguf', status: 'downloading', progress: 50 },
            }),
        });

        await updateDownloads();

        const html = document.getElementById('download-status').innerHTML;
        expect(html).toContain('model.gguf');
        expect(html).toContain('Concluído');
        expect(html).toContain('Baixando');
        expect(window.updateModels).toHaveBeenCalled();
    });

    test('ignora erro de rede', async () => {
        fetch.mockRejectedValue(new Error('net'));
        await expect(updateDownloads()).resolves.toBeUndefined();
    });
});

describe('renewToken', () => {
    test('retorna quando usuario cancela confirm', async () => {
        confirm.mockReturnValue(false);
        await renewToken();
        expect(fetch).not.toHaveBeenCalled();
    });

    test('atualiza token e alerta ao sucesso', async () => {
        fetch.mockResolvedValue({ ok: true, json: async () => ({ key: 'nova-chave' }) });
        await renewToken();
        expect(document.getElementById('api-token').innerText).toBe('nova-chave');
        expect(alert).toHaveBeenCalledWith('Nova chave gerada!');
    });

    test('alerta erro ao falhar', async () => {
        fetch.mockRejectedValue(new Error('fail'));
        await renewToken();
        expect(alert).toHaveBeenCalledWith('Erro ao renovar token.');
    });
});
