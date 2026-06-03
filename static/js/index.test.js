import { jest, test, expect, beforeEach, afterEach } from '@jest/globals';

/** Funções expostas em window por index.js (wiring para onclick no HTML). */
const WINDOW_WIRING = [
    'initDashboard',
    'startDashboardPolling',
    'stopDashboardPolling',
    'handleLogin',
    'handleLogout',
    'changePassword',
    'apiFetch',
    'cancelAutoBalance',
    'onContextSizePresetChange',
    'onContextSizeCustomInput',
    'getContextSize',
    'setContextSize',
    'stopModel',
    'startModel',
    'renameModel',
    'deleteModel',
    'setDefaultModel',
    'selectModel',
    'applyModelConfig',
    'renewToken',
    'updateTotal',
    'balanceWeights',
    'resetToDefaults',
    'hideAutoBalanceCapacityAlert',
    'getModelButtonsHtml',
    'updateModels',
    'updateStatus',
    'downloadModel',
    'updateMetrics',
    'updateDownloads',
];

function setupIndexDom({ dashboardDisplay = 'none', withInitWidgets = false } = {}) {
    const initWidgets = withInitWidgets
        ? `
        <span id="total-percent"></span>
        <div id="status-badge"></div>
        <div id="active-card"></div>
        `
        : '';
    document.body.innerHTML = `
        <a id="chat-link" href="#"></a>
        <span id="api-link"></span>
        <div id="dashboard" style="display:${dashboardDisplay}"></div>
        ${initWidgets}
    `;
    window.fixedIp = '10.0.0.5';
}

beforeEach(() => {
    jest.resetModules();
    jest.useFakeTimers();
    setupIndexDom({ dashboardDisplay: 'none' });
    global.fetch = jest.fn();
});

afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
});

async function loadIndex() {
    return import('./index.js');
}

test('exposes critical functions on window after dynamic import', async () => {
    await loadIndex();

    for (const name of WINDOW_WIRING) {
        expect(typeof window[name]).toBe('function');
    }
});

test('window wiring references the same functions as source modules', async () => {
    const auth = await import('./auth.js');
    const gpu = await import('./gpu.js');
    const metrics = await import('./metrics.js');
    const models = await import('./models.js');

    await loadIndex();

    expect(window.handleLogin).toBe(auth.handleLogin);
    expect(window.apiFetch).toBe(auth.apiFetch);
    expect(window.cancelAutoBalance).toBe(gpu.cancelAutoBalance);
    expect(window.getContextSize).toBe(gpu.getContextSize);
    expect(window.startDashboardPolling).toBe(metrics.startDashboardPolling);
    expect(window.stopDashboardPolling).toBe(metrics.stopDashboardPolling);
    expect(window.updateMetrics).toBe(metrics.updateMetrics);
    expect(window.updateDownloads).toBe(metrics.updateDownloads);
    expect(window.initDashboard).toBe(models.initDashboard);
    expect(window.selectModel).toBe(models.selectModel);
});

test('initializes window.modelConfigs when absent', async () => {
    delete window.modelConfigs;

    await loadIndex();

    expect(window.modelConfigs).toEqual({});
});

test('preserves existing window.modelConfigs', async () => {
    window.modelConfigs = { '/models/a.gguf': { context_size: 8192 } };

    await loadIndex();

    expect(window.modelConfigs['/models/a.gguf'].context_size).toBe(8192);
});

test('sets chat-link href and api-link text from window.fixedIp', async () => {
    window.fixedIp = '192.168.0.42';

    await loadIndex();

    expect(document.getElementById('chat-link').href).toBe('http://192.168.0.42:8085/');
    expect(document.getElementById('api-link').innerText).toBe('http://192.168.0.42:8085/v1');
});

test('auto-inits dashboard when #dashboard is visible', async () => {
    jest.resetModules();
    setupIndexDom({ dashboardDisplay: 'block', withInitWidgets: true });
    fetch.mockResolvedValue({ ok: true, json: async () => ({ running: false }) });

    const { state } = await import('./state.js');
    await loadIndex();
    await Promise.resolve();

    expect(state.metricsTimer).not.toBeNull();
    expect(fetch.mock.calls.map((c) => c[0])).toContain('/status');
});

test('skips auto-init when #dashboard display is none', async () => {
    const { state } = await import('./state.js');

    await loadIndex();

    expect(state.metricsTimer).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
});

test('auto-inits when dashboard display is empty (not none)', async () => {
    jest.resetModules();
    setupIndexDom({ dashboardDisplay: '', withInitWidgets: true });
    fetch.mockResolvedValue({ ok: true, json: async () => ({ running: false }) });

    const { state } = await import('./state.js');
    await loadIndex();

    expect(state.metricsTimer).not.toBeNull();
});
