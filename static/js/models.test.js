import { jest, test, expect, beforeEach, afterEach } from '@jest/globals';
import { resetAuthSessionFlags, sessionExpiredHandled } from './auth.js';
import { state } from './state.js';
import * as models from './models.js';

const {
    initDashboard,
    getModelButtonsHtml,
    selectModel,
    applyModelConfig,
    setDefaultModel,
    downloadModel,
    updateModels,
    renameModel,
    deleteModel,
    startModel,
    stopModel,
} = models;

function setupConstants() {
    window.__constants = {
        CONTEXT_PRESET_VALUES: [8192, 32768, 65536],
        DEFAULT_CONTEXT_SIZE: 65536,
        CONTEXT_K_MULTIPLIER: 1024,
        DEFAULT_PARALLEL_SLOTS: 4,
        DEFAULT_BATCH_SIZE: 512,
    };
    window.modelConfigs = {};
    window.fixedIp = '192.168.1.10';
}

function setupGpuControls() {
    document.body.insertAdjacentHTML('beforeend', `
        <div id="log-box"></div>
        <div id="status-badge"></div>
        <span id="total-percent"></span>
        <button id="auto-balance-cancel-btn" class="hidden"></button>
        <div class="gpu-row" data-index="0">
            <input type="checkbox" class="gpu-checkbox" checked />
            <input type="radio" class="gpu-main-radio" name="main" checked />
            <input type="number" class="gpu-weight" value="100" />
            <input type="checkbox" class="gpu-pin" />
        </div>
        <div class="gpu-row" data-index="1">
            <input type="checkbox" class="gpu-checkbox" />
            <input type="radio" class="gpu-main-radio" name="main" />
            <input type="number" class="gpu-weight" value="0" />
            <input type="checkbox" class="gpu-pin" />
        </div>
        <select id="context-size">
            <option value="8192">8K</option>
            <option value="32768">32K</option>
            <option value="65536" selected>64K</option>
            <option value="custom">Custom</option>
        </select>
        <div id="context-size-custom-wrap" class="hidden">
            <input id="context-size-custom" value="64" />
        </div>
        <select id="mmproj-path"><option value="">Auto</option></select>
        <select id="split-mode"><option value="layer">layer</option></select>
        <input id="parallel-slots" value="4" />
        <input id="batch-size" value="512" />
        <input id="auto-balance-toggle" type="checkbox" />
        <input id="auto-balance-badge" class="hidden" />
    `);
}

function setupModelsListDom() {
    document.body.insertAdjacentHTML('beforeend', `
        <span id="model-count"></span>
        <div id="model-list-container"></div>
        <input id="download-url" value="" />
    `);
}

beforeEach(() => {
    document.body.innerHTML = '';
    setupConstants();
    setupGpuControls();
    setupModelsListDom();
    resetAuthSessionFlags();
    state.currentSelectedModel = null;
    state.currentRunningModelPath = null;
    state.manualGpuOverride = true;
    state.autoBalancePending = false;

    global.fetch = jest.fn();
    global.alert = jest.fn();
    global.confirm = jest.fn(() => true);
    global.prompt = jest.fn(() => 'new-model-name');

    window.updateStatus = jest.fn();
    window.updateMetrics = jest.fn();
    window.updateDownloads = jest.fn();
    window.updateModels = jest.fn();
    window.stopDashboardPolling = jest.fn();
    jest.useFakeTimers();
});

afterEach(() => {
    jest.useRealTimers();
});

test('getModelButtonsHtml com isRunning=true inclui ABRIR INTERFACE e ENCERRAR', () => {
    const html = getModelButtonsHtml('/models/a.gguf', 'hash-1', true);
    expect(html).toContain('ABRIR INTERFACE');
    expect(html).toContain('stopModel()');
    expect(html).toContain('uptime-val');
    expect(html).toContain('192.168.1.10:8085');
});

test('getModelButtonsHtml com isRunning=false inclui botao CARREGAR', () => {
    const html = getModelButtonsHtml('/models/b.gguf', 'hash-2', false);
    expect(html).toContain('CARREGAR');
    expect(html).toContain("startModel('/models/b.gguf', 'hash-2')");
    expect(html).not.toContain('ABRIR INTERFACE');
});

test('selectModel marca item ativo e aplica config quando existe', () => {
    const path = '/media/models/test.gguf';
    window.modelConfigs[path] = {
        context_size: 32768,
        parallel_slots: 8,
        batch_size: 256,
        split_mode: 'row',
        mmproj_path: '',
        auto_balance: false,
    };
    document.body.insertAdjacentHTML('beforeend', `
        <div id="item-1" class="model-item-container"></div>
        <div data-path="${path}"><p class="model-name">Test</p></div>
    `);

    selectModel(path, 'item-1');

    expect(state.currentSelectedModel).toBe(path);
    expect(document.getElementById('item-1').classList.contains('active-selection')).toBe(true);
    expect(document.getElementById('parallel-slots').value).toBe('8');
});

test('selectModel sem config chama resetToDefaults', () => {
    document.body.insertAdjacentHTML('beforeend', `
        <div id="item-2" class="model-item-container"></div>
    `);
    const path = '/media/models/other.gguf';

    selectModel(path, 'item-2');

    expect(state.currentSelectedModel).toBe(path);
    expect(document.getElementById('context-size').value).toBe('65536');
});

test('applyModelConfig retorna cedo se config ausente', () => {
    document.getElementById('parallel-slots').value = '99';
    applyModelConfig('/inexistente.gguf');
    expect(document.getElementById('parallel-slots').value).toBe('99');
});

test('applyModelConfig preenche campos e pesos GPU', () => {
    const path = '/media/models/cfg.gguf';
    window.modelConfigs[path] = {
        context_size: 8192,
        parallel_slots: 2,
        batch_size: 128,
        split_mode: 'layer',
        mmproj_path: '/proj/v.gguf',
        auto_balance: true,
        auto_balance_profile: true,
        gpu_weights: [
            { index: 0, weight: 60, active: true, is_main: true, pinned: true },
            { index: 1, weight: 40, active: true, is_main: false, pinned: false },
        ],
    };
    document.getElementById('mmproj-path').innerHTML = '<option value="">Auto</option>';
    document.body.insertAdjacentHTML('beforeend', `
        <div data-path="${path}"><p class="model-name">Cfg</p></div>
    `);

    applyModelConfig(path);

    expect(document.getElementById('context-size').value).toBe('8192');
    expect(document.getElementById('parallel-slots').value).toBe('2');
    expect(document.querySelector('.gpu-row[data-index="0"] .gpu-weight').value).toBe('60');
    expect(state.manualGpuOverride).toBe(false);
});

test('applyModelConfig adiciona opcao mmproj quando nao existe no select', () => {
    const path = '/media/models/mm.gguf';
    const mmPath = '/projectors/new.gguf';
    window.modelConfigs[path] = { mmproj_path: mmPath };
    document.getElementById('mmproj-path').innerHTML = '<option value="">Auto</option>';

    applyModelConfig(path);

    const select = document.getElementById('mmproj-path');
    expect(select.value).toBe(mmPath);
    expect(select.options.length).toBe(2);
});

test('setDefaultModel envia path quando checkbox marcado', async () => {
    fetch.mockResolvedValue({ ok: true });
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    const other = document.createElement('input');
    other.type = 'checkbox';
    other.className = 'model-default-checkbox';
    other.checked = true;
    document.body.append(other);

    await setDefaultModel(cb, '/models/a.gguf');

    expect(fetch).toHaveBeenCalledWith('/set_default', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ path: '/models/a.gguf' }),
    }));
    expect(other.checked).toBe(false);
});

test('setDefaultModel envia null quando desmarcado', async () => {
    fetch.mockResolvedValue({ ok: true });
    const cb = document.createElement('input');
    cb.checked = false;

    await setDefaultModel(cb, '/models/a.gguf');

    expect(JSON.parse(fetch.mock.calls[0][1].body).path).toBeNull();
});

test('setDefaultModel mostra alert em erro de rede', async () => {
    fetch.mockRejectedValue(new Error('network'));
    const cb = document.createElement('input');
    cb.checked = true;

    await setDefaultModel(cb, '/models/a.gguf');

    expect(alert).toHaveBeenCalledWith('Erro ao salvar configuracao.');
});

test('downloadModel com URL vazia nao chama fetch', async () => {
    document.getElementById('download-url').value = '   ';
    await downloadModel();
    expect(fetch).not.toHaveBeenCalled();
});

test('downloadModel com URL valida limpa input e chama updateDownloads', async () => {
    document.getElementById('download-url').value = 'https://example.com/model.gguf';
    fetch.mockResolvedValue({ ok: true });

    await downloadModel();

    expect(fetch).toHaveBeenCalledWith('/downloads', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ url: 'https://example.com/model.gguf' }),
    }));
    expect(document.getElementById('download-url').value).toBe('');
    expect(window.updateDownloads).toHaveBeenCalled();
});

test('downloadModel ignora erro de rede', async () => {
    document.getElementById('download-url').value = 'https://x.gguf';
    fetch.mockRejectedValue(new Error('fail'));
    await expect(downloadModel()).resolves.toBeUndefined();
});

test('updateModels renderiza lista e projectors', async () => {
    const modelsPayload = {
        models: [{
            id: 'm1',
            path: '/media/a.gguf',
            name: 'Model A',
            dir: '/media',
            last_config: { context_size: 65536, hardware_incapable: false },
        }],
        projectors: [{ path: '/proj/p.gguf', name: 'Proj' }],
    };
    fetch
        .mockResolvedValueOnce({ ok: true, json: async () => modelsPayload })
        .mockResolvedValueOnce({ ok: true, json: async () => ({ default_model: '/media/a.gguf' }) });

    await updateModels();

    expect(document.getElementById('model-count').innerText).toBe('1 UNIDADES');
    expect(document.getElementById('model-list-container').innerHTML).toContain('Model A');
    expect(document.getElementById('mmproj-path').innerHTML).toContain('Proj');
    expect(window.modelConfigs['/media/a.gguf']).toBeDefined();
});

test('updateModels retorna cedo quando sessao expirada', async () => {
    fetch.mockResolvedValue({
        status: 401,
        ok: false,
        clone: () => ({ json: async () => ({ detail: 'expirada' }) }),
    });

    await updateModels();

    expect(document.getElementById('model-count').textContent).toBe('');
});

test('updateModels ignora resposta nao ok', async () => {
    fetch
        .mockResolvedValueOnce({ ok: false, status: 500 })
        .mockResolvedValueOnce({ ok: true, json: async () => ({ default_model: null }) });

    await updateModels();

    expect(document.getElementById('model-count').textContent).toBe('');
});

test('updateModels ignora erro de rede', async () => {
    fetch.mockRejectedValue(new Error('network'));
    await expect(updateModels()).resolves.toBeUndefined();
});

test('updateModels marca modelo em execucao', async () => {
    state.currentRunningModelPath = '/media/run.gguf';
    fetch
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                models: [{
                    id: 'r1',
                    path: '/media/run.gguf',
                    name: 'Running',
                    dir: '/media',
                }],
                projectors: [],
            }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ default_model: null }),
        });

    await updateModels();

    expect(document.getElementById('model-list-container').innerHTML).toContain('running-now');
    expect(document.getElementById('model-list-container').innerHTML).toContain('ABRIR INTERFACE');
});

test('renameModel cancelado quando prompt retorna null', async () => {
    prompt.mockReturnValue(null);
    await renameModel('/media/old.gguf');
    expect(fetch).not.toHaveBeenCalled();
});

test('renameModel nao envia quando nome igual', async () => {
    prompt.mockReturnValue('old');
    await renameModel('/media/old.gguf');
    expect(fetch).not.toHaveBeenCalled();
});

test('renameModel com sucesso chama updateModels', async () => {
    prompt.mockReturnValue('renamed');
    fetch.mockResolvedValue({ ok: true });

    await renameModel('/media/old.gguf');

    expect(fetch).toHaveBeenCalledWith('/rename', expect.objectContaining({
        body: JSON.stringify({ path: '/media/old.gguf', new_name: 'renamed' }),
    }));
});

test('renameModel com erro da API mostra alert', async () => {
    prompt.mockReturnValue('x');
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({ detail: 'nome invalido' }),
    });

    await renameModel('/media/old.gguf');

    expect(alert).toHaveBeenCalledWith('Erro ao renomear: nome invalido');
});

test('renameModel com erro de rede', async () => {
    prompt.mockReturnValue('x');
    fetch.mockRejectedValue(new Error('net'));

    await renameModel('/media/old.gguf');

    expect(alert).toHaveBeenCalledWith('Erro de rede ao renomear modelo.');
});

test('deleteModel com confirm false nao exclui', async () => {
    confirm.mockReturnValue(false);
    await deleteModel('/media/del.gguf');
    expect(fetch).not.toHaveBeenCalled();
});

test('deleteModel com confirm true chama delete e updateModels', async () => {
    confirm.mockReturnValue(true);
    fetch.mockResolvedValue({ ok: true });

    await deleteModel('/media/del.gguf');

    expect(fetch).toHaveBeenCalledWith('/delete', expect.objectContaining({
        body: JSON.stringify({ path: '/media/del.gguf' }),
    }));
});

test('deleteModel com erro da API', async () => {
    confirm.mockReturnValue(true);
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({}),
    });

    await deleteModel('/media/del.gguf');

    expect(alert).toHaveBeenCalledWith('Erro ao excluir: Erro desconhecido');
});

test('deleteModel com erro de rede', async () => {
    confirm.mockReturnValue(true);
    fetch.mockRejectedValue(new Error('net'));

    await deleteModel('/media/del.gguf');

    expect(alert).toHaveBeenCalledWith('Erro de rede ao excluir modelo.');
});

test('startModel alerta quando nenhuma GPU ativa', async () => {
    const path = '/media/m.gguf';
    state.currentSelectedModel = path;
    document.querySelectorAll('.gpu-checkbox').forEach(cb => { cb.checked = false; });

    await startModel(path, 'id1');

    expect(alert).toHaveBeenCalledWith('SELECIONE PELO MENOS UMA GPU');
    expect(fetch).not.toHaveBeenCalledWith('/start', expect.anything());
});

test('startModel alerta quando GPU principal ausente', async () => {
    const path = '/media/m.gguf';
    state.currentSelectedModel = path;
    document.querySelectorAll('.gpu-main-radio').forEach(r => { r.checked = false; });

    await startModel(path, 'id1');

    expect(alert).toHaveBeenCalledWith('DEFINA A GPU PRINCIPAL (coluna Principal)');
});

test('startModel alerta contexto invalido', async () => {
    const path = '/media/m.gguf';
    state.currentSelectedModel = path;
    document.getElementById('context-size').value = 'custom';
    document.getElementById('context-size-custom').value = '0';

    await startModel(path, 'id1');

    expect(alert).toHaveBeenCalledWith(
        'Informe um contexto válido em K (mínimo 1). Ex.: 100 = 100K tokens por slot.',
    );
});

test('startModel hardware incapaz cancelado no confirm', async () => {
    const path = '/media/incap.gguf';
    window.modelConfigs[path] = { hardware_incapable: true };
    confirm.mockReturnValueOnce(false);

    await startModel(path, 'id-inc');

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('INCOMPATÍVEL'));
    expect(fetch).not.toHaveBeenCalledWith('/start', expect.anything());
});

test('startModel hardware incapaz prossegue quando confirm true', async () => {
    const path = '/media/incap2.gguf';
    window.modelConfigs[path] = {
        hardware_incapable: true,
        hardware_incapable_message: 'VRAM insuficiente',
    };
    confirm.mockReturnValue(true);
    fetch.mockResolvedValue({ ok: true, json: async () => ({}) });
    state.currentSelectedModel = path;
    document.body.insertAdjacentHTML('beforeend', '<div id="id-inc2" class="model-item-container"></div>');

    await startModel(path, 'id-inc2');

    expect(fetch).toHaveBeenCalledWith('/start', expect.objectContaining({ method: 'POST' }));
});

test('startModel seleciona modelo diferente antes de iniciar', async () => {
    const path = '/media/sel.gguf';
    fetch.mockResolvedValue({ ok: true, json: async () => ({}) });
    document.body.insertAdjacentHTML('beforeend', '<div id="sel-id" class="model-item-container"></div>');

    const p = startModel(path, 'sel-id');
    await jest.advanceTimersByTimeAsync(100);
    await p;

    expect(state.currentSelectedModel).toBe(path);
});

test('startModel com sucesso agenda updateStatus e probing', async () => {
    const path = '/media/ok.gguf';
    state.currentSelectedModel = path;
    fetch.mockResolvedValue({
        ok: true,
        json: async () => ({ probing: true }),
    });

    await startModel(path, 'ok-id');

    expect(state.autoBalancePending).toBe(true);
    expect(fetch).toHaveBeenCalledWith('/start', expect.objectContaining({
        body: expect.stringContaining('"path":"/media/ok.gguf"'),
    }));
    jest.advanceTimersByTime(2000);
    expect(window.updateStatus).toHaveBeenCalled();
});

test('startModel com erro da API', async () => {
    const path = '/media/err.gguf';
    state.currentSelectedModel = path;
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({ detail: 'falhou' }),
    });

    await startModel(path, 'err-id');

    expect(alert).toHaveBeenCalledWith('Erro ao iniciar: falhou');
});

test('startModel com sessao expirada nao alerta erro de API', async () => {
    const path = '/media/sess.gguf';
    state.currentSelectedModel = path;
    fetch.mockResolvedValue({
        status: 401,
        ok: false,
        clone: () => ({ json: async () => ({ detail: 'Sessao expirada' }) }),
    });

    await startModel(path, 'sess-id');

    expect(alert).not.toHaveBeenCalledWith(expect.stringMatching(/^Erro ao iniciar:/));
});

test('startModel com erro de rede', async () => {
    const path = '/media/net.gguf';
    state.currentSelectedModel = path;
    fetch.mockRejectedValue(new Error('network'));

    await startModel(path, 'net-id');

    expect(alert).toHaveBeenCalledWith('Erro ao iniciar modelo.');
});

test('startModel manual override limpa badge quando nao auto balance', async () => {
    const path = '/media/man.gguf';
    state.currentSelectedModel = path;
    state.manualGpuOverride = true;
    window.modelConfigs[path] = { auto_balance_profile: true };
    document.getElementById('auto-balance-toggle').checked = false;
    fetch.mockResolvedValue({ ok: true, json: async () => ({ probing: false }) });

    await startModel(path, 'man-id');

    expect(state.manualGpuOverride).toBe(false);
    expect(window.modelConfigs[path].auto_balance_profile).toBe(false);
});

test('stopModel com confirm false nao chama stop', async () => {
    confirm.mockReturnValue(false);
    await stopModel();
    expect(fetch).not.toHaveBeenCalledWith('/stop', expect.anything());
});

test('stopModel com confirm true chama stop e agenda updateStatus', async () => {
    confirm.mockReturnValue(true);
    fetch.mockResolvedValue({ ok: true });

    await stopModel();

    expect(fetch).toHaveBeenCalledWith('/stop', { method: 'POST' });
    jest.advanceTimersByTime(1000);
    expect(window.updateStatus).toHaveBeenCalled();
});

test('stopModel ignora quando sessao expirada', async () => {
    const { handleSessionExpired } = await import('./auth.js');
    confirm.mockReturnValue(true);
    handleSessionExpired('expirada');
    fetch.mockResolvedValue({ ok: true });

    await stopModel();
    jest.advanceTimersByTime(1000);

    expect(window.updateStatus).not.toHaveBeenCalled();
});

test('initDashboard chama rotinas de inicializacao', () => {
    initDashboard();

    expect(window.updateStatus).toHaveBeenCalled();
    expect(window.updateMetrics).toHaveBeenCalled();
    expect(window.updateDownloads).toHaveBeenCalled();
    expect(window.updateModels).toHaveBeenCalled();
});

test('applyModelConfig limpa mmproj quando config vazio', () => {
    const path = '/media/models/empty-mm.gguf';
    document.getElementById('mmproj-path').innerHTML =
        '<option value="">Auto</option><option value="/p.gguf">P</option>';
    document.getElementById('mmproj-path').value = '/p.gguf';
    window.modelConfigs[path] = { mmproj_path: '' };

    applyModelConfig(path);

    expect(document.getElementById('mmproj-path').value).toBe('');
});

test('selectModel ignora elemento inexistente', () => {
    selectModel('/media/x.gguf', 'missing-id');
    expect(state.currentSelectedModel).toBe('/media/x.gguf');
});

test('applyModelConfig com auto balance e gpu sem pin', () => {
    const path = '/media/pinless.gguf';
    document.querySelector('.gpu-row[data-index="1"] .gpu-pin')?.remove();
    window.modelConfigs[path] = {
        auto_balance: true,
        auto_balance_profile: 'fast',
        gpu_weights: [{ index: 1, weight: 50, active: undefined, is_main: false }],
    };

    applyModelConfig(path);

    expect(document.getElementById('auto-balance-toggle').checked).toBe(true);
    expect(document.querySelector('.gpu-row[data-index="1"] .gpu-checkbox').checked).toBe(true);
});

test('setDefaultModel nao desmarca outros quando checkbox desligado', async () => {
    fetch.mockResolvedValue({ ok: true });
    const cb = document.createElement('input');
    cb.checked = false;
    const other = document.createElement('input');
    other.className = 'model-default-checkbox';
    other.checked = true;
    document.body.append(other);

    await setDefaultModel(cb, '/models/z.gguf');

    expect(other.checked).toBe(true);
});

test('downloadModel nao limpa input quando resposta nao ok', async () => {
    document.getElementById('download-url').value = 'https://fail.gguf';
    fetch.mockResolvedValue({ ok: false });

    await downloadModel();

    expect(document.getElementById('download-url').value).toBe('https://fail.gguf');
    expect(window.updateDownloads).toHaveBeenCalled();
});

test('updateModels com modelo selecionado ativo', async () => {
    state.currentSelectedModel = '/media/sel.gguf';
    fetch
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                models: [{
                    id: 'sel1',
                    path: '/media/sel.gguf',
                    name: 'Sel',
                    dir: '/media',
                    last_config: { context_size: 65536 },
                }],
                projectors: [],
            }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ default_model: '/media/sel.gguf' }),
        });

    await updateModels();

    expect(document.getElementById('model-list-container').innerHTML)
        .toContain('active-selection');
});

test('updateModels reseta mmproj quando valor salvo nao existe mais', async () => {
    document.getElementById('mmproj-path').value = '/proj/removed.gguf';
    fetch
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ models: [], projectors: [] }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ default_model: null }),
        });

    await updateModels();

    expect(document.getElementById('mmproj-path').value).toBe('');
});

test('startModel hardware incapaz inclui mensagem detalhada no confirm', async () => {
    const path = '/media/det.gguf';
    state.currentSelectedModel = path;
    window.modelConfigs[path] = {
        hardware_incapable: true,
        hardware_incapable_message: 'VRAM 24GB insuficiente',
    };
    confirm.mockReturnValue(false);

    await startModel(path, 'det-id');

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('VRAM 24GB insuficiente'));
});

test('updateModels renderiza modelo hardware incapaz', async () => {
    fetch
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                models: [{
                    id: 'inc',
                    path: '/media/bad.gguf',
                    name: 'Bad',
                    dir: '/media',
                    last_config: { hardware_incapable: true },
                }],
                projectors: [],
            }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ default_model: null }),
        });

    await updateModels();

    const html = document.getElementById('model-list-container').innerHTML;
    expect(html).toContain('Incapaz');
    expect(html).toContain('data-hardware-incapable="true"');
});

test('updateModels mantem valor mmproj quando lista de projectors nao muda', async () => {
    const proj = { path: '/proj/x.gguf', name: 'X' };
    document.getElementById('mmproj-path').innerHTML =
        '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>' +
        '<option value="/proj/x.gguf" class="bg-slate-900">X</option>';
    document.getElementById('mmproj-path').value = '/proj/x.gguf';

    fetch
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ models: [], projectors: [proj] }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ default_model: null }),
        });
    await updateModels();

    expect(document.getElementById('mmproj-path').value).toBe('/proj/x.gguf');
});
