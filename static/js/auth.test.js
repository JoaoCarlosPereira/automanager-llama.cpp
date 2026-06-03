import { jest, test, expect, beforeEach } from '@jest/globals';
import * as auth from './auth.js';
import { state } from './state.js';

const {
    handleLogin,
    handleLogout,
    changePassword,
    handleSessionExpired,
    apiFetch,
    showAlert,
    showConfirm,
    resetAuthSessionFlags,
} = auth;

function setupLoginDom() {
    document.body.innerHTML = `
        <div id="login-overlay" style="display:flex">
            <form id="login-form">
                <input id="login-username" value="admin"/>
                <input id="login-password" value="secret"/>
            </form>
        </div>
        <div id="dashboard" style="display:none"></div>
        <p id="login-error" class="hidden"></p>
    `;
}

beforeEach(() => {
    setupLoginDom();
    resetAuthSessionFlags();
    global.fetch = jest.fn();
    global.alert = jest.fn();
    global.confirm = jest.fn(() => true);
    window.initDashboard = jest.fn();
    window.startDashboardPolling = jest.fn();
    window.stopDashboardPolling = jest.fn();
    delete window.location;
    window.location = { reload: jest.fn() };
});

test('handleLogin mostra erro ao falhar autenticacao', async () => {
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({ detail: 'Credenciais invalidas' }),
    });

    const event = { preventDefault: jest.fn() };
    await handleLogin(event);

    expect(event.preventDefault).toHaveBeenCalled();
    const errEl = document.getElementById('login-error');
    expect(errEl.textContent).toBe('Credenciais invalidas');
    expect(errEl.classList.contains('hidden')).toBe(false);
    expect(document.getElementById('login-overlay').style.display).toBe('flex');
});

test('handleLogin sucesso sem elemento login-error', async () => {
    document.getElementById('login-error')?.remove();
    fetch.mockResolvedValue({ ok: true });

    await handleLogin({ preventDefault: jest.fn() });

    expect(document.getElementById('dashboard').style.display).toBe('block');
    expect(window.initDashboard).toHaveBeenCalled();
});

test('handleLogin falha sem detail usa mensagem padrao', async () => {
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({}),
    });

    await handleLogin({ preventDefault: jest.fn() });

    expect(document.getElementById('login-error').textContent).toBe('Erro no login');
});

test('handleLogin esconde overlay e mostra dashboard ao sucesso', async () => {
    fetch.mockResolvedValue({ ok: true });

    const event = { preventDefault: jest.fn() };
    await handleLogin(event);

    expect(document.getElementById('login-overlay').style.display).toBe('none');
    expect(document.getElementById('dashboard').style.display).toBe('block');
    expect(window.initDashboard).toHaveBeenCalled();
    expect(window.startDashboardPolling).toHaveBeenCalled();
    expect(auth.sessionExpiredHandled).toBe(false);
});

test('handleLogin mostra erro de rede', async () => {
    fetch.mockRejectedValue(new Error('network'));

    await handleLogin({ preventDefault: jest.fn() });

    expect(document.getElementById('login-error').textContent).toBe('Erro de rede');
});

test('handleLogout chama logout e recarrega pagina', async () => {
    fetch.mockResolvedValue({ ok: true });

    await handleLogout();

    expect(fetch).toHaveBeenCalledWith('/api/auth/logout', { method: 'POST' });
    expect(window.location.reload).toHaveBeenCalled();
});

test('changePassword valida campos vazios via showAlert', async () => {
    document.body.innerHTML += `
        <input id="current-password" value=""/>
        <input id="new-password" value=""/>
        <p id="password-change-status"></p>
    `;

    await changePassword();

    expect(alert).toHaveBeenCalledWith('Informe a senha atual e a nova senha.');
});

test('changePassword valida senha curta via showAlert', async () => {
    document.body.innerHTML += `
        <input id="current-password" value="old"/>
        <input id="new-password" value="123"/>
        <p id="password-change-status"></p>
    `;

    await changePassword();

    expect(alert).toHaveBeenCalledWith('A nova senha deve ter pelo menos 6 caracteres.');
});

test('changePassword erro API sem detail usa fallback', async () => {
    document.body.innerHTML += `
        <input id="current-password" value="oldpass"/>
        <input id="new-password" value="newpass1"/>
        <p id="password-change-status"></p>
    `;
    fetch.mockResolvedValue({ ok: false, json: async () => ({}) });

    await changePassword();

    expect(document.getElementById('password-change-status').textContent).toBe('Erro ao alterar senha.');
});

test('changePassword com erro da API mostra mensagem', async () => {
    document.body.innerHTML += `
        <input id="current-password" value="oldpass"/>
        <input id="new-password" value="newpass1"/>
        <p id="password-change-status"></p>
    `;
    fetch.mockResolvedValue({
        ok: false,
        json: async () => ({ detail: 'Senha atual incorreta' }),
    });

    await changePassword();

    const statusEl = document.getElementById('password-change-status');
    expect(statusEl.textContent).toBe('Senha atual incorreta');
    expect(statusEl.classList.contains('text-red-500')).toBe(true);
});

test('changePassword com erro de rede', async () => {
    document.body.innerHTML += `
        <input id="current-password" value="oldpass"/>
        <input id="new-password" value="newpass1"/>
        <p id="password-change-status"></p>
    `;
    fetch.mockRejectedValue(new Error('network'));

    await changePassword();

    const statusEl = document.getElementById('password-change-status');
    expect(statusEl.textContent).toBe('Erro de rede ao alterar senha.');
});

test('apiFetch com 401 sem JSON usa mensagem padrao', async () => {
    fetch.mockResolvedValue({
        status: 401,
        clone: () => ({ json: async () => { throw new Error('parse'); } }),
    });

    await apiFetch('/protected');

    expect(document.getElementById('login-error').textContent).toContain('Sessao expirada');
});

test('changePassword com sucesso limpa campos e mostra mensagem', async () => {
    document.body.innerHTML += `
        <input id="current-password" value="oldpass"/>
        <input id="new-password" value="newpass1"/>
        <p id="password-change-status"></p>
    `;
    fetch.mockResolvedValue({ ok: true });

    await changePassword();

    expect(document.getElementById('current-password').value).toBe('');
    expect(document.getElementById('new-password').value).toBe('');
    const statusEl = document.getElementById('password-change-status');
    expect(statusEl.textContent).toBe('Senha alterada com sucesso.');
    expect(statusEl.classList.contains('text-emerald-500')).toBe(true);
});

test('handleSessionExpired sem elementos de DOM opcionais', () => {
    document.getElementById('login-error')?.remove();
    document.getElementById('dashboard')?.remove();

    expect(() => handleSessionExpired()).not.toThrow();
    expect(auth.sessionExpiredHandled).toBe(true);
});

test('handleSessionExpired restaura overlay e para polling', () => {
    document.getElementById('dashboard').style.display = 'block';
    document.getElementById('login-overlay').style.display = 'none';

    handleSessionExpired('Sessao expirada.');

    expect(auth.sessionExpiredHandled).toBe(true);
    expect(window.stopDashboardPolling).toHaveBeenCalled();
    expect(document.getElementById('dashboard').style.display).toBe('none');
    expect(document.getElementById('login-overlay').style.display).toBe('flex');
    expect(document.getElementById('login-error').textContent).toBe('Sessao expirada.');
});

test('handleSessionExpired ignora chamadas duplicadas', () => {
    handleSessionExpired('primeira');
    window.stopDashboardPolling.mockClear();
    handleSessionExpired('segunda');
    expect(window.stopDashboardPolling).not.toHaveBeenCalled();
});

test('apiFetch com 401 e detail no body', async () => {
    fetch.mockResolvedValue({
        status: 401,
        clone: () => ({ json: async () => ({ detail: 'Token revogado' }) }),
    });

    await apiFetch('/status');

    expect(document.getElementById('login-error').textContent).toBe('Token revogado');
});

test('apiFetch com 401 chama handleSessionExpired', async () => {
    fetch.mockResolvedValue({
        status: 401,
        clone: () => ({
            json: async () => ({ detail: 'Token invalido' }),
        }),
    });

    const res = await apiFetch('/status');

    expect(res.status).toBe(401);
    expect(auth.sessionExpiredHandled).toBe(true);
});

test('apiFetch com 200 retorna response', async () => {
    const mockRes = { status: 200, ok: true };
    fetch.mockResolvedValue(mockRes);

    const res = await apiFetch('/metrics');

    expect(res).toBe(mockRes);
    expect(auth.sessionExpiredHandled).toBe(false);
});

test('showAlert chama alert', () => {
    showAlert('mensagem');
    expect(alert).toHaveBeenCalledWith('mensagem');
});

test('showConfirm chama confirm e retorna resultado', () => {
    confirm.mockReturnValue(false);
    expect(showConfirm('continuar?')).toBe(false);
    expect(confirm).toHaveBeenCalledWith('continuar?');
});

test('sessionExpiredHandled e exportado e mutavel via handleSessionExpired', () => {
    handleSessionExpired('teste');
    expect(auth.sessionExpiredHandled).toBe(true);
    resetAuthSessionFlags();
    expect(auth.sessionExpiredHandled).toBe(false);
});
