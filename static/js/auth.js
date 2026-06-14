import { state } from './state.js?v=4.0.6';

export let sessionExpiredHandled = false;

/** Reseta flags de sessão (usado em testes). */
export function resetAuthSessionFlags() {
    sessionExpiredHandled = false;
    state.autoBalancePending = false;
    state.autoBalanceTabId = null;
    state.autoBalanceSeenActive = false;
    state.autoBalanceRunId = null;
}

export function setAuthenticatedShellVisible(visible) {
    const display = visible ? 'flex' : 'none';
    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('main-content');
    if (sidebar) sidebar.style.display = display;
    if (main) main.style.display = display;
}

export function handleSessionExpired(message) {
    if (sessionExpiredHandled) return;
    sessionExpiredHandled = true;
    state.autoBalancePending = false;
    state.autoBalanceTabId = null;
    state.autoBalanceSeenActive = false;
    state.autoBalanceRunId = null;
    window.stopDashboardPolling();
    setAuthenticatedShellVisible(false);
    const dashboard = document.getElementById('dashboard');
    const overlay = document.getElementById('login-overlay');
    if (dashboard) dashboard.style.display = 'none';
    if (overlay) overlay.style.display = 'flex';
    const errEl = document.getElementById('login-error');
    if (errEl) {
        errEl.textContent = message || 'Sessao expirada. Faca login novamente.';
        errEl.classList.remove('hidden');
        errEl.classList.remove('text-red-500');
        errEl.classList.add('text-amber-500');
    }
}

export async function apiFetch(url, options = {}) {
    const res = await fetch(url, { credentials: 'include', ...options });
    if (res.status === 401) {
        let detail = 'Sessao expirada. Faca login novamente.';
        try {
            const body = await res.clone().json();
            if (body && body.detail) detail = body.detail;
        } catch (e) {}
        handleSessionExpired(detail);
    }
    return res;
}

export async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, password}),
        });
        if (res.ok) {
            sessionExpiredHandled = false;
            const errEl = document.getElementById('login-error');
            if (errEl) {
                errEl.textContent = '';
                errEl.classList.add('hidden');
                errEl.classList.remove('text-amber-500');
                errEl.classList.add('text-red-500');
            }
            document.getElementById('login-overlay').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            setAuthenticatedShellVisible(true);
            window.initDashboard();
            window.startDashboardPolling();
        } else {
            const err = await res.json();
            const el = document.getElementById('login-error');
            el.textContent = err.detail || 'Erro no login';
            el.classList.remove('hidden');
        }
    } catch (e) {
        const el = document.getElementById('login-error');
        el.textContent = 'Erro de rede';
        el.classList.remove('hidden');
    }
}

export async function handleLogout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
    } catch (e) {}
    sessionExpiredHandled = false;
    window.stopDashboardPolling?.();
    setAuthenticatedShellVisible(false);
    const dashboard = document.getElementById('dashboard');
    const overlay = document.getElementById('login-overlay');
    if (dashboard) dashboard.style.display = 'none';
    if (overlay) overlay.style.display = 'flex';
    location.reload();
}

export async function changePassword() {
    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const statusEl = document.getElementById('password-change-status');

    statusEl.textContent = '';
    statusEl.className = 'text-[10px] font-bold min-h-[1rem]';

    if (!currentPassword || !newPassword) {
        showAlert('Informe a senha atual e a nova senha.');
        return;
    }

    if (newPassword.length < 6) {
        showAlert('A nova senha deve ter pelo menos 6 caracteres.');
        return;
    }

    try {
        const res = await fetch('/api/auth/change-password', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: currentPassword, password: newPassword}),
        });
        if (res.ok) {
            document.getElementById('current-password').value = '';
            document.getElementById('new-password').value = '';
            statusEl.textContent = 'Senha alterada com sucesso.';
            statusEl.classList.add('text-emerald-500');
        } else {
            const err = await res.json();
            statusEl.textContent = err.detail || 'Erro ao alterar senha.';
            statusEl.classList.add('text-red-500');
        }
    } catch (e) {
        statusEl.textContent = 'Erro de rede ao alterar senha.';
        statusEl.classList.add('text-red-500');
    }
}

export function showAlert(msg) { alert(msg); }
export function showConfirm(msg) { return confirm(msg); }

export async function handleShutdown() {
    if (!confirm('DESLIGAR O SISTEMA?\nO servidor sera desligado imediatamente.')) return;
    try {
        const res = await apiFetch('/system/shutdown', {method: 'POST'});
        if (res.ok) {
            alert('Comando de desligamento enviado. O sistema sera desligado em breve.');
        } else {
            const err = await res.json();
            alert('Erro ao desligar: ' + (err.detail || 'Erro desconhecido'));
        }
    } catch (e) {
        alert('Erro de rede ao desligar.');
    }
}

export async function handleUpdate() {
    if (!confirm('ATUALIZAR E REINICIAR?\nO servidor atualizara o codigo e sera reiniciado.')) return;
    try {
        const res = await apiFetch('/system/update', {method: 'POST'});
        if (res.ok) {
            alert('Comando de atualizacao enviado. O servidor sera reiniciado em breve.');
        } else {
            const err = await res.json();
            alert('Erro ao atualizar: ' + (err.detail || 'Erro desconhecido'));
        }
    } catch (e) {
        alert('Erro de rede ao atualizar.');
    }
}
