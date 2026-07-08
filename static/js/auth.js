import { state } from './state.js?v=4.2.1';

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

/** Reseta a flag de sessão expirada para permitir nova tentativa de requisições. */
export function resetSessionExpiredFlag() {
    sessionExpiredHandled = false;
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
        resetSessionExpiredFlag();
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            credentials: 'include',
            body: JSON.stringify({username, password}),
        });
        if (res.ok) {
            state.initialTabsSynced = false;
            const errEl = document.getElementById('login-error');
            if (errEl) {
                errEl.textContent = '';
                errEl.classList.add('hidden');
                errEl.classList.remove('text-amber-500');
                errEl.classList.add('text-red-500');
            }
            document.getElementById('login-overlay').style.display = 'none';
            document.getElementById('dashboard').style.display = 'flex';
            setAuthenticatedShellVisible(true);
            window.refreshApiToken?.();
            await window.initDashboard();
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
    resetSessionExpiredFlag();
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
    statusEl.className = 'text-ui-body-sm font-bold min-h-[1rem]';

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
            body: JSON.stringify({current: currentPassword, new: newPassword}),
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

const TOAST_ICONS = {
    error: 'fa-circle-exclamation',
    success: 'fa-circle-check',
    info: 'fa-circle-info',
};

/** Notificação não-bloqueante no canto inferior direito. */
export function showToast(message, type = 'info', timeout = 4500) {
    const container = document.getElementById('toast-container');
    if (!container) { console.log(`[toast:${type}] ${message}`); return; }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
    const icon = document.createElement('i');
    icon.className = `toast-icon fas ${TOAST_ICONS[type] || TOAST_ICONS.info}`;
    const text = document.createElement('span');
    text.className = 'flex-1';
    text.textContent = message;
    toast.append(icon, text);
    container.appendChild(toast);

    requestAnimationFrame(() => toast.classList.add('show'));
    const remove = () => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 220);
    };
    toast.addEventListener('click', remove);
    if (timeout > 0) setTimeout(remove, timeout);
}

export function showAlert(msg, type = 'error') { showToast(msg, type); }

/** Confirmação estilizada; retorna Promise<boolean>. Substitui window.confirm. */
export function showConfirm(msg, { danger = true, confirmLabel = 'Confirmar', cancelLabel = 'Cancelar' } = {}) {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 z-[110] flex items-center justify-center p-4';
        const accent = danger ? 'bg-red-600 hover:bg-red-500' : 'bg-blue-600 hover:bg-blue-500';
        overlay.innerHTML = `
            <div class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"></div>
            <div class="relative glass border border-slate-700 rounded-2xl w-full max-w-md p-6 shadow-2xl" role="dialog" aria-modal="true">
                <p class="text-slate-200 text-ui-body whitespace-pre-line mb-6"></p>
                <div class="flex justify-end gap-3">
                    <button data-act="cancel" class="px-5 py-2.5 rounded-xl text-ui-body-sm font-black uppercase tracking-widest text-slate-300 border border-slate-700 hover:bg-slate-800 transition-all"></button>
                    <button data-act="ok" class="px-5 py-2.5 rounded-xl text-ui-body-sm font-black uppercase tracking-widest text-white transition-all active:scale-95 ${accent}"></button>
                </div>
            </div>`;
        overlay.querySelector('p').textContent = msg;
        overlay.querySelector('[data-act="cancel"]').textContent = cancelLabel;
        overlay.querySelector('[data-act="ok"]').textContent = confirmLabel;

        const done = (result) => {
            document.removeEventListener('keydown', onKey);
            overlay.remove();
            resolve(result);
        };
        const onKey = (e) => {
            if (e.key === 'Escape') done(false);
            else if (e.key === 'Enter') done(true);
        };

        overlay.querySelector('[data-act="ok"]').addEventListener('click', () => done(true));
        overlay.querySelector('[data-act="cancel"]').addEventListener('click', () => done(false));
        overlay.querySelector('.absolute').addEventListener('click', () => done(false));
        document.addEventListener('keydown', onKey);
        document.body.appendChild(overlay);
        overlay.querySelector('[data-act="ok"]').focus();
    });
}

/** Entrada de texto estilizada; retorna Promise<string|null>. Substitui window.prompt. */
export function showPrompt(msg, initialValue = '', { confirmLabel = 'Salvar', cancelLabel = 'Cancelar' } = {}) {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 z-[110] flex items-center justify-center p-4';
        overlay.innerHTML = `
            <div class="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"></div>
            <div class="relative glass border border-slate-700 rounded-2xl w-full max-w-md p-6 shadow-2xl" role="dialog" aria-modal="true">
                <label class="text-slate-300 text-ui-body-sm font-bold block mb-3"></label>
                <input type="text" class="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 outline-none focus:ring-2 focus:ring-blue-500/50 mb-6">
                <div class="flex justify-end gap-3">
                    <button data-act="cancel" class="px-5 py-2.5 rounded-xl text-ui-body-sm font-black uppercase tracking-widest text-slate-300 border border-slate-700 hover:bg-slate-800 transition-all"></button>
                    <button data-act="ok" class="px-5 py-2.5 rounded-xl text-ui-body-sm font-black uppercase tracking-widest text-white bg-blue-600 hover:bg-blue-500 transition-all active:scale-95"></button>
                </div>
            </div>`;
        overlay.querySelector('label').textContent = msg;
        const input = overlay.querySelector('input');
        input.value = initialValue;
        overlay.querySelector('[data-act="cancel"]').textContent = cancelLabel;
        overlay.querySelector('[data-act="ok"]').textContent = confirmLabel;

        const done = (result) => {
            document.removeEventListener('keydown', onKey);
            overlay.remove();
            resolve(result);
        };
        const submit = () => {
            const v = input.value.trim();
            done(v ? v : null);
        };
        const onKey = (e) => {
            if (e.key === 'Escape') done(null);
            else if (e.key === 'Enter') submit();
        };

        overlay.querySelector('[data-act="ok"]').addEventListener('click', submit);
        overlay.querySelector('[data-act="cancel"]').addEventListener('click', () => done(null));
        overlay.querySelector('.absolute').addEventListener('click', () => done(null));
        document.addEventListener('keydown', onKey);
        document.body.appendChild(overlay);
        input.focus();
        input.select();
    });
}

export async function handleShutdown() {
    if (!await showConfirm('Desligar o sistema? O servidor será desligado imediatamente.', { confirmLabel: 'Desligar' })) return;
    try {
        const res = await apiFetch('/system/shutdown', {method: 'POST'});
        if (res.ok) {
            showToast('Comando de desligamento enviado. O sistema será desligado em breve.', 'info');
        } else {
            const err = await res.json();
            showToast('Erro ao desligar: ' + (err.detail || 'Erro desconhecido'), 'error');
        }
    } catch (e) {
        showToast('Erro de rede ao desligar.', 'error');
    }
}

export async function handleUpdate() {
    if (!await showConfirm('Atualizar e reiniciar? O servidor atualizará o código e será reiniciado.', { danger: false, confirmLabel: 'Atualizar' })) return;
    try {
        const res = await apiFetch('/system/update', {method: 'POST'});
        if (res.ok) {
            showToast('Comando de atualização enviado. O servidor será reiniciado em breve.', 'info');
        } else {
            const err = await res.json();
            showToast('Erro ao atualizar: ' + (err.detail || 'Erro desconhecido'), 'error');
        }
    } catch (e) {
        showToast('Erro de rede ao atualizar.', 'error');
    }
}
