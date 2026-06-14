import { apiFetch } from './auth.js?v=4.0.2';

const DISMISS_KEY = 'version-update-dismissed';
let checked = false;

function getModal() {
    return document.getElementById('version-update-modal');
}

function hideVersionModal() {
    const modal = getModal();
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

function formatCommitDate(isoDate) {
    if (!isoDate) return '';
    try {
        return new Date(isoDate).toLocaleString('pt-BR', {
            dateStyle: 'short',
            timeStyle: 'short',
        });
    } catch (e) {
        return isoDate;
    }
}

function bindModalDismissHandlers() {
    const modal = getModal();
    if (!modal || modal.dataset.bound === '1') return;
    modal.dataset.bound = '1';

    const backdrop = document.getElementById('version-update-backdrop');
    if (backdrop) {
        backdrop.addEventListener('click', dismissVersionModal);
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
            dismissVersionModal();
        }
    });
}

export function showVersionModal(data) {
    const modal = getModal();
    const list = document.getElementById('version-commits-list');
    if (!modal || !list) return;

    const currentRef = document.getElementById('version-current-ref');
    const remoteRef = document.getElementById('version-remote-ref');
    if (currentRef) currentRef.textContent = data.current_ref || '--';
    if (remoteRef) remoteRef.textContent = data.remote_ref || '--';

    list.innerHTML = '';
    const commits = Array.isArray(data.commits) ? data.commits : [];
    if (commits.length === 0) {
        list.innerHTML = '<p class="text-sm text-slate-500">Nenhum commit listado.</p>';
    } else {
        commits.forEach((commit) => {
            const item = document.createElement('article');
            item.className = 'p-4 rounded-2xl border border-slate-800 bg-slate-900/40';
            const shortSha = (commit.sha || '').slice(0, 7);
            item.innerHTML = `
                <p class="text-sm text-slate-100 leading-relaxed whitespace-pre-wrap break-words">${escapeHtml(commit.message || '')}</p>
                <p class="text-[10px] text-slate-500 mt-2 font-mono">
                    <span class="text-blue-400">${escapeHtml(shortSha)}</span>
                    <span class="mx-2">·</span>${escapeHtml(commit.author || '')}
                    <span class="mx-2">·</span>${escapeHtml(formatCommitDate(commit.date))}
                </p>`;
            list.appendChild(item);
        });
    }

    bindModalDismissHandlers();
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    const dismissBtn = document.getElementById('version-dismiss-btn');
    if (dismissBtn) dismissBtn.focus();
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

export function dismissVersionModal() {
    sessionStorage.setItem(DISMISS_KEY, '1');
    hideVersionModal();
}

export async function checkForUpdates() {
    if (checked) return;
    checked = true;

    const dashboard = document.getElementById('dashboard');
    if (!dashboard || dashboard.style.display === 'none') return;
    if (sessionStorage.getItem(DISMISS_KEY) === '1') return;

    try {
        const res = await apiFetch('/api/system/version-check');
        if (!res.ok) return;
        const data = await res.json();
        if (data.status !== 'ok' || !data.update_available) return;
        showVersionModal(data);
    } catch (e) {
        // falha silenciosa — dashboard continua operacional
    }
}

/** Reseta estado interno (usado em testes). */
export function resetVersionCheckState() {
    checked = false;
    sessionStorage.removeItem(DISMISS_KEY);
    const modal = getModal();
    if (modal) {
        delete modal.dataset.bound;
    }
}
