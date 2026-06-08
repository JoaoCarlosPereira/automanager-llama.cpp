import { jest, test, expect, beforeEach } from '@jest/globals';
import * as version from './version.js';

const {
    checkForUpdates,
    dismissVersionModal,
    showVersionModal,
    resetVersionCheckState,
} = version;

function setupVersionModalDom() {
    document.body.innerHTML = `
        <div id="dashboard" style="display:block"></div>
        <div id="version-update-modal" class="hidden">
            <div id="version-update-backdrop"></div>
            <span id="version-current-ref"></span>
            <span id="version-remote-ref"></span>
            <div id="version-commits-list"></div>
            <button id="version-dismiss-btn"></button>
        </div>
    `;
}

beforeEach(() => {
    setupVersionModalDom();
    resetVersionCheckState();
    global.fetch = jest.fn();
    sessionStorage.clear();
});

test('checkForUpdates chama apiFetch apenas uma vez', async () => {
    fetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'ok', update_available: false }),
    });

    await checkForUpdates();
    await checkForUpdates();

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/system/version-check');
});

test('checkForUpdates abre modal quando update_available=true', async () => {
    fetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
            status: 'ok',
            update_available: true,
            current_ref: 'abc1234',
            remote_ref: 'def5678',
            commits: [
                {
                    sha: 'fullsha1',
                    message: 'feat: alert',
                    author: 'Dev',
                    date: '2026-06-07T12:00:00-03:00',
                },
            ],
        }),
    });

    await checkForUpdates();

    const modal = document.getElementById('version-update-modal');
    expect(modal.classList.contains('flex')).toBe(true);
    expect(modal.classList.contains('hidden')).toBe(false);
    expect(document.getElementById('version-current-ref').textContent).toBe('abc1234');
    expect(document.getElementById('version-remote-ref').textContent).toBe('def5678');
    expect(document.getElementById('version-commits-list').children.length).toBe(1);
});

test('checkForUpdates nao abre modal quando update_available=false', async () => {
    fetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'ok', update_available: false }),
    });

    await checkForUpdates();

    const modal = document.getElementById('version-update-modal');
    expect(modal.classList.contains('hidden')).toBe(true);
});

test('checkForUpdates nao abre modal quando status=error', async () => {
    fetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'error', update_available: false }),
    });

    await checkForUpdates();

    expect(document.getElementById('version-update-modal').classList.contains('hidden')).toBe(true);
});

test('checkForUpdates nao abre modal quando status=unavailable', async () => {
    fetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'unavailable', update_available: false }),
    });

    await checkForUpdates();

    expect(document.getElementById('version-update-modal').classList.contains('hidden')).toBe(true);
});

test('dismissVersionModal grava sessionStorage e oculta modal', () => {
    showVersionModal({
        current_ref: 'aaa',
        remote_ref: 'bbb',
        commits: [],
    });

    dismissVersionModal();

    expect(sessionStorage.getItem('version-update-dismissed')).toBe('1');
    expect(document.getElementById('version-update-modal').classList.contains('hidden')).toBe(true);
});

test('checkForUpdates respeita dismiss em sessionStorage', async () => {
    sessionStorage.setItem('version-update-dismissed', '1');
    fetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'ok', update_available: true, commits: [] }),
    });

    await checkForUpdates();

    expect(fetch).not.toHaveBeenCalled();
    expect(document.getElementById('version-update-modal').classList.contains('hidden')).toBe(true);
});

test('tecla Esc fecha o modal', () => {
    showVersionModal({ current_ref: 'a', remote_ref: 'b', commits: [] });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(document.getElementById('version-update-modal').classList.contains('hidden')).toBe(true);
});

test('clique no backdrop fecha o modal', () => {
    showVersionModal({ current_ref: 'a', remote_ref: 'b', commits: [] });
    document.getElementById('version-update-backdrop').click();
    expect(document.getElementById('version-update-modal').classList.contains('hidden')).toBe(true);
});
