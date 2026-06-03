import { test, expect } from '@playwright/test';
import {
  DEFAULT_MODEL_PATH,
  loginAsAdmin,
  refreshDashboardStatus,
  resetMockState,
  setMockRunning,
} from './helpers';

test.describe.configure({ mode: 'serial' });

test.beforeEach(async ({ request }) => {
  await resetMockState(request);
});

test('metricas aparecem com valores nao-zero', async ({ page }) => {
  await loginAsAdmin(page);

  const metricsResponse = page.waitForResponse(
    (resp) => resp.url().includes('/metrics') && resp.request().method() === 'GET' && resp.ok(),
  );
  await metricsResponse;

  await expect(page.locator('#cpu-val')).toHaveText(/\d+(\.\d+)?%/);
  await expect(page.locator('#ram-val')).toHaveText(/\d+(\.\d+)?%/);
  await expect(page.locator('#cpu-bar')).toHaveAttribute('style', /width:\s*[1-9]/);
});

test('logs SSE exibe linhas no terminal', async ({ page, request }) => {
  await loginAsAdmin(page);
  await setMockRunning(request, true, DEFAULT_MODEL_PATH);
  await refreshDashboardStatus(page);

  await expect(page.locator('#status-badge')).toContainText('ONLINE', { timeout: 15000 });
  await expect(page.locator('#log-box')).toContainText('llama', { timeout: 15000 });
});

test('download progresso aparece apos iniciar download', async ({ page }) => {
  await loginAsAdmin(page);

  await page.locator('#download-url').fill('https://example.com/model.gguf');
  const downloadPost = page.waitForResponse(
    (resp) =>
      resp.url().includes('/downloads') &&
      resp.request().method() === 'POST' &&
      resp.ok(),
  );
  await page.locator('button:has-text("EXECUTAR DOWNLOAD")').click();
  await downloadPost;

  const downloadGet = page.waitForResponse(
    (resp) =>
      resp.url().includes('/downloads') &&
      resp.request().method() === 'GET' &&
      resp.ok(),
  );
  await downloadGet;

  const status = page.locator('#download-status');
  await expect(status).not.toBeEmpty();
  await expect(status.locator('.h-full.bg-blue-500')).toBeVisible();
  await expect(status).toContainText('Baixando');
});
