import { test, expect } from '@playwright/test';
import {
  DEFAULT_MODEL_PATH,
  loginAsAdmin,
  refreshDashboardStatus,
  resetMockState,
  setMockRunning,
  waitForModelsLoaded,
} from './helpers';

test.describe.configure({ mode: 'serial' });

test.beforeEach(async ({ request }) => {
  await resetMockState(request);
});

test('start modelo mostra ONLINE e active card', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const firstModel = page.locator('.model-item-container').first();
  await firstModel.click();
  const startResponse = page.waitForResponse(
    (resp) => resp.url().includes('/start') && resp.request().method() === 'POST',
  );
  await firstModel.locator('button:has-text("CARREGAR")').click();
  await startResponse;
  await refreshDashboardStatus(page);

  await expect(page.locator('#status-badge')).toContainText('ONLINE', { timeout: 15000 });
  await expect(page.locator('#active-card')).toBeVisible();
});

test('stop modelo mostra OFFLINE e esconde active card', async ({ page, request }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);
  await setMockRunning(request, true, DEFAULT_MODEL_PATH);
  await refreshDashboardStatus(page);

  await expect(page.locator('#status-badge')).toContainText('ONLINE', { timeout: 15000 });
  await expect(page.locator('#active-card')).toBeVisible();

  await page.evaluate(() => {
    window.confirm = () => true;
  });
  const stopResponse = page.waitForResponse(
    (resp) => resp.url().includes('/stop') && resp.request().method() === 'POST',
    { timeout: 15_000 },
  );
  await page.locator('#active-card button:has-text("ENCERRAR")').click();
  await stopResponse;
  await refreshDashboardStatus(page);

  await expect(page.locator('#status-badge')).toContainText('OFFLINE', { timeout: 15000 });
  await expect(page.locator('#active-card')).toBeHidden();
});

test('rename modelo atualiza nome na lista', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const newName = 'renamed-model';
  page.once('dialog', (dialog) => {
    expect(dialog.type()).toBe('prompt');
    dialog.accept(newName);
  });

  const renameResponse = page.waitForResponse(
    (resp) => resp.url().includes('/rename') && resp.request().method() === 'POST',
  );
  await page.locator('.model-item-container').first().locator('.rename-btn').click();
  await renameResponse;

  const modelsResponse = page.waitForResponse(
    (resp) => resp.url().includes('/models') && resp.request().method() === 'GET' && resp.ok(),
  );
  await modelsResponse;
  await expect(page.locator('.model-name').first()).toContainText(`${newName}.gguf`);
});

test('delete modelo remove da lista', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const initialCount = await page.locator('.model-item-container').count();
  expect(initialCount).toBeGreaterThan(1);

  page.once('dialog', (dialog) => dialog.accept());

  const deleteResponse = page.waitForResponse(
    (resp) => resp.url().includes('/delete') && resp.request().method() === 'POST',
  );
  await page.locator('.model-item-container').last().locator('.delete-btn').click();
  await deleteResponse;

  await expect(page.locator('.model-item-container')).toHaveCount(initialCount - 1);
});
