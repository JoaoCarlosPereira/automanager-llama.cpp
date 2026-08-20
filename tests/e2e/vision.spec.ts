import { test, expect } from '@playwright/test';
import {
  DEFAULT_MODEL_PATH,
  loginAsAdmin,
  resetMockState,
  waitForModelsLoaded,
} from './helpers';

test.describe.configure({ mode: 'serial' });

test.beforeEach(async ({ request }) => {
  await resetMockState(request);
});

test('dashboard nao exibe seletor global de vision', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  await expect(page.locator('#mmproj-path')).toHaveCount(0);
});

test('cada modelo exibe botao de importar vision', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const items = page.locator('.model-item-container');
  const count = await items.count();
  expect(count).toBeGreaterThan(0);

  for (let i = 0; i < count; i += 1) {
    await expect(items.nth(i).locator('.vision-import-btn')).toBeVisible();
  }
});

test('modelo com mmproj exibe combobox no item', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const llamaItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'llama-3.1-8b.gguf' }),
  });
  await expect(llamaItem.locator('.model-mmproj-select')).toBeVisible();
  await expect(llamaItem.locator('.model-mmproj-select')).toContainText('llama-3.1-8b-mmproj.gguf');
});

test('modelo sem mmproj nao exibe combobox no item', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const mistralItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'mistral-7b.gguf' }),
  });
  await expect(mistralItem.locator('.vision-import-btn')).toBeVisible();
  await expect(mistralItem.locator('.model-mmproj-select')).toHaveCount(0);
});

test('modal de importacao envia model_path no download', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const mistralItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'mistral-7b.gguf' }),
  });
  await mistralItem.locator('.vision-import-btn').click();

  await expect(page.locator('#vision-import-modal')).toBeVisible();
  await expect(page.locator('#vision-import-model-path')).toHaveValue('/models/text/mistral-7b.gguf');

  await page.locator('#vision-import-url').fill('https://example.com/mistral-mmproj.gguf');

  const downloadResponse = page.waitForResponse(
    (resp) =>
      resp.url().includes('/downloads') &&
      resp.request().method() === 'POST' &&
      resp.ok(),
  );
  await page.locator('#vision-import-form').evaluate((form: HTMLFormElement) => form.requestSubmit());
  const response = await downloadResponse;
  const body = response.request().postDataJSON() as { url: string; model_path: string };
  expect(body.model_path).toBe('/models/text/mistral-7b.gguf');
  expect(body.url).toContain('mistral-mmproj.gguf');

  await expect(page.locator('#vision-import-modal')).toBeHidden();

  const modelsResponse = page.waitForResponse(
    (resp) =>
      resp.url().includes('/models') &&
      resp.request().method() === 'GET' &&
      resp.ok(),
  );
  await modelsResponse;
  await expect(mistralItem.locator('.model-mmproj-select')).toBeVisible();
  await expect(mistralItem.locator('.model-mmproj-select')).toContainText('mistral-mmproj.gguf');
});

test('Sem visão persiste após atualizar a página', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const llamaItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'llama-3.1-8b.gguf' }),
  });
  const select = llamaItem.locator('.model-mmproj-select');
  await expect(select).toBeVisible();

  const mmprojResponse = page.waitForResponse(
    (resp) =>
      resp.url().includes('/models/mmproj') &&
      resp.request().method() === 'POST' &&
      resp.ok(),
  );
  await select.selectOption('__no_vision__');
  const response = await mmprojResponse;
  const body = response.request().postDataJSON() as {
    model_path: string;
    mmproj_path: string;
  };
  expect(body.model_path).toBe(DEFAULT_MODEL_PATH);
  expect(body.mmproj_path).toBe('__no_vision__');
  await expect(select).toHaveValue('__no_vision__');

  await page.reload({ waitUntil: 'domcontentloaded' });
  await waitForModelsLoaded(page);
  const refreshedItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'llama-3.1-8b.gguf' }),
  });
  await expect(refreshedItem.locator('.model-mmproj-select')).toHaveValue('__no_vision__');
});

test('checkbox Vision local oculta o combo e persiste', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const llamaItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'llama-3.1-8b.gguf' }),
  });
  const visionCheckbox = llamaItem.locator('.model-vision-checkbox');
  await expect(visionCheckbox).toBeChecked();
  await visionCheckbox.uncheck();
  await expect(llamaItem.locator('.model-mmproj-control')).toBeHidden();

  await page.reload({ waitUntil: 'domcontentloaded' });
  await waitForModelsLoaded(page);
  const refreshedItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'llama-3.1-8b.gguf' }),
  });
  await expect(refreshedItem.locator('.model-vision-checkbox')).not.toBeChecked();
  await expect(refreshedItem.locator('.model-mmproj-control')).toBeHidden();
});

test('start envia mmproj_path selecionado no item do modelo', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const llamaItem = page.locator('.model-item-container').filter({
    has: page.locator('.model-name', { hasText: 'llama-3.1-8b.gguf' }),
  });
  await llamaItem.click();

  const startResponse = page.waitForResponse(
    (resp) => resp.url().includes('/start') && resp.request().method() === 'POST',
  );
  await llamaItem.locator('button:has-text("CARREGAR")').click();
  const response = await startResponse;
  const body = response.request().postDataJSON() as { path: string; mmproj_path: string };
  expect(body.path).toBe(DEFAULT_MODEL_PATH);
  expect(body.mmproj_path).toBe('/models/llama/llama-3.1-8b-mmproj.gguf');
});
