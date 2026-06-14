import { APIRequestContext, Page, expect } from '@playwright/test';

export const ADMIN_USER = 'admin';
export const ADMIN_PASS = 'admin';

export const DEFAULT_MODEL_PATH = '/models/llama/llama-3.1-8b.gguf';

export async function resetMockState(request: APIRequestContext): Promise<void> {
  await request.post('/__e2e/reset');
}

export async function loginAsAdmin(page: Page): Promise<void> {
  await page.goto('/', { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect(page.locator('#login-overlay')).toBeVisible();
  await page.locator('#login-username').fill(ADMIN_USER);
  await page.locator('#login-password').fill(ADMIN_PASS);
  const loginResponse = page.waitForResponse(
    (resp) => resp.url().includes('/login') && resp.ok(),
  );
  await page.locator('#login-form').evaluate((form: HTMLFormElement) => form.requestSubmit());
  await loginResponse;
  await expect(page.locator('#dashboard')).toBeVisible();
  await expect(page.locator('#login-overlay')).toBeHidden();
}

export async function waitForModelsLoaded(page: Page): Promise<void> {
  const firstModel = page.locator('.model-item-container').first();
  const alreadyVisible = await firstModel.isVisible().catch(() => false);
  if (!alreadyVisible) {
    await page.waitForResponse(
      (resp) =>
        resp.url().includes('/models') &&
        resp.request().method() === 'GET' &&
        resp.ok(),
      { timeout: 15000 },
    );
  }
  await expect(firstModel).toBeVisible({ timeout: 15000 });
}

export async function refreshDashboardStatus(page: Page): Promise<void> {
  await page.evaluate(() => {
    if (window.updateStatus) window.updateStatus();
    if (window.updateMetrics) window.updateMetrics();
  });
  await page.waitForResponse(
    (resp) => resp.url().includes('/status') && resp.request().method() === 'GET' && resp.ok(),
  );
}

export async function setMockRunning(
  request: APIRequestContext,
  running: boolean,
  path: string = DEFAULT_MODEL_PATH,
): Promise<void> {
  if (running) {
    await request.post('/start', {
      data: {
        path,
        gpu_weights: [
          { index: 0, weight: 100, active: true, is_main: true, pinned: false, name: 'GPU0', device: 'gpu' },
        ],
        context_size: 65536,
        parallel_slots: 1,
        batch_size: 512,
        split_mode: 'layer',
        auto_balance: false,
        manual_gpu_override: false,
      },
    });
  } else {
    await request.post('/stop');
  }
}
