import { test, expect, type Page } from '@playwright/test';
import { resetMockState } from './helpers';

test.describe.configure({ mode: 'serial' });

test.beforeEach(async ({ context, request }) => {
  await resetMockState(request);
  await context.clearCookies();
});

async function openDashboard(page: Page) {
  await page.goto('/', { waitUntil: 'domcontentloaded', timeout: 60_000 });
}

test('overlay de login visivel ao acessar a pagina', async ({ page }) => {
  await openDashboard(page);

  await expect(page.locator('#login-overlay')).toBeVisible();
  await expect(page.locator('#login-username')).toBeVisible();
  await expect(page.locator('#login-password')).toBeVisible();
  await expect(page.locator('#dashboard')).toBeHidden();
});

test('login com credenciais validas mostra dashboard', async ({ page }) => {
  await openDashboard(page);

  await expect(page.locator('#login-overlay')).toBeVisible();

  await page.locator('#login-username').fill('admin');
  await page.locator('#login-password').fill('qualquer-senha');

  await Promise.all([
    page.waitForResponse(
      (res) =>
        res.url().includes('/api/auth/login') && res.request().method() === 'POST',
    ),
    page.locator('#login-form').evaluate((form) =>
      (form as HTMLFormElement).requestSubmit(),
    ),
  ]);

  await expect(page.locator('#dashboard')).toBeVisible();
  await expect(page.locator('#login-overlay')).toBeHidden();
  await expect(page.locator('#status-badge')).toBeVisible();
  await expect(page.locator('#cpu-val')).toBeVisible();
  await expect(page.locator('#model-list-container')).toBeVisible();
  await expect(page.locator('#dashboard')).toHaveCSS('display', 'flex');
});

test('login mostra modelos frequentes sem recarregar pagina', async ({ page }) => {
  await openDashboard(page);

  await page.locator('#login-username').fill('admin');
  await page.locator('#login-password').fill('qualquer-senha');
  await Promise.all([
    page.waitForResponse(
      (res) =>
        res.url().includes('/api/auth/login') &&
        res.request().method() === 'POST' &&
        res.ok(),
    ),
    page.locator('#login-form').evaluate((form) =>
      (form as HTMLFormElement).requestSubmit(),
    ),
  ]);

  await page.waitForResponse(
    (res) =>
      res.url().includes('/models') &&
      res.request().method() === 'GET' &&
      res.ok(),
  );

  await expect(page.locator('#no-tab-shortcuts')).toBeVisible({ timeout: 15000 });
  await expect(page.locator('#no-tab-shortcuts-grid button')).toHaveCount(1);
});

test('login com credenciais invalidas mostra mensagem de erro', async ({ page }) => {
  await openDashboard(page);

  await page.locator('#login-username').fill('invalid');
  await page.locator('#login-password').fill('wrong');

  const [loginResponse] = await Promise.all([
    page.waitForResponse(
      (res) =>
        res.url().includes('/api/auth/login') && res.request().method() === 'POST',
      { timeout: 15_000 },
    ),
    page.locator('#login-form').evaluate((form) =>
      (form as HTMLFormElement).requestSubmit(),
    ),
  ]);
  expect(loginResponse.status()).toBe(401);

  await expect(page.locator('#login-error')).toBeVisible();
  await expect(page.locator('#login-error')).toContainText('Credenciais invalidas');
  await expect(page.locator('#login-overlay')).toBeVisible();
  await expect(page.locator('#dashboard')).toBeHidden();
});

test('apos logout overlay de login reaparece', async ({ page }) => {
  await openDashboard(page);

  await page.locator('#login-username').fill('admin');
  await page.locator('#login-password').fill('qualquer-senha');
  await Promise.all([
    page.waitForResponse((res) => res.url().includes('/api/auth/login') && res.ok()),
    page.locator('#login-form').evaluate((form) =>
      (form as HTMLFormElement).requestSubmit(),
    ),
  ]);
  await expect(page.locator('#dashboard')).toBeVisible();

  await Promise.all([
    page.waitForURL('/'),
    page.getByTitle('Sair').click(),
  ]);

  await expect(page.locator('#login-overlay')).toBeVisible({ timeout: 15000 });
  await expect(page.locator('#dashboard')).toBeHidden();
});
