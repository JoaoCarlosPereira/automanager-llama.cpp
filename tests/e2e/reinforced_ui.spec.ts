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

test('selecionar modelo abre aba', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const firstModel = page.locator('.model-item-container').first();
  const modelName = await firstModel.locator('.model-name').innerText();
  
  await firstModel.click();
  
  // Verifica se o botão da aba apareceu
  const tabBtn = page.locator('.tab-btn', { hasText: modelName });
  await expect(tabBtn).toBeVisible();
  
  // Verifica se o conteúdo da tab apareceu
  await expect(page.locator('.tab-content.active')).toBeVisible();
  await expect(page.locator('.model-tab-name')).toHaveText(modelName);
});

test('selecionar dois modelos abre duas abas', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const models = page.locator('.model-item-container');
  const firstName = await models.nth(0).locator('.model-name').innerText();
  const secondName = await models.nth(1).locator('.model-name').innerText();

  await models.nth(0).click();
  await models.nth(1).click();

  await expect(page.locator('.tab-btn')).toHaveCount(2);
  await expect(page.locator('.tab-btn', { hasText: firstName })).toBeVisible();
  await expect(page.locator('.tab-btn', { hasText: secondName })).toBeVisible();
});

test('ctrl+clique abre nova aba do mesmo modelo', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  const firstModel = page.locator('.model-item-container').first();
  const modelName = await firstModel.locator('.model-name').innerText();

  await firstModel.click();
  await firstModel.click({ modifiers: ['Control'] });

  await expect(page.locator('.tab-btn')).toHaveCount(2);
  await expect(page.locator('.tab-btn', { hasText: `${modelName} (1)` })).toBeVisible();
  await expect(page.locator('.tab-btn', { hasText: `${modelName} (2)` })).toBeVisible();
});

test('start modelo na aba mostra status ONLINE', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  await page.locator('.model-item-container').first().click();
  
  const tab = page.locator('.tab-content.active');
  const startBtn = tab.locator('button:has-text("Iniciar Instância")');
  
  const startResponse = page.waitForResponse(
    (resp) => resp.url().includes('/start') && resp.request().method() === 'POST',
  );
  await startBtn.click();
  await startResponse;
  
  await refreshDashboardStatus(page);

  // Verifica status na aba
  const statusBadge = tab.locator('.tab-status-badge');
  await expect(statusBadge).toContainText('ONLINE', { timeout: 15000 });
  
  // Verifica ponto de status na barra de abas
  const tabBtn = page.locator('.tab-btn.active');
  await expect(tabBtn.locator('.tab-status-dot')).toHaveClass(/bg-emerald-500/);
});

test('smart calibration gera proposta e permite efetivar', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  await page.locator('.model-item-container').first().click();
  const tab = page.locator('.tab-content.active');
  
  // Clicar em calibrar
  await tab.locator('button:has-text("CALIBRAR SMART")').click();
  
  // Aguardar proposta (mock server leva 1s)
  const proposalArea = tab.locator('.tab-proposed-config');
  await expect(proposalArea).toBeVisible({ timeout: 10000 });
  
  await expect(proposalArea).toContainText('Configuração Otimizada Sugerida');
  
  // Efetivar
  const applyBtn = proposalArea.locator('button:has-text("EFETIVAR E SALVAR")');
  const startResponse = page.waitForResponse(
    (resp) => resp.url().includes('/start') && resp.request().method() === 'POST',
  );
  await applyBtn.click();
  await startResponse;
  
  await expect(proposalArea).toBeHidden();
  await refreshDashboardStatus(page);
  await expect(tab.locator('.tab-status-badge')).toContainText('ONLINE');
});

test('fixar campos (PIN) é respeitado visualmente', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  await page.locator('.model-item-container').first().click();
  const tab = page.locator('.tab-content.active');
  
  const ctxPin = tab.locator('.tab-pin-context');
  const ctxPinIcon = ctxPin.locator('..').locator('i'); // thumbtack icon
  
  // Antes: cinza
  await expect(ctxPinIcon).toHaveClass(/text-slate-700/);
  
  // Clicar no label do pin (que contém o checkbox hidden e o icon)
  await ctxPin.locator('..').click();
  
  // Depois: azul
  await expect(ctxPinIcon).toHaveClass(/text-blue-500/);
});

test('fechar aba limpa estado', async ({ page }) => {
  await loginAsAdmin(page);
  await waitForModelsLoaded(page);

  await page.locator('.model-item-container').first().click();
  const modelName = await page.locator('.model-item-container').first().locator('.model-name').innerText();
  
  const tabBtn = page.locator('.tab-btn', { hasText: modelName });
  await expect(tabBtn).toBeVisible();
  
  // Clicar no X
  await tabBtn.locator('.tab-close-btn').click();
  
  await expect(tabBtn).toBeHidden();
  await expect(page.locator('#no-tab-content')).toBeVisible();
});

test('sidebar retrátil funciona', async ({ page }) => {
  await loginAsAdmin(page);
  
  const sidebar = page.locator('#sidebar');
  const toggleBtn = page.locator('#sidebar-toggle');
  
  // Inicialmente aberta (desktop)
  await expect(sidebar).not.toHaveClass(/collapsed/);
  
  // Clicar para fechar
  await toggleBtn.click();
  await expect(sidebar).toHaveClass(/collapsed/);
  
  // Clicar para abrir
  await toggleBtn.click();
  await expect(sidebar).not.toHaveClass(/collapsed/);
});
