import { test, expect } from '@playwright/test'

test('run a forecast and see a verdict', async ({ page }) => {
  await page.goto('/forecast')
  await page.getByLabel('物品').selectOption('Gold Ore')
  const seasons = page.getByLabel('赛季')
  await expect(seasons.locator('option')).not.toHaveCount(0)
  await seasons.selectOption({ label: 'Season 2' })
  await page.getByLabel('预测步长').selectOption('24')
  for (const m of ['hgb', 'rf', 'ets', 'arima']) { const cb = page.getByLabel(m); if (await cb.isChecked()) await cb.click() }
  await page.getByText('开始预测').click()
  await expect(page.locator('pre')).toContainText('backtest', { timeout: 120_000 })
  await expect(page.getByText(/优于基线/)).toBeVisible({ timeout: 120_000 })
  await expect(page.getByText('回测', { exact: true })).toBeVisible()
})
