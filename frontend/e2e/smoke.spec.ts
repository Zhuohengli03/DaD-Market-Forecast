import { test, expect } from '@playwright/test'

test('run a forecast and see a verdict', async ({ page }) => {
  await page.goto('/forecast')
  await page.getByLabel('道具').selectOption('Test Ore')
  await page.getByLabel('赛季').selectOption({ label: 'Season 2' })
  await page.getByLabel('预测多久').selectOption('24')
  for (const m of ['梯度提升树', '随机森林', '指数平滑', 'ARIMA 时间序列']) { const cb = page.getByLabel(m); if (await cb.isChecked()) await cb.click() }
  await page.getByRole('button', { name: '开始预测' }).click()
  await expect(page.getByText(/^(模型比基准更准|没有找到比基准明显更准的方法)/)).toBeVisible({ timeout: 120_000 })
  await expect(page.getByText('回测成绩（越低越好）')).toBeVisible()
  await expect(page.getByText(/最新数据已经是 \d+ 天前的了/)).toBeVisible()
})

test('first run asks for an API key and can be postponed', async ({ page }) => {
  await page.route('**/api/setup', route => route.fulfill({ json: { ok: true, error: null, meta: null,
    data: { api_key_configured: false, api_key_hint: null, data_source: 'csv', seasons: 1, tracked_items: 1, catalog_items: 2 } } }))
  await page.goto('/')
  const dialog = page.getByRole('dialog')
  await expect(dialog).toContainText('连接你的 DarkerDB 账号')
  await expect(dialog.getByRole('link', { name: /darkerdb\.com/ })).toHaveAttribute('href', 'https://darkerdb.com/dashboard/api-keys')
  await dialog.getByRole('button', { name: '稍后再说' }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('catalog shows parameters and tracks an item', async ({ page }) => {
  await page.goto('/catalog')
  await expect(page.getByText('Arming Sword')).toBeVisible()
  await page.getByRole('button', { name: 'Arming Sword' }).click()
  await expect(page.getByText('"weapon_damage": 30')).toBeVisible()
  await page.getByRole('button', { name: '追踪' }).click()
  await expect(page.getByText('已开始追踪 Arming Sword')).toBeVisible()
  await page.goto('/collect')
  await expect(page.getByText('Arming Sword')).toBeVisible()
})
