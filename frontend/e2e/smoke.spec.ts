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

test('first run opens the setup wizard in order and can be postponed', async ({ page }) => {
  await page.route('**/api/setup', route => route.fulfill({ json: { ok: true, error: null, meta: null, data: {
    initialized: false, collect_interval_hours: 6, auto_collect: false, api_key_configured: false, api_key_hint: null,
    data_source: 'csv', seasons: 1, tracked_items: 1, catalog_items: 5,
    environment: { python_version: '3.13.5', python_ok: true, storage: 'csv', data_dir: '/tmp/data', db_host: null, db_database: null } } } }))
  await page.goto('/')
  const dialog = page.getByRole('dialog')
  await expect(dialog).toContainText('欢迎使用 Darker Market')
  await expect(dialog).toContainText('1 环境')
  await expect(dialog.getByLabel('本机文件（推荐）')).toBeChecked()
  await dialog.getByRole('button', { name: '下一步' }).click()
  await expect(dialog.getByRole('link', { name: /darkerdb\.com/ })).toHaveAttribute('href', 'https://darkerdb.com/dashboard/api-keys')
  await expect(dialog.getByRole('button', { name: '下一步' })).toBeDisabled()        // key required before initialization
  await dialog.getByRole('button', { name: '上一步' }).click()
  await dialog.getByRole('button', { name: '稍后再说' }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('catalog lists tradable types, tracks stat-free items and routes equipment to the market page', async ({ page }) => {
  await page.goto('/catalog')
  await expect(page.getByText('只显示能在交易行买卖的道具。')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Gold Coin' })).toHaveCount(0)       // never listed, hidden
  await expect(page.getByRole('link', { name: '查看行情' })).toHaveAttribute('href', '/equipment?type=id.item.arming_sword')
  await page.getByRole('button', { name: 'Bandage' }).click()
  await expect(page.getByText('duration')).toBeVisible()
  await page.getByRole('button', { name: '追踪' }).click()
  await expect(page.getByText('已开始追踪 Bandage')).toBeVisible()
  await page.goto('/collect')
  await expect(page.getByText('Bandage')).toBeVisible()
})
