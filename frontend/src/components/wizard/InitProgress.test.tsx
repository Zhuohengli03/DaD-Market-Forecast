import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import '../../i18n'
import { InitProgress } from './InitProgress'

it('shows each item with plain-language status and forecast outcome', () => {
  render(<MemoryRouter><InitProgress status="succeeded" progress={{ stage: 'done', done: 4, total: 4, detail: { items: [
    { item_id: 'a', name: 'Iron Ore', name_zh: '铁矿', status: 'done', pages: 3, collected: 120, forecast: 'not_enough_data', run_id: null, hours: 30 },
    { item_id: 'b', name: 'Gold Ore', name_zh: '金矿', status: 'done', pages: 9, collected: 900, forecast: 'ok', run_id: 'r1', hours: null },
    { item_id: 'c', name: 'Bandage', name_zh: '绷带', status: 'collect_failed', pages: 0, collected: 0, forecast: null, run_id: null, hours: null, error: 'auth' },
  ] } }} error={null} /></MemoryRouter>)
  expect(screen.getByText('铁矿')).toBeInTheDocument()
  expect(screen.getByText('采集到 120 条挂单')).toBeInTheDocument()
  expect(screen.getByText('数据还不够预测（已有 30 小时，需要 72 小时），自动采集几天后再试')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '查看预测' })).toHaveAttribute('href', '/forecast?run=r1')
  expect(screen.getByText('采集失败：API 密钥 被拒绝')).toBeInTheDocument()
  expect(screen.getAllByRole('progressbar').length).toBeGreaterThan(0)
})
