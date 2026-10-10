import { render, screen, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { Shell } from './Shell'

vi.mock('../../api/hooks')

function mount(status: object, quota: object | undefined = undefined) {
  vi.mocked(hooks.useStatus).mockReturnValue({ data: status } as never)
  vi.mocked(hooks.useQuota).mockReturnValue({ data: quota } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Shell /></MemoryRouter></QueryClientProvider>)
}

it('orders navigation by workflow: get data first, then analyse, settings last', () => {
  mount({ source: 'csv', jobs_running: [] })
  const nav = screen.getByRole('navigation')
  const links = within(nav).getAllByRole('link').map(a => a.textContent)
  expect(links).toEqual(['概览', '道具库', '数据采集', '定时采集', '价格走势', '价格预测', '行情报告', '装备行情', '设置'])
  expect(within(nav).getByText('① 获取数据')).toBeInTheDocument()
  expect(within(nav).getByText('② 查看与分析')).toBeInTheDocument()
  expect(screen.getByText('还没有用过 DarkerDB，用一次后显示')).toBeInTheDocument()
})

it('shows DarkerDB credits under Settings', () => {
  mount({ source: 'sqlite', jobs_running: [] }, { credits_remaining: 7232, monthly_quota: 10000, rate_limit: 60, updated_at: new Date().toISOString() })
  const widget = screen.getByLabelText('DarkerDB 额度')
  expect(within(widget).getByText('7,232')).toBeInTheDocument()
  expect(within(widget).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '72')
})

it('groups running work by area and counts every collected item', () => {
  mount({ source: 'sqlite', collecting: { 'Gold Ore': 'collecting', 'Iron Ore': 'collecting', 'Copper Ore': 'waiting' },
    jobs_running: [{ id: '1', kind: 'collect', meta: { item: '*' } }, { id: '2', kind: 'forecast', meta: {} }] })
  const list = screen.getByLabelText('正在运行的任务')
  expect(within(list).getByText('数据采集')).toBeInTheDocument()
  expect(within(list).getByText('2 采集中 · 1 待采集')).toBeInTheDocument()
  expect(within(list).getByText('价格预测')).toBeInTheDocument()
})
