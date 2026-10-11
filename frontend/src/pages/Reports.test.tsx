import { render, screen, fireEvent, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import * as stream from '../api/jobStream'
import Reports from './Reports'

vi.mock('../api/hooks')
vi.mock('../api/jobStream')
vi.mock('../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

const REPORT = {
  id: 'r1', created_at: '2026-10-10T08:00:00Z', period: { start: '2026-10-02T11:00:00Z', end: '2026-10-09T11:00:00Z', days: 7 }, season: { id: 's11', name: '第 11 赛季' },
  model: 'qwen-plus', usage: { prompt_tokens: 1000, completion_tokens: 400 }, cost: 0.0016, unmatched: ['999.99'],
  public: '## 本周\n金矿**上涨**。', body: '- 金矿：中位价 216.67',
  facts: { skipped: [], movers: { up: [], down: [] }, items: [{ name: 'Gold Ore', name_zh: '金矿', median: 216.67, prev_median: 190, change_pct: 14, low: 100, high: 240, listings: 292, sold: 10, volatility_pct: 12.5, forecast: { beats_baseline: false, change_pct: 2, horizon_hours: 24 } }] },
}

function mount(url = '/reports', configured = true) {
  const create = vi.fn()
  vi.mocked(hooks.useLLM).mockReturnValue({ data: { configured }, isSuccess: true } as never)
  vi.mocked(hooks.useReports).mockReturnValue({ data: [{ ...REPORT, items: 1, unmatched: 1 }] } as never)
  vi.mocked(hooks.useReport).mockReturnValue({ data: REPORT } as never)
  vi.mocked(hooks.useReportEstimate).mockReturnValue({ data: { items: 1, skipped: 0, period: REPORT.period, prompt_tokens: 1200, completion_tokens: 1500, cost: 0.0039 }, isError: false } as never)
  vi.mocked(hooks.useCreateReport).mockReturnValue({ mutate: create, isPending: false } as never)
  for (const h of ['useEditReport', 'useDeleteReport'] as const) vi.mocked(hooks[h]).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, error: null, disconnected: false, resultRef: null })
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={[url]}><Reports /></MemoryRouter></QueryClientProvider>)
  return create
}

it('shows the estimate and writes a report for the chosen period', () => {
  const create = mount()
  expect(screen.getByText(/将包含 1 个道具.*预计输入约 1,200 个词元/)).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('时间范围'), { target: { value: '14' } })
  fireEvent.click(screen.getByRole('button', { name: '生成报告' }))
  expect(create.mock.calls[0][0]).toBe(14)
})

it('asks for a Bailian key first', () => {
  mount('/reports', false)
  expect(screen.getByText(/先在「设置 → AI 分析」填写/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '生成报告' })).toBeDisabled()
})

it('shows a report: preview, flagged numbers and the facts behind it', () => {
  mount('/reports?report=r1')
  expect(screen.getByRole('alert')).toHaveTextContent('999.99')
  expect(screen.getByText('上涨').tagName).toBe('B')                          // markdown preview, no raw HTML
  fireEvent.click(screen.getByRole('tab', { name: '订阅者正文' }))
  expect(screen.getByText('金矿：中位价 216.67')).toBeInTheDocument()
  expect(screen.getByText('接近随机波动')).toBeInTheDocument()
})

it('copies the shown part', async () => {
  const write = vi.fn(async () => {})
  Object.assign(navigator, { clipboard: { writeText: write } })
  mount('/reports?report=r1')
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: '复制' })) })
  expect(write).toHaveBeenCalledWith('## 本周\n金矿**上涨**。')
})
