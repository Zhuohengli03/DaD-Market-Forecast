import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import * as stream from '../api/jobStream'
import Collect from './Collect'

vi.mock('../api/hooks')
vi.mock('../api/jobStream')
vi.mock('../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
vi.mock('../components/JobList', () => ({ JobList: () => null }))

it('opens on items without data and collects just those', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useItems).mockReturnValue({ isSuccess: true, data: [
    { name: 'Gold Ore', name_zh: '金矿', rows: 50, season_rows: 50, last_ts: '2026-10-09T10:00:00Z' },
    { name: 'Iron Ore', name_zh: '铁矿', rows: 0, season_rows: 0, last_ts: null },
    { name: 'Cobalt Ore', name_zh: '钴矿', rows: 0, season_rows: 0, last_ts: null }] } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ data: { api_key_configured: true } } as never)
  vi.mocked(hooks.useQuota).mockReturnValue({ data: { credits_remaining: 5000 } } as never)
  vi.mocked(hooks.useCollectAllEstimate).mockReturnValue({ data: { requests: 404, items: 3, unknown: 2, per_item: { 'Gold Ore': 2, 'Iron Ore': 201, 'Cobalt Ore': 201 } } } as never)
  vi.mocked(hooks.useCollectAll).mockReturnValue({ mutate, isPending: false } as never)
  for (const h of ['useStartCollect', 'useUntrackItem'] as const) vi.mocked(hooks[h]).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, error: null, disconnected: false, resultRef: null })
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  render(<MemoryRouter initialEntries={['/collect?filter=nodata']}><Collect /></MemoryRouter>)
  expect(screen.getByRole('tab', { name: '没有数据（2）' })).toHaveAttribute('aria-selected', 'true')
  expect(screen.queryByText('金矿')).not.toBeInTheDocument()
  expect(screen.getByText('预计最多 402 次请求（剩余 5,000 次）')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '采集这些（2 个）' }))
  expect(mutate.mock.calls[0][0]).toEqual(['Iron Ore', 'Cobalt Ore'])
})
