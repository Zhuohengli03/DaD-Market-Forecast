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

const SEASONS = [{ id: 's10', name: '第 10 赛季', start: '2026-07-17T00:00:00+00:00', end: '2026-10-09T03:00:00+00:00' }, { id: 's11', name: '第 11 赛季', start: '2026-10-09T03:00:00+00:00', end: null, market_open: '2026-10-23' }]

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
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: SEASONS } as never)
  for (const h of ['useStartCollect', 'useUntrackItem'] as const) vi.mocked(hooks[h]).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, error: null, disconnected: false, resultRef: null })
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  render(<MemoryRouter initialEntries={['/collect?filter=nodata']}><Collect /></MemoryRouter>)
  expect(screen.getByRole('tab', { name: '没有数据（2）' })).toHaveAttribute('aria-selected', 'true')
  expect(screen.queryByText('金矿')).not.toBeInTheDocument()
  expect(screen.getByText('预计最多消耗 402 次 DarkerDB 额度（剩余 5,000 次）')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '采集这些（2 个）' }))
  expect(mutate.mock.calls[0][0]).toEqual({ names: ['Iron Ore', 'Cobalt Ore'], seasonId: 's11' })   // current season by default
})


it('collects the chosen season and says when a market has not opened yet', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useItems).mockReturnValue({ isSuccess: true, data: [{ name: 'Gold Ore', name_zh: '金矿', rows: 50, season_rows: 0, last_ts: null }] } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ data: { api_key_configured: true } } as never)
  vi.mocked(hooks.useQuota).mockReturnValue({ data: { credits_remaining: 5000 } } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: SEASONS } as never)
  vi.mocked(hooks.useCollectAll).mockReturnValue({ mutate, isPending: false } as never)
  for (const h of ['useStartCollect', 'useUntrackItem'] as const) vi.mocked(hooks[h]).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, error: null, disconnected: false, resultRef: null })
  vi.mocked(hooks.useCollectAllEstimate).mockImplementation(((_on: boolean, sid?: string) => ({ data: sid === 's11'
    ? { requests: 0, items: 1, unknown: 0, per_item: { 'Gold Ore': 0 }, since: '2026-10-23T00:00:00+00:00', until: null, opens: '2026-10-23' }
    : { requests: 40, items: 1, unknown: 0, per_item: { 'Gold Ore': 40 }, since: '2026-07-30T00:00:00+00:00', until: '2026-10-09T03:00:00+00:00', opens: null } })) as never)
  render(<MemoryRouter initialEntries={['/collect']}><Collect /></MemoryRouter>)
  expect(screen.getByText(/交易行 2026-10-23 才开放/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /全部采集/ })).toBeDisabled()
  fireEvent.change(screen.getByLabelText('赛季'), { target: { value: 's10' } })
  expect(screen.getByText(/采集 2026-07-30 至 2026-10-09/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /全部采集/ }))
  expect(mutate.mock.calls[0][0]).toEqual({ names: undefined, seasonId: 's10' })
})
