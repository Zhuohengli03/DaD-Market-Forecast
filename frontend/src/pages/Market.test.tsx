import { act, cleanup, render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import * as stream from '../api/jobStream'
import Market from './Market'

vi.mock('../api/hooks')
vi.mock('../api/jobStream')
vi.mock('echarts-for-react', () => ({ default: () => null }))
vi.mock('../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
vi.mock('./market/ListingsTable', () => ({ ListingsTable: () => null }))

function Where() { const l = useLocation(); return <p data-testid="url">{l.pathname + l.search}</p> }

function mount(qc = new QueryClient(), marketOpen?: string) {
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', name_zh: '金矿', rows: 30, item_id: 'id.item.gold_ores_5001' }] } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's11', name: '第 11 赛季', start: '2026-10-09T00:00:00Z', end: null, market_open: marketOpen }] } as never)
  vi.mocked(hooks.useSeries).mockReturnValue({ data: { points: [{ ts: '2026-10-09T10:00:00+00:00', median_price: 10, p25: 9, p75: 11, min_price: 8, max_price: 12, listings: 3, sold: 1, volume: 4 }] }, isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useJobs).mockReturnValue({ data: [] } as never)
  vi.mocked(hooks.useNextHour).mockReturnValue({ data: undefined, isLoading: false, isError: false } as never)
  vi.mocked(hooks.useCollectEstimate).mockReturnValue({ data: { requests: 4, known: true, since: '2026-10-09T00:00:00+00:00', until: null } } as never)
  render(<QueryClientProvider client={qc}><MemoryRouter initialEntries={['/market']}><Routes><Route path="*" element={<><Market /><Where /></>} /></Routes></MemoryRouter></QueryClientProvider>)
}

beforeEach(() => {
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, error: null, disconnected: false, resultRef: null })
  vi.mocked(hooks.useStartCollect).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
})

it('says from when this season counts, and that times are UTC', () => {
  mount(new QueryClient(), '2026-10-23')
  expect(screen.getByText(/交易行 2026-10-23 开放/)).toHaveTextContent('只用这天起的记录')
  expect(screen.getByText(/交易行 2026-10-23 开放/)).toHaveTextContent('图表时间为 UTC')
})

it('starts a default forecast for the shown item and season, then opens it on the item page', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate, isPending: false } as never)
  mount()
  fireEvent.click(screen.getByRole('button', { name: '一键预测' }))
  expect(mutate.mock.calls[0][0]).toEqual({ item: 'Gold Ore', season_id: 's11', horizon_hours: 24, models: ['ridge', 'hgb', 'rf', 'ets', 'arima'], coverage: 0.9 })
  act(() => mutate.mock.calls[0][1].onSuccess('job42'))
  expect(screen.getByTestId('url').textContent).toBe('/item/id.item.gold_ores?q=Gold+Ore&season=s11&job=job42')
})

it('gets the latest data for the shown item and redraws when it is done', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useStartCollect).mockReturnValue({ mutate, isPending: false } as never)
  const qc = new QueryClient(); const spy = vi.spyOn(qc, 'invalidateQueries')
  mount(qc)
  expect(screen.getByText('采集 2026-10-09 至 现在，预计最多消耗 4 次 DarkerDB 额度')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '获取第 11 赛季数据' }))
  expect(mutate.mock.calls[0][0]).toEqual({ item: 'Gold Ore', seasonId: 's11' })   // the season picked above
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: { stage: 'done', done: null, total: null, detail: { new: 12 } }, status: 'succeeded', error: null, disconnected: false, resultRef: null })
  act(() => mutate.mock.calls[0][1].onSuccess('job7'))
  expect(screen.getByText(/新增 12 条/)).toBeInTheDocument()
  expect(spy).toHaveBeenCalledWith({ queryKey: ['series'] })
})

it('when nothing new comes back, says how old DarkerDB\'s newest listing is', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useStartCollect).mockReturnValue({ mutate, isPending: false } as never)
  mount()
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', name_zh: '金矿', rows: 30, last_ts: '2026-10-09T11:31:30Z' }] } as never)
  fireEvent.click(screen.getByRole('button', { name: '获取第 11 赛季数据' }))
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: { stage: 'done', done: null, total: null, detail: { new: 0 } }, status: 'succeeded', error: null, disconnected: false, resultRef: null })
  act(() => mutate.mock.calls[0][1].onSuccess('job8'))
  expect(screen.getByText(/DarkerDB 上还没有更新的挂单，它最新一条是 2026-10-09 11:31Z/)).toBeInTheDocument()
})


it('the button names the season it collects, the newest one too', () => {
  mount()
  expect(screen.getByRole('button', { name: '获取第 11 赛季数据' })).toBeInTheDocument()
  cleanup()
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's10', name: '第 10 赛季', start: '2026-07-17T00:00:00+00:00', end: '2026-10-09T03:00:00+00:00' },
    { id: 's11', name: '第 11 赛季', start: '2026-10-09T03:00:00Z', end: null }] } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={['/market?season=s10']}><Market /></MemoryRouter></QueryClientProvider>)
  expect(screen.getByRole('button', { name: '获取第 10 赛季数据' })).toBeInTheDocument()
})

it('an earlier season with nothing new says its window is already complete', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useStartCollect).mockReturnValue({ mutate, isPending: false } as never)
  mount()
  cleanup()
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's10', name: '第 10 赛季', start: '2026-07-17T00:00:00+00:00', end: '2026-10-09T03:00:00+00:00' },
    { id: 's11', name: '第 11 赛季', start: '2026-10-09T03:00:00Z', end: null }] } as never)
  vi.mocked(hooks.useCollectEstimate).mockReturnValue({ data: { requests: 3, known: true, since: '2026-07-30T00:00:00+00:00', until: '2026-10-09T03:00:00+00:00', opens: null } } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter initialEntries={['/market?season=s10']}><Market /></MemoryRouter></QueryClientProvider>)
  expect(screen.getByText('采集 2026-07-30 至 2026-10-09，预计最多消耗 3 次 DarkerDB 额度')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '获取第 10 赛季数据' }))
  expect(mutate.mock.calls[0][0]).toEqual({ item: 'Gold Ore', seasonId: 's10' })
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: { stage: 'done', done: null, total: null, detail: { new: 0 } }, status: 'succeeded', error: null, disconnected: false, resultRef: null })
  act(() => mutate.mock.calls[0][1].onSuccess('job9'))
  expect(screen.getByText('第 10 赛季（2026-07-30 至 2026-10-09）没有更多挂单，本机这段时间的数据已经完整')).toBeInTheDocument()
})
