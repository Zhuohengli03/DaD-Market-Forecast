import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { SeasonHoldout } from './SeasonHoldout'

vi.mock('../../api/hooks')
vi.mock('../market/RefreshData', () => ({ RefreshData: ({ seasonName }: { seasonName: string }) => <span>fetch {seasonName}</span> }))

const SEASONS = [
  { id: 's10', name: 'Season 10', start: '2026-07-17T00:00:00Z', end: '2026-10-09T03:00:00Z', note: '' },
  { id: 's11', name: 'Season 11', start: '2026-10-09T03:00:00Z', end: null, note: '' },
]

it('offers only ended seasons, shows how much of it is collected, and starts a 70% holdout', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: SEASONS, isSuccess: true } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', rows: 10 }, { name: 'Copper Ore', rows: 0 }] } as never)
  vi.mocked(hooks.useCollectEstimate).mockReturnValue({ data: { requests: 40, known: true, since: '2026-09-01T00:00:00Z', until: '2026-10-09T03:00:00Z', opens: null, progress: { through: '2026-09-01T00:00:00Z', complete: false, share: 0.45 } } } as never)
  vi.mocked(hooks.useStartHoldout).mockReturnValue({ mutate, isPending: false } as never)
  vi.mocked(hooks.useHoldouts).mockReturnValue({ data: [] } as never)
  vi.mocked(hooks.useHoldout).mockReturnValue({ data: undefined } as never)
  vi.mocked(hooks.useDeleteHoldout).mockReturnValue({ mutate: vi.fn() } as never)
  vi.mocked(hooks.useJob).mockReturnValue({ data: undefined } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><SeasonHoldout /></MemoryRouter></QueryClientProvider>)
  const season = screen.getByLabelText('已结束的赛季') as HTMLSelectElement
  expect([...season.options].map(o => o.value)).toEqual(['s10'])
  expect(screen.getByText(/本机已采集到 2026-09-01（这个赛季的 45%）/)).toBeInTheDocument()
  expect(screen.getByText('fetch 第 10 赛季')).toBeInTheDocument()
  // every tracked item can be picked, even one without data yet: its season can be fetched right here
  expect((screen.getByLabelText('道具') as HTMLSelectElement).options).toHaveLength(2)
  fireEvent.change(screen.getByLabelText('预测粒度'), { target: { value: '4h' } })
  fireEvent.click(screen.getByRole('button', { name: '开始回测' }))
  expect(mutate.mock.calls[0][0]).toEqual({ item: 'Gold Ore', season_id: 's10', train_frac: 0.7, resolution: '4h' })
})
