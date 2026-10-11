import { fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import * as stream from '../../api/jobStream'
import { ItemForecast } from './ItemForecast'

vi.mock('../../api/hooks')
vi.mock('../../api/jobStream')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
vi.mock('../forecast/RunResult', () => ({ RunResult: ({ runId }: { runId: string }) => <p>result:{runId}</p> }))

const RUNS = [
  { run_id: 'r2', created_at: '2026-10-10T10:00:00Z', item: 'Gold Ore', season_id: 's10', season_name: 'Season 10', horizon_hours: 24, selected: 'ridge', skill_vs_naive: 0.1, beats_baseline: true, source: 'sqlite', warnings: 0 },
  { run_id: 'r1', created_at: '2026-10-09T10:00:00Z', item: 'Gold Ore', season_id: 's10', season_name: 'Season 10', horizon_hours: 72, selected: 'naive', skill_vs_naive: 0, beats_baseline: false, source: 'sqlite', warnings: 0 },
  { run_id: 'x', created_at: '2026-10-10T11:00:00Z', item: 'Cobalt Ore', season_id: 's10', season_name: 'Season 10', horizon_hours: 24, selected: 'ridge', skill_vs_naive: 0.1, beats_baseline: true, source: 'sqlite', warnings: 0 },
]

function mount(mutate = vi.fn()) {
  vi.mocked(hooks.useRuns).mockReturnValue({ data: RUNS } as never)
  vi.mocked(hooks.useModels).mockReturnValue({ data: ['naive', 'seasonal_naive_24', 'ridge', 'hgb', 'rf', 'ets', 'arima'] } as never)
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate, isPending: false } as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ status: null, progress: null, error: null, disconnected: false, resultRef: null } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><ItemForecast item="Gold Ore" season="s10" /></MemoryRouter></QueryClientProvider>)
  return mutate
}

it('one click forecasts with the default settings, and shows this item\'s newest forecast in full', () => {
  const mutate = mount()
  expect(screen.getByText('result:r2')).toBeInTheDocument()
  expect((screen.getByLabelText('预测记录') as HTMLSelectElement).options).toHaveLength(2)    // this item's runs only
  fireEvent.click(screen.getByRole('button', { name: '一键预测' }))
  expect(mutate.mock.calls[0][0]).toMatchObject({ item: 'Gold Ore', season_id: 's10', horizon_hours: 24, coverage: 0.9, resolution: '1h' })
})

it('the settings change what the next forecast does', () => {
  const mutate = mount()
  fireEvent.change(screen.getByLabelText('预测多久'), { target: { value: '72' } })
  fireEvent.click(screen.getByLabelText('随机森林'))
  fireEvent.click(screen.getByRole('button', { name: '一键预测' }))
  expect(mutate.mock.calls[0][0].horizon_hours).toBe(72)
  expect(mutate.mock.calls[0][0].models).not.toContain('rf')
})
