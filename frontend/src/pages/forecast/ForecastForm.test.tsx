import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { ForecastForm } from './ForecastForm'

vi.mock('../../api/hooks')

it('submits selected item, season, horizon and models', async () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Iron Ore', rows: 0 }, { name: 'Gold Ore', rows: 30 }] } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's2', name: 'S2', start: '', end: null, note: '' }] } as never)
  vi.mocked(hooks.useModels).mockReturnValue({ data: ['naive', 'seasonal_naive_24', 'ridge', 'hgb'] } as never)
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate, isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><ForecastForm onStarted={() => {}} /></MemoryRouter></QueryClientProvider>)
  fireEvent.click(screen.getByLabelText('梯度提升树'))
  fireEvent.click(screen.getByText('开始预测'))
  expect(mutate).toHaveBeenCalledWith({ item: 'Gold Ore', season_id: 's2', horizon_hours: 72, models: ['ridge'], coverage: 0.9, backtest_origin_step: 6 }, expect.anything())
})


it('lists only items that have data', () => {
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Iron Ore', name_zh: '铁矿', rows: 0 }, { name: 'Gold Ore', name_zh: '金矿', rows: 30 }] } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's2', name: 'S2', start: '', end: null, note: '' }] } as never)
  vi.mocked(hooks.useModels).mockReturnValue({ data: ['naive', 'ridge'] } as never)
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><ForecastForm onStarted={() => {}} /></MemoryRouter></QueryClientProvider>)
  expect(screen.getByRole('option', { name: '金矿' })).toBeInTheDocument()
  expect(screen.queryByRole('option', { name: '铁矿' })).not.toBeInTheDocument()
})
