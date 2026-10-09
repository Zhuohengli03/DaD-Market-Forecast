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
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore' }] } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's2', name: 'S2', start: '', end: null, note: '' }] } as never)
  vi.mocked(hooks.useModels).mockReturnValue({ data: ['naive', 'seasonal_naive_24', 'ridge', 'hgb'] } as never)
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate, isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><ForecastForm onStarted={() => {}} /></MemoryRouter></QueryClientProvider>)
  fireEvent.click(screen.getByLabelText('hgb'))
  fireEvent.click(screen.getByText('开始预测'))
  expect(mutate).toHaveBeenCalledWith({ item: 'Gold Ore', season_id: 's2', horizon_hours: 72, models: ['ridge'], coverage: 0.9 }, expect.anything())
})
