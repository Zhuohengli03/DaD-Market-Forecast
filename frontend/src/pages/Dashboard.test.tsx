import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import Dashboard from './Dashboard'
import * as hooks from '../api/hooks'

vi.mock('../api/hooks')

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient()
  return render(<QueryClientProvider client={qc}><MemoryRouter>{ui}</MemoryRouter></QueryClientProvider>)
}

it('renders one card per item with latest price', () => {
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', rows: 10, first_ts: null, last_ts: '2026-01-29T12:00:00+00:00', file: '', csv: '', category: 'ore', description: '', enabled: true }], isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'csv', python: '', items: [], jobs_running: [], scheduler_running: false } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ isSuccess: false } as never)
  vi.mocked(hooks.useSeries).mockReturnValue({ data: { item: 'Gold Ore', resolution: '1h', start: null, end: null, points: [
    { ts: '2026-01-28T12:00:00+00:00', median_price: 20, min_price: 19, max_price: 21, p25: 19.5, p75: 20.5, listings: 3, sold: 1, volume: 5 },
    { ts: '2026-01-29T12:00:00+00:00', median_price: 22, min_price: 21, max_price: 23, p25: 21.5, p75: 22.5, listings: 3, sold: 1, volume: 5 },
  ] } } as never)
  wrap(<Dashboard />)
  expect(screen.getByText('Gold Ore')).toBeInTheDocument()
  expect(screen.getByText('22.00')).toBeInTheDocument()
  expect(screen.getByText('+10.0%')).toBeInTheDocument()
})
