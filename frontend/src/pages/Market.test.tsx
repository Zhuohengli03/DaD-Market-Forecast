import { act, render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import Market from './Market'

vi.mock('../api/hooks')
vi.mock('echarts-for-react', () => ({ default: () => null }))
vi.mock('../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
vi.mock('./market/ListingsTable', () => ({ ListingsTable: () => null }))

function Where() { const l = useLocation(); return <p data-testid="url">{l.pathname + l.search}</p> }

it('starts a default forecast for the shown item and season, then opens it', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', name_zh: '金矿', rows: 30 }] } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's11', name: '第 11 赛季', start: '2026-10-09T00:00:00Z', end: null }] } as never)
  vi.mocked(hooks.useSeries).mockReturnValue({ data: { points: [{ ts: '2026-10-09T10:00:00+00:00', median_price: 10, p25: 9, p75: 11, min_price: 8, max_price: 12, listings: 3, sold: 1, volume: 4 }] }, isLoading: false } as never)
  vi.mocked(hooks.useStartForecast).mockReturnValue({ mutate, isPending: false } as never)
  render(<MemoryRouter initialEntries={['/market']}><Routes><Route path="*" element={<><Market /><Where /></>} /></Routes></MemoryRouter>)
  fireEvent.click(screen.getByRole('button', { name: '一键预测' }))
  expect(mutate.mock.calls[0][0]).toEqual({ item: 'Gold Ore', season_id: 's11', horizon_hours: 24, models: ['ridge', 'hgb', 'rf', 'ets', 'arima'], coverage: 0.9 })
  act(() => mutate.mock.calls[0][1].onSuccess('job42'))
  expect(screen.getByTestId('url').textContent).toBe('/forecast?item=Gold%20Ore&season=s11&job=job42')
})
