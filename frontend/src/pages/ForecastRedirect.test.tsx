import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { vi } from 'vitest'
import * as hooks from '../api/hooks'
import ForecastRedirect from './ForecastRedirect'

vi.mock('../api/hooks')
function Where() { const l = useLocation(); return <p>at:{l.pathname}{l.search}</p> }

it('an old forecast link opens that forecast on its item page', () => {
  vi.mocked(hooks.useRuns).mockReturnValue({ data: [{ run_id: 'r2', item: 'Gold Ore', season_id: 's10' }], isSuccess: true } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', item_id: 'id.item.gold_ores_5001' }], isSuccess: true } as never)
  render(<MemoryRouter initialEntries={['/forecast?run=r2']}><Routes>
    <Route path="/forecast" element={<ForecastRedirect />} /><Route path="*" element={<Where />} /></Routes></MemoryRouter>)
  expect(screen.getByText('at:/item/id.item.gold_ores?q=Gold+Ore&season=s10&run=r2')).toBeInTheDocument()
})

it('without a forecast to show, goes to the overview', () => {
  vi.mocked(hooks.useRuns).mockReturnValue({ data: [], isSuccess: true } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [], isSuccess: true } as never)
  render(<MemoryRouter initialEntries={['/forecast']}><Routes>
    <Route path="/forecast" element={<ForecastRedirect />} /><Route path="*" element={<Where />} /></Routes></MemoryRouter>)
  expect(screen.getByText('at:/')).toBeInTheDocument()
})
