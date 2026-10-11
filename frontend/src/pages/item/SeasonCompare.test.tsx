import { fireEvent, render, screen, within } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { SeasonCompare } from './SeasonCompare'

vi.mock('../../api/hooks')
vi.mock('echarts-for-react', () => ({ default: () => null }))

const row = (id: string, name: string, open: number, close: number) => ({ id, name, market_start: '2026-07-30T00:00:00Z', days: [{ day: 0, median: open }, { day: 1, median: close }],
  open, high: Math.max(open, close), low: Math.min(open, close), close, mean: (open + close) / 2, listings: 120 })

it('lists each season with open, high, low, close, mean and listings', () => {
  vi.mocked(hooks.useSeasonCompare).mockReturnValue({ data: { seasons: [row('s9', 'Season 9', 40, 30), row('s10', 'Season 10', 50, 21)] }, isLoading: false } as never)
  render(<SeasonCompare item="Cobalt Ore" />)
  const rows = screen.getAllByRole('row').slice(1)
  expect(rows).toHaveLength(2)
  expect(within(rows[0]).getByText('第 10 赛季')).toBeInTheDocument()            // newest first
  const cells = within(rows[0]).getAllByRole('cell').map(c => c.textContent)
  expect(cells).toEqual(['第 10 赛季', '50.00', '50.00', '21.00', '21.00', '35.50', '120'])   // season, open, high, low, close, mean, sales
})

it('says why there is nothing to compare', () => {
  vi.mocked(hooks.useSeasonCompare).mockReturnValue({ data: { seasons: [] }, isLoading: false } as never)
  render(<SeasonCompare item="Cobalt Ore" />)
  expect(screen.getByText(/2026-07-31/)).toBeInTheDocument()
})

it('compares any two seasons, the two newest by default', () => {
  vi.mocked(hooks.useSeasonCompare).mockReturnValue({ data: { seasons: [row('s8', 'Season 8', 60, 55), row('s9', 'Season 9', 40, 30), row('s10', 'Season 10', 50, 21)] }, isLoading: false } as never)
  render(<SeasonCompare item="Cobalt Ore" />)
  expect(screen.getByLabelText('对比赛季 1')).toHaveValue('s10')
  expect(screen.getByLabelText('对比赛季 2')).toHaveValue('s9')
  expect(screen.getAllByRole('row').slice(1).map(r => within(r).getAllByRole('cell')[0].textContent)).toEqual(['第 10 赛季', '第 9 赛季'])
  fireEvent.change(screen.getByLabelText('对比赛季 2'), { target: { value: 's8' } })
  expect(screen.getAllByRole('row').slice(1).map(r => within(r).getAllByRole('cell')[0].textContent)).toEqual(['第 10 赛季', '第 8 赛季'])
})
