import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { NextHourRange, priceVerdict } from './NextHourRange'

vi.mock('../../api/hooks')

const P = { p10: 13.1, p25: 17.4, p50: 20.6, p75: 23.4, p90: 29.6 }
const DATA = { level: 20.8, level_hour: '2026-10-09T02:00:00+00:00', next_hour: '2026-10-09T03:00:00+00:00', prices: P,
  check: { mid: 0.45, wide: 0.77, hours: 168 }, calib_days: 3, check_days: 7, sales_used: 1461, stale_hours: 0.5 }

it('judges a price against next hour sales', () => {
  expect([10, 15, 20, 25, 40].map(x => priceVerdict(x, P))).toEqual(['very_cheap', 'cheap', 'normal', 'pricey', 'very_pricey'])
})

it('shows where half and most of next hour sales land, how often that held, and judges a typed price', () => {
  vi.mocked(hooks.useNextHour).mockReturnValue({ data: DATA, isLoading: false, isError: false } as never)
  render(<NextHourRange item="Cobalt Ore" seasonId="s10" />)
  expect(screen.getByText(/一半的成交在 17.40 – 23.40/)).toBeInTheDocument()
  expect(screen.getByText(/八成在 13.10 – 29.60/)).toBeInTheDocument()
  expect(screen.getByText(/中间一半 45%（目标 50%）/)).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('看到的挂单价'), { target: { value: '15' } })
  expect(screen.getByText(/偏便宜/)).toBeInTheDocument()
})

it('warns when the newest sale is old, and explains when there is too little data', () => {
  vi.mocked(hooks.useNextHour).mockReturnValue({ data: { ...DATA, stale_hours: 40 }, isLoading: false, isError: false } as never)
  const { unmount } = render(<NextHourRange item="Cobalt Ore" />)
  expect(screen.getByText(/不是现在/)).toBeInTheDocument()
  unmount()
  vi.mocked(hooks.useNextHour).mockReturnValue({ data: undefined, isLoading: false, isError: true, error: new Error('最近 3 天只有 20 条成交') } as never)
  render(<NextHourRange item="Cobalt Ore" />)
  expect(screen.getByText(/最近 3 天只有 20 条成交/)).toBeInTheDocument()
})
