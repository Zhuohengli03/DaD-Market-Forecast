import { buildPriceOption } from './PriceChart'

it('hides min/max outlier lines by default so the median scale stays readable', () => {
  const opt = buildPriceOption([{ ts: '2026-01-01T00:00:00+00:00', median_price: 20, min_price: 1, max_price: 42222, p25: 19, p75: 21, listings: 3, sold: 1, volume: 5 }], null)
  expect(opt.legend.selected).toEqual({ min: false, max: false })
})

it('caps the price axis with a robust maximum so one outlier hour cannot flatten the median', () => {
  const pt = (median: number, p75: number) => ({ ts: 't', median_price: median, min_price: 1, max_price: p75, p25: median - 1, p75, listings: 3, sold: 1, volume: 5 })
  const points = [...Array.from({ length: 100 }, (_, i) => pt(20 + (i % 5), 22)), pt(40000, 42222)]
  const opt = buildPriceOption(points, null)
  expect(opt.yAxis[0].max).toBeLessThan(100)
  expect(opt.yAxis[0].max).toBeGreaterThan(25)
  expect(opt.yAxis[0].min).toBe(0)
})
