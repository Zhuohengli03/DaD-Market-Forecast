import { buildPriceOption } from './PriceChart'

it('hides min/max outlier lines by default so the median scale stays readable', () => {
  const opt = buildPriceOption([{ ts: '2026-01-01T00:00:00+00:00', median_price: 20, min_price: 1, max_price: 42222, p25: 19, p75: 21, listings: 3, sold: 1, volume: 5 }], null)
  expect(opt.legend.selected).toEqual({ min: false, max: false })
})
