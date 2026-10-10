import { change24h, floorToBucket, summarizeRange } from './market'

const pt = (ts: string, p: number | null) => ({ ts, median_price: p, min_price: p, max_price: p, p25: p, p75: p, listings: 1, sold: 0, volume: 1 })

it('compares against the price 24 real hours earlier, skipping gaps', () => {
  const pts = [pt('2026-01-01T00:00:00+00:00', 10), pt('2026-01-01T01:00:00+00:00', null), pt('2026-01-01T20:00:00+00:00', 30), pt('2026-01-02T01:00:00+00:00', 12)]
  expect(change24h(pts)).toEqual({ latest: 12, change: 0.2 })
  expect(change24h([pt('2026-01-01T00:00:00+00:00', 10)])).toEqual({ latest: 10, change: null })
})

it('floors a timestamp to the chart bucket', () => {
  expect(floorToBucket('2026-01-01T13:47:00Z', '1h')).toBe('2026-01-01T13:00:00+00:00')
  expect(floorToBucket('2026-01-01T13:47:00Z', '4h')).toBe('2026-01-01T12:00:00+00:00')
  expect(floorToBucket('2026-01-01T13:47:00Z', '1d')).toBe('2026-01-01T00:00:00+00:00')
})

it('summarizes a brushed range', () => {
  const pts = [pt('a', 10), pt('b', null), pt('c', 30), pt('d', 20)]
  expect(summarizeRange(pts, 0, 3)).toEqual({ hours: 4, low: 10, median: 20, high: 30, listings: 4 })
})
