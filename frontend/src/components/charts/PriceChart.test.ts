import { buildPriceOption } from './PriceChart'

it('keeps the UTC date at a wipe boundary regardless of the browser timezone', () => {
  const opt = buildPriceOption([])
  for (const axis of opt.xAxis) {
    expect(axis.axisLabel.formatter('2026-10-09T00:00:00Z')).toBe('2026-10-09 00:00Z')
  }
})

it('hides min/max outlier lines by default so the median scale stays readable', () => {
  const opt = buildPriceOption([{ ts: '2026-01-01T00:00:00+00:00', median_price: 20, min_price: 1, max_price: 42222, p25: 19, p75: 21, listings: 3, sold: 1, volume: 5 }], null)
  expect(opt.legend.selected).toEqual({ min: false, max: false })
  expect(opt.brush.xAxisIndex).toBe(0)
})

it('caps the price axis with a robust maximum so one outlier hour cannot flatten the median', () => {
  const pt = (median: number, p75: number) => ({ ts: 't', median_price: median, min_price: 1, max_price: p75, p25: median - 1, p75, listings: 3, sold: 1, volume: 5 })
  const points = [...Array.from({ length: 100 }, (_, i) => pt(20 + (i % 5), 22)), pt(40000, 42222)]
  const opt = buildPriceOption(points, null)
  expect(opt.yAxis[0].max).toBeLessThan(100)
  expect(opt.yAxis[0].max).toBeGreaterThan(25)
  expect(opt.yAxis[0].min).toBe(0)
})

it('marks game updates on the chart, snapped to the next point, strong ones in red', () => {
  const pts = ['2026-01-01T00:00:00Z', '2026-01-01T04:00:00Z', '2026-01-01T08:00:00Z'].map(ts => ({ ts, median_price: 1, min_price: 1, max_price: 1, p25: 1, p75: 1, listings: 1, sold: 1, volume: 1 }))
  const opt = buildPriceOption(pts, null, undefined, [{ ts: '2025-12-01T00:00:00Z', label: 'before the chart', strong: false }, { ts: '2026-01-01T03:00:00Z', label: 'Hotfix #9', strong: true }, { ts: '2026-02-01T00:00:00Z', label: 'later', strong: false }])
  const median = opt.series.find(s => s.name === 'median') as { markLine: { data: { xAxis: string; lineStyle: { color: string } }[] } }
  expect(median.markLine.data).toEqual([expect.objectContaining({ xAxis: '2026-01-01T04:00:00Z', lineStyle: expect.objectContaining({ color: '#dc2626' }) })])
})

it('places a mark by time, whatever the timestamp format', async () => {
  const { snapTs } = await import('./PriceChart')
  const pts = ['2026-01-01T00:00:00+00:00', '2026-01-01T04:00:00+00:00'].map(ts => ({ ts, median_price: 1, min_price: 1, max_price: 1, p25: 1, p75: 1, listings: 1, sold: 1, volume: 1 }))
  expect(snapTs(pts, '2026-01-01T00:00:00Z')).toBe('2026-01-01T00:00:00+00:00')     // same instant, other format
  expect(snapTs(pts, '2026-01-01T01:00:00Z')).toBe('2026-01-01T04:00:00+00:00')
  expect(snapTs(pts, '2026-01-02T00:00:00Z')).toBeNull()
  expect(snapTs(pts, '2025-12-31T00:00:00Z')).toBeNull()
})
