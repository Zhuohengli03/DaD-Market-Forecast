import type { Resolution, SeriesPoint } from '../api/types'

const HOUR = 3.6e6

/** Latest hourly median and its change versus the last price at least 24 real hours earlier. */
export function change24h(points: SeriesPoint[]): { latest: number | null; change: number | null } {
  const valid = points.filter(p => p.median_price !== null) as (SeriesPoint & { median_price: number })[]
  const last = valid.at(-1)
  if (!last) return { latest: null, change: null }
  const cutoff = new Date(last.ts).getTime() - 24 * HOUR
  const ref = [...valid].reverse().find(p => new Date(p.ts).getTime() <= cutoff)
  return { latest: last.median_price, change: ref && ref.median_price ? Math.round((last.median_price / ref.median_price - 1) * 1e6) / 1e6 : null }
}

/** Start of the chart bucket (UTC) that contains `iso`, formatted like the API's timestamps. */
export function floorToBucket(iso: string, res: Resolution): string {
  const d = new Date(iso)
  d.setUTCMinutes(0, 0, 0)
  if (res === '4h') d.setUTCHours(d.getUTCHours() - (d.getUTCHours() % 4))
  if (res === '1d') d.setUTCHours(0)
  return d.toISOString().replace('.000Z', '+00:00')
}

export interface RangeSummary { hours: number; low: number | null; median: number | null; high: number | null; listings: number }

export function summarizeRange(points: SeriesPoint[], from: number, to: number): RangeSummary {
  const slice = points.slice(Math.max(0, from), to + 1)
  const prices = slice.map(p => p.median_price).filter((v): v is number => v !== null).sort((a, b) => a - b)
  const mid = prices.length ? (prices.length % 2 ? prices[(prices.length - 1) / 2] : (prices[prices.length / 2 - 1] + prices[prices.length / 2]) / 2) : null
  return { hours: slice.length, low: prices[0] ?? null, median: mid, high: prices.at(-1) ?? null, listings: slice.reduce((s, p) => s + p.listings, 0) }
}
