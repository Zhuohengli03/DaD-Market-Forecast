import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import type { SeriesPoint } from '../../api/types'
import { fmtTs } from '../../lib/format'

/** 99th percentile of the hourly medians x 1.4: thin hours with bundle outliers are clipped, not scaled to. */
export function robustPriceMax(points: SeriesPoint[]): number | undefined {
  const vals = points.map(p => p.median_price).filter((v): v is number => v !== null && Number.isFinite(v)).sort((a, b) => a - b)
  if (!vals.length) return undefined
  const p99 = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))]
  const p75s = points.map(p => p.p75).filter((v): v is number => v !== null && v <= p99 * 1.4)
  return Math.ceil(Math.max(p99, ...p75s) * 1.4)
}

export interface Legend { median: string; band: string; min: string; max: string; volume: string; listings: string }
const DEFAULT_LEGEND: Legend = { median: 'median', band: 'p25–p75', min: 'min', max: 'max', volume: 'volume', listings: 'listings' }

export function buildPriceOption(points: SeriesPoint[], highlightTs?: string | null, L: Legend = DEFAULT_LEGEND) {
  const x = points.map(p => p.ts)
  const yMax = robustPriceMax(points)
  return {
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' }, valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)) },
    // min/max carry bundle outliers (e.g. 42,222 per unit) that would flatten the median; opt-in via legend
    legend: { top: 0, data: [L.median, L.band, L.min, L.max, L.volume, L.listings], selected: { [L.min]: false, [L.max]: false } },
    toolbox: { right: 8, top: 0, feature: { brush: { type: ['lineX', 'clear'] } } },
    brush: { xAxisIndex: 0, brushLink: 'all', throttleType: 'debounce', throttleDelay: 200, outOfBrush: { colorAlpha: 0.3 } },
    grid: [{ left: 56, right: 56, top: 36, height: '52%' }, { left: 56, right: 56, top: '70%', height: '18%' }],
    xAxis: [{ type: 'category', data: x, gridIndex: 0, axisLabel: { formatter: (v: string) => fmtTs(v) } }, { type: 'category', data: x, gridIndex: 1, axisLabel: { formatter: (v: string) => fmtTs(v) } }],
    yAxis: [{ type: 'value', min: 0, max: yMax, gridIndex: 0 }, { type: 'value', gridIndex: 1 }],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1], bottom: 4 }],
    series: [
      { name: L.band, type: 'line', data: points.map(p => (p.p25 === null ? null : Math.max(0, p.p25))), lineStyle: { opacity: 0 }, stack: 'band', symbol: 'none', silent: true },
      { name: L.band, type: 'line', data: points.map(p => (p.p75 === null || p.p25 === null ? null : Math.max(0, p.p75 - Math.max(0, p.p25)))), lineStyle: { opacity: 0 }, areaStyle: { color: '#2563eb', opacity: 0.12 }, stack: 'band', symbol: 'none', silent: true },
      { name: L.min, type: 'line', data: points.map(p => p.min_price), symbol: 'none', lineStyle: { width: 1, type: 'dotted', color: '#9ca3af' } },
      { name: L.max, type: 'line', data: points.map(p => p.max_price), symbol: 'none', lineStyle: { width: 1, type: 'dotted', color: '#9ca3af' } },
      { name: L.median, type: 'line', data: points.map(p => p.median_price), symbol: 'none', lineStyle: { width: 2, color: '#2563eb' }, connectNulls: false,
        markLine: highlightTs ? { silent: true, symbol: 'none', data: [{ xAxis: highlightTs }], lineStyle: { color: '#d97706', width: 2 } } : undefined },
      { name: L.volume, type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: points.map(p => p.volume), itemStyle: { color: '#93c5fd' } },
      { name: L.listings, type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: points.map(p => p.listings), symbol: 'none', lineStyle: { color: '#6b7280', width: 1 } },
    ],
  }
}

interface BrushEvent { batch?: { areas?: { coordRange?: [number, number] }[] }[] }

export function PriceChart({ points, highlightTs, onRange }: { points: SeriesPoint[]; highlightTs?: string | null; onRange?: (r: [number, number] | null) => void }) {
  const { t } = useTranslation()
  const L = t('market.legend', { returnObjects: true }) as Legend
  const onEvents = {
    brushselected: (e: BrushEvent) => {
      const range = e.batch?.[0]?.areas?.[0]?.coordRange
      onRange?.(range ? [Math.round(range[0]), Math.round(range[1])] : null)
    },
  }
  return <ReactECharts option={buildPriceOption(points, highlightTs, L)} notMerge onEvents={onEvents} style={{ height: 460, width: '100%' }} />
}
