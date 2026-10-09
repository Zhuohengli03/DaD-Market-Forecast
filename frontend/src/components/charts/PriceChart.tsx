import ReactECharts from 'echarts-for-react'
import type { SeriesPoint } from '../../api/types'
import { fmtTs } from '../../lib/format'

export function buildPriceOption(points: SeriesPoint[], highlightTs?: string | null) {
  const x = points.map(p => p.ts)
  return {
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' }, valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)) },
    // min/max carry bundle outliers (e.g. 42,222 per unit) that would flatten the median; opt-in via legend
    legend: { top: 0, data: ['median', 'p25–p75', 'min', 'max', 'volume', 'listings'], selected: { min: false, max: false } },
    grid: [{ left: 56, right: 56, top: 36, height: '52%' }, { left: 56, right: 56, top: '70%', height: '18%' }],
    xAxis: [{ type: 'category', data: x, gridIndex: 0, axisLabel: { formatter: (v: string) => fmtTs(v) } }, { type: 'category', data: x, gridIndex: 1, axisLabel: { formatter: (v: string) => fmtTs(v) } }],
    yAxis: [{ type: 'value', scale: true, gridIndex: 0, name: 'price' }, { type: 'value', gridIndex: 1, name: 'vol' }],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1], bottom: 4 }],
    series: [
      { name: 'p25–p75', type: 'line', data: points.map(p => p.p25), lineStyle: { opacity: 0 }, stack: 'band', symbol: 'none', silent: true },
      { name: 'p25–p75', type: 'line', data: points.map(p => (p.p75 ?? 0) - (p.p25 ?? 0)), lineStyle: { opacity: 0 }, areaStyle: { color: '#2563eb', opacity: 0.12 }, stack: 'band', symbol: 'none', silent: true },
      { name: 'min', type: 'line', data: points.map(p => p.min_price), symbol: 'none', lineStyle: { width: 1, type: 'dotted', color: '#9ca3af' } },
      { name: 'max', type: 'line', data: points.map(p => p.max_price), symbol: 'none', lineStyle: { width: 1, type: 'dotted', color: '#9ca3af' } },
      { name: 'median', type: 'line', data: points.map(p => p.median_price), symbol: 'none', lineStyle: { width: 2, color: '#2563eb' }, connectNulls: false,
        markLine: highlightTs ? { silent: true, symbol: 'none', data: [{ xAxis: highlightTs }], lineStyle: { color: '#d97706', width: 2 } } : undefined },
      { name: 'volume', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: points.map(p => p.volume), itemStyle: { color: '#93c5fd' } },
      { name: 'listings', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: points.map(p => p.listings), symbol: 'none', lineStyle: { color: '#6b7280', width: 1 } },
    ],
  }
}

export function PriceChart({ points, highlightTs }: { points: SeriesPoint[]; highlightTs?: string | null }) {
  return <ReactECharts option={buildPriceOption(points, highlightTs)} notMerge style={{ height: 460, width: '100%' }} />
}
