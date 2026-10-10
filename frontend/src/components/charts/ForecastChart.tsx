import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import type { Actual, ForecastRun } from '../../api/types'
import { fmtTs } from '../../lib/format'
import { seasonLabel } from '../../lib/labels'

const REF_COLORS = ['#d97706', '#7c3aed', '#db2777']

/** Other seasons at their raw hourly medians, aligned by hours since season start (spec §6.7). */
export function referenceSeries(run: ForecastRun, label: (name: string) => string = n => `ref: ${n}`) {
  const nH = run.series_tail.length
  return run.references.map((r, i) => {
    const byH = new Map(r.points.map(p => [p.h, p.median_price]))
    return { name: label(r.name), type: 'line', color: REF_COLORS[i % REF_COLORS.length], data: [...Array(nH).fill(null), ...run.forecast.map(f => byH.get(f.h) ?? null)], symbol: 'none', lineStyle: { width: 1, type: 'dotted' } }
  })
}

export function ForecastChart({ run, showRefs, actuals }: { run: ForecastRun; showRefs: boolean; actuals?: Actual[] }) {
  const { t } = useTranslation()
  const N = { history: t('chart.history'), band: t('chart.band'), forecast: t('chart.forecast'), actual: t('chart.actual') }
  const histX = run.series_tail.map(p => p.ts)
  const fcX = run.forecast.map(p => p.ts)
  const x = [...histX, ...fcX]
  const pad = (arr: (number | null)[], before: number, after: number) => [...Array(before).fill(null), ...arr, ...Array(after).fill(null)]
  const nH = histX.length, nF = fcX.length
  const lower = run.forecast.map(p => p.lower), upper = run.forecast.map(p => p.upper)
  const actualMap = new Map((actuals ?? []).map(a => [a.ts, a.median_price]))
  const series: object[] = [
    { name: N.history, type: 'line', color: '#374151', data: pad(run.series_tail.map(p => p.median_price), 0, nF), symbol: 'none', lineStyle: { width: 1.5 } },
    { name: N.band, type: 'line', color: '#93c5fd', data: pad(lower.map(l => Math.max(0, l)), nH, 0), lineStyle: { opacity: 0 }, stack: 'ci', symbol: 'none', silent: true, tooltip: { show: false } },
    { name: N.band, type: 'line', color: '#93c5fd', data: pad(upper.map((u, i) => Math.max(0, u - Math.max(0, lower[i]))), nH, 0), lineStyle: { opacity: 0 }, stack: 'ci', areaStyle: { color: '#2563eb', opacity: 0.15 }, symbol: 'none', silent: true, tooltip: { show: false } },
    { name: N.forecast, type: 'line', color: '#2563eb', data: pad(run.forecast.map(p => p.pred), nH, 0), symbol: 'none', lineStyle: { width: 2, type: 'dashed' } },
  ]
  if (actualMap.size) series.push({ name: N.actual, type: 'line', color: '#16a34a', data: pad(fcX.map(ts => actualMap.get(ts) ?? null), nH, 0), symbol: 'circle', symbolSize: 4, lineStyle: { width: 1.5 } })
  if (showRefs) series.push(...referenceSeries(run, n => t('chart.reference', { name: seasonLabel(n, t) })))
  const names = series.map(s => (s as { name: string }).name).filter((n, i, a) => a.indexOf(n) === i)
  const option = { animation: false, tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)) },
    legend: { top: 0, type: 'scroll', data: names.map(n => (n === N.band ? { name: n, icon: 'roundRect' } : n)) }, grid: { left: 56, right: 24, top: 36, bottom: 60 },
    xAxis: { type: 'category', data: x, axisLabel: { formatter: (v: string) => fmtTs(v) } }, yAxis: { type: 'value', min: 0, name: t('chart.unitPrice') },
    dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 4 }], series }
  return <ReactECharts option={option} notMerge style={{ height: 420, width: '100%' }} />
}
