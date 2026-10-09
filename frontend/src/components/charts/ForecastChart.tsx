import ReactECharts from 'echarts-for-react'
import type { Actual, ForecastRun } from '../../api/types'
import { fmtTs } from '../../lib/format'

export function ForecastChart({ run, showRefs, actuals }: { run: ForecastRun; showRefs: boolean; actuals?: Actual[] }) {
  const histX = run.series_tail.map(p => p.ts)
  const fcX = run.forecast.map(p => p.ts)
  const x = [...histX, ...fcX]
  const pad = (arr: (number | null)[], before: number, after: number) => [...Array(before).fill(null), ...arr, ...Array(after).fill(null)]
  const nH = histX.length, nF = fcX.length
  const lower = run.forecast.map(p => p.lower), upper = run.forecast.map(p => p.upper)
  const actualMap = new Map((actuals ?? []).map(a => [a.ts, a.median_price]))
  const series: object[] = [
    { name: 'history', type: 'line', data: pad(run.series_tail.map(p => p.median_price), 0, nF), symbol: 'none', lineStyle: { color: '#1f2937', width: 1.5 } },
    { name: 'band', type: 'line', data: pad(lower, nH, 0), lineStyle: { opacity: 0 }, stack: 'ci', symbol: 'none', silent: true },
    { name: 'band', type: 'line', data: pad(upper.map((u, i) => u - lower[i]), nH, 0), lineStyle: { opacity: 0 }, stack: 'ci', areaStyle: { color: '#2563eb', opacity: 0.15 }, symbol: 'none', silent: true },
    { name: 'forecast', type: 'line', data: pad(run.forecast.map(p => p.pred), nH, 0), symbol: 'none', lineStyle: { color: '#2563eb', width: 2, type: 'dashed' } },
  ]
  if (actualMap.size) series.push({ name: 'actual', type: 'line', data: pad(fcX.map(ts => actualMap.get(ts) ?? null), nH, 0), symbol: 'circle', symbolSize: 4, lineStyle: { color: '#16a34a', width: 1.5 } })
  if (showRefs) {
    const lastHist = run.series_tail.at(-1)?.median_price ?? null
    run.references.forEach((r, i) => {
      const first = r.points[0]?.median_price ?? 1
      const scale = lastHist && first ? lastHist / first : 1   // align level to current price
      const byH = new Map(r.points.map(p => [p.h, p.median_price * scale]))
      series.push({ name: `ref: ${r.name}`, type: 'line', data: pad(run.forecast.map(f => byH.get(f.h) ?? null), nH, 0), symbol: 'none', lineStyle: { color: ['#d97706', '#7c3aed', '#db2777'][i % 3], width: 1, type: 'dotted' } })
    })
  }
  const option = { animation: false, tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)) }, legend: { top: 0 }, grid: { left: 56, right: 24, top: 36, bottom: 60 },
    xAxis: { type: 'category', data: x, axisLabel: { formatter: (v: string) => fmtTs(v) } }, yAxis: { type: 'value', scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 4 }], series }
  return <ReactECharts option={option} notMerge style={{ height: 420, width: '100%' }} />
}
