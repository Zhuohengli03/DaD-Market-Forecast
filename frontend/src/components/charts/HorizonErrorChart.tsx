import ReactECharts from 'echarts-for-react'
import type { ForecastRun } from '../../api/types'
export function HorizonErrorChart({ run }: { run: ForecastRun }) {
  const names = Object.keys(run.backtest.per_horizon)
  const h = run.backtest.horizon
  const option = { animation: false, tooltip: { trigger: 'axis' }, legend: { top: 0, type: 'scroll' }, grid: { left: 48, right: 16, top: 32, bottom: 32 },
    xAxis: { type: 'category', data: Array.from({ length: h }, (_, i) => i + 1), name: 'h' }, yAxis: { type: 'value', name: 'MAE' },
    series: names.map(n => ({ name: n, type: 'line', data: run.backtest.per_horizon[n], symbol: 'none', lineStyle: { width: n === run.selected.name || run.selected.members.some(m => m.name === n) ? 2.5 : 1 } })) }
  return <ReactECharts option={option} notMerge style={{ height: 260, width: '100%' }} />
}
