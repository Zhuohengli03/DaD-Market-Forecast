import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import type { ForecastRun } from '../../api/types'
import { modelLabel } from '../../lib/labels'

/** Average error at each step ahead, per method; the chosen method(s) drawn thicker. */
export function HorizonErrorChart({ run }: { run: ForecastRun }) {
  const { t } = useTranslation()
  const names = Object.keys(run.backtest.per_horizon)
  const chosen = new Set([run.selected.name, ...run.selected.members.map(m => m.name)])
  const h = run.backtest.horizon
  const option = {
    animation: false,
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)), axisPointer: { type: 'line' } },
    legend: { bottom: 0, type: 'scroll' }, grid: { left: 56, right: 24, top: 36, bottom: 56 },
    xAxis: { type: 'category', data: Array.from({ length: h }, (_, i) => i + 1), name: t('forecast.hoursAhead'), nameLocation: 'middle', nameGap: 26 },
    yAxis: { type: 'value', name: t('forecast.avgError'), min: 0 },
    series: names.map(n => ({ name: modelLabel(n, t), type: 'line', data: run.backtest.per_horizon[n], symbol: 'none',
      lineStyle: { width: chosen.has(n) ? 3 : 1, type: n === 'naive' || n === 'seasonal_naive_24' ? 'dashed' : 'solid' } })),
  }
  return <ReactECharts option={option} notMerge style={{ height: 300, width: '100%' }} />
}
