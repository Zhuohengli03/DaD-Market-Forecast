import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import type { ForecastRun } from '../../api/types'
import { modelLabel } from '../../lib/labels'

/** Average error at each step ahead, per method; the chosen method(s) drawn thicker. */
const BLOCK_HOURS: Record<string, number> = { '1h': 1, '4h': 4, '1d': 24 }

/** Hours ahead at the end of each scored step: 1, 2, 3… or 4, 8, 12… for 4-hour blocks. */
export function horizonLabels(run: ForecastRun): number[] {
  const k = BLOCK_HOURS[run.backtest.resolution ?? '1h'] ?? 1
  const n = Math.max(0, ...Object.values(run.backtest.per_horizon).map(v => v.length))
  return Array.from({ length: n }, (_, i) => (i + 1) * k)
}

export function HorizonErrorChart({ run }: { run: ForecastRun }) {
  const { t } = useTranslation()
  const names = Object.keys(run.backtest.per_horizon)
  const chosen = new Set([run.selected.name, ...run.selected.members.map(m => m.name)])
  const option = {
    animation: false,
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)), axisPointer: { type: 'line' } },
    legend: { bottom: 0, type: 'scroll' }, grid: { left: 56, right: 24, top: 36, bottom: 56 },
    xAxis: { type: 'category', data: horizonLabels(run), name: t('forecast.hoursAhead'), nameLocation: 'middle', nameGap: 26 },
    yAxis: { type: 'value', name: t('forecast.avgError'), min: 0 },
    series: names.map(n => ({ name: modelLabel(n, t), type: 'line', data: run.backtest.per_horizon[n], symbol: 'none',
      lineStyle: { width: chosen.has(n) ? 3 : 1, type: n === 'naive' || n === 'seasonal_naive_24' ? 'dashed' : 'solid' } })),
  }
  return <ReactECharts option={option} notMerge style={{ height: 300, width: '100%' }} />
}
