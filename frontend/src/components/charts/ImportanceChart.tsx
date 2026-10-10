import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'

/** price_lag_24 -> "24 小时前的价格", price_roll_mean_6 -> "过去 6 小时均价", … */
export function featureLabel(name: string, t: TFunction): string {
  const m = /^(price_lag|volume_lag|listings_lag|price_roll_mean|price_roll_std|price_change)_(\d+)$/.exec(name)
  if (m) return t(`feature.${m[1]}`, { n: m[2] })
  return t(`feature.${name}`, { defaultValue: name })
}

export function ImportanceChart({ items }: { items: { name: string; value: number }[] }) {
  const { t } = useTranslation()
  const top = items.slice(0, 12).reverse()
  const total = items.reduce((s, i) => s + Math.max(i.value, 0), 0) || 1
  const option = {
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => `${(v * 100).toFixed(1)}%` },
    grid: { left: 150, right: 24, top: 8, bottom: 28 },
    xAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
    yAxis: { type: 'category', data: top.map(i => featureLabel(i.name, t)) },
    series: [{ type: 'bar', data: top.map(i => Math.max(i.value, 0) / total), color: '#2563eb' }],
  }
  return <ReactECharts option={option} notMerge style={{ height: 320, width: '100%' }} />
}
