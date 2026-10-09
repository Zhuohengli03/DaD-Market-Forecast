import ReactECharts from 'echarts-for-react'
export function Sparkline({ values, color = '#2563eb' }: { values: (number | null)[]; color?: string }) {
  const option = { grid: { left: 0, right: 0, top: 2, bottom: 2 }, xAxis: { type: 'category', show: false, data: values.map((_, i) => i) }, yAxis: { type: 'value', show: false, scale: true }, series: [{ type: 'line', data: values, showSymbol: false, smooth: true, lineStyle: { width: 1.5, color }, areaStyle: { color, opacity: 0.08 } }], animation: false }
  return <ReactECharts option={option} style={{ height: 40, width: '100%' }} />
}
