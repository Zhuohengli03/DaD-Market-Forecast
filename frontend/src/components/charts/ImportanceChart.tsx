import ReactECharts from 'echarts-for-react'
export function ImportanceChart({ items }: { items: { name: string; value: number }[] }) {
  const top = items.slice(0, 12).reverse()
  const option = { animation: false, grid: { left: 140, right: 16, top: 8, bottom: 24 }, xAxis: { type: 'value' }, yAxis: { type: 'category', data: top.map(i => i.name) }, series: [{ type: 'bar', data: top.map(i => i.value), itemStyle: { color: '#2563eb' } }] }
  return <ReactECharts option={option} notMerge style={{ height: 300, width: '100%' }} />
}
