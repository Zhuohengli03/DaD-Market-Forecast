import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import { useSeasonCompare } from '../../api/hooks'
import { Card } from '../../components/ui/Card'
import { inputCls } from '../../components/ui/Field'
import { fmtNum } from '../../lib/format'
import { seasonLabel } from '../../lib/labels'

const COLORS = ['#2563eb', '#d97706', '#7c3aed', '#16a34a', '#db2777', '#0891b2']

/** Every season's daily median on one chart, lined up by days since trading opened, plus a row of numbers each. */
export function SeasonCompare({ item }: { item: string }) {
  const { t } = useTranslation()
  const q = useSeasonCompare(item)
  const all = [...(q.data?.seasons ?? [])].reverse()          // newest first
  const [a, setA] = useState<string | null>(null)
  const [b, setB] = useState<string | null>(null)
  const first = all.find(s => s.id === a) ?? all[0]
  const second = all.find(s => s.id === b && s.id !== first?.id) ?? all.find(s => s.id !== first?.id)
  const seasons = [first, second].filter((s): s is NonNullable<typeof s> => !!s)
  const pick = (label: string, value: string | undefined, set: (id: string) => void) => (
    <select aria-label={label} className={inputCls} value={value ?? ''} onChange={e => set(e.target.value)}>
      {all.map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}</option>)}</select>)
  const option = {
    animation: false,
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)) },
    legend: { top: 0, type: 'scroll' }, grid: { left: 56, right: 24, top: 36, bottom: 40 },
    xAxis: { type: 'value', name: t('item.dayOfSeason'), nameLocation: 'middle', nameGap: 26, minInterval: 1 },
    yAxis: { type: 'value', scale: true, name: t('chart.unitPrice') },
    series: seasons.map((s, i) => ({ name: seasonLabel(s.name, t), type: 'line', color: COLORS[i % COLORS.length], symbol: 'none',
      lineStyle: { width: i === 0 ? 2.5 : 1.5 }, data: s.days.map(d => [d.day, d.median]) })),
  }
  return (
    <Card title={t('item.compare')} actions={all.length > 1 ? <>{pick(t('item.compareA'), first?.id, setA)}<span className="text-[var(--muted)]">vs</span>{pick(t('item.compareB'), second?.id, setB)}</> : undefined}>
      {q.isLoading ? <p className="text-sm">{t('common.loading')}</p> : !seasons.length ? <p className="text-sm text-[var(--muted)]">{t('item.compareEmpty')}</p> : <>
        <p className="text-xs text-[var(--muted)] mb-2">{t('item.compareHint')}</p>
        <ReactECharts option={option} notMerge style={{ height: 300, width: '100%' }} />
        <table className="w-full text-sm tabular-nums mt-2">
          <thead className="text-left text-xs text-[var(--muted)]"><tr><th className="py-1 font-normal">{t('common.season')}</th>
            {(['open', 'high', 'low', 'close', 'mean'] as const).map(k => <th key={k} className="text-right font-normal">{t(`item.stat.${k}`)}</th>)}
            <th className="text-right font-normal">{t('item.stat.listings')}</th></tr></thead>
          <tbody>{seasons.map(s => <tr key={s.id} className="border-t border-[var(--line)]"><td className="py-1">{seasonLabel(s.name, t)}</td>
            {(['open', 'high', 'low', 'close', 'mean'] as const).map(k => <td key={k} className="text-right">{fmtNum(s[k])}</td>)}
            <td className="text-right">{s.listings.toLocaleString()}</td></tr>)}</tbody>
        </table></>}
    </Card>
  )
}
