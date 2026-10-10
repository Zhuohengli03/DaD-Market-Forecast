import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import ReactECharts from 'echarts-for-react'
import { Plus, X } from 'lucide-react'
import { useCatalog, useEquipmentQuery, useEquipmentStats, useSetup } from '../api/hooks'
import type { EquipmentDay } from '../api/types'
import { Notice } from '../components/Notice'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { Field, inputCls } from '../components/ui/Field'
import { Badge } from '../components/ui/Badge'
import { fmtNum, fmtTs } from '../lib/format'
import { useDebounced } from '../lib/useDebounced'
import { useUrlState } from '../lib/urlState'
import { buildQuery, formatStat, type FilterRow, type StatLabel } from './equipment/filters'

const DEPTHS = [10, 20, 40]

function DailyChart({ days }: { days: EquipmentDay[] }) {
  const { t } = useTranslation()
  const x = days.map(d => d.day)
  const option = {
    animation: false, tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(0)) },
    legend: { top: 0, data: [t('equipment.median'), t('equipment.soldMedian')] }, grid: { left: 56, right: 56, top: 32, bottom: 32 },
    xAxis: { type: 'category', data: x }, yAxis: [{ type: 'value', scale: true }, { type: 'value', name: '#', splitLine: { show: false } }],
    series: [
      { name: 'low', type: 'line', data: days.map(d => d.low), lineStyle: { opacity: 0 }, stack: 'r', symbol: 'none', silent: true },
      { name: t('equipment.range'), type: 'line', data: days.map(d => d.high - d.low), lineStyle: { opacity: 0 }, stack: 'r', areaStyle: { color: '#2563eb', opacity: 0.08 }, symbol: 'none', silent: true },
      { name: t('equipment.median'), type: 'line', data: days.map(d => d.median), lineStyle: { color: '#2563eb', width: 2 } },
      { name: t('equipment.soldMedian'), type: 'line', data: days.map(d => d.sold_median), lineStyle: { color: '#16a34a', width: 2 }, connectNulls: true },
      { name: '#', type: 'bar', yAxisIndex: 1, data: days.map(d => d.listings), itemStyle: { color: '#e5e7eb' } },
    ],
  }
  return <ReactECharts option={option} notMerge style={{ height: 300, width: '100%' }} />
}

export default function Equipment() {
  const { t } = useTranslation()
  const setup = useSetup()
  const [type, setType] = useUrlState('type', '')
  const [search, setSearch] = useState('')
  const q = useDebounced(search, 300)
  const types = useCatalog({ q: q || undefined, has_stats: true, page: 1, page_size: 200 })
  const stats = useEquipmentStats(type)
  const query = useEquipmentQuery()
  const [rarity, setRarity] = useState('')
  const [rows, setRows] = useState<FilterRow[]>([])
  const [soldOnly, setSoldOnly] = useState(false)
  const [depth, setDepth] = useState(10)
  const labels = useMemo(() => Object.fromEntries((stats.data?.stats ?? []).map(s => [s.key, { label: s.label, percent: s.percent }])) as Record<string, StatLabel>, [stats.data])
  const selectedName = types.data?.rows.find(r => r.archetype === type)?.name
  const setRow = (i: number, patch: Partial<FilterRow>) => setRows(rs => rs.map((r, k) => (k === i ? { ...r, ...patch } : r)))
  const res = query.data
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('equipment.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('equipment.intro')}</p></div>
      {setup.data && !setup.data.api_key_configured && <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noKey')}</Notice>}
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-3">
          <Field label={t('equipment.type')}>
            <div className="flex gap-2">
              <input className={`${inputCls} w-40`} placeholder={t('catalog.search')} value={search} onChange={e => setSearch(e.target.value)} />
              <select aria-label={t('equipment.type')} className={`${inputCls} flex-1`} value={type} onChange={e => { setType(e.target.value); setRows([]); setRarity('') }}>
                <option value="">{t('equipment.pickType')}</option>
                {type && !selectedName && <option value={type}>{type}</option>}
                {(types.data?.rows ?? []).map(r => <option key={r.archetype} value={r.archetype}>{r.name}</option>)}
              </select>
            </div>
          </Field>
          <Field label={t('equipment.rarity')}>
            <select aria-label={t('equipment.rarity')} className={inputCls} value={rarity} onChange={e => setRarity(e.target.value)}>
              <option value="">{t('equipment.anyRarity')}</option>{(stats.data?.rarities ?? []).map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </Field>
        </div>
        <div className="mt-4 space-y-2">
          {rows.map((r, i) => {
            const s = stats.data?.stats.find(x => x.key === r.key)
            return (
              <div key={i} className="flex flex-wrap items-center gap-2 text-sm">
                <select aria-label={t('equipment.stat', { n: i + 1 })} className={`${inputCls} w-64`} value={r.key} onChange={e => setRow(i, { key: e.target.value })}>
                  <option value="">{t('equipment.pickStat')}</option>
                  {(stats.data?.stats ?? []).map(x => <option key={x.key} value={x.key}>{x.label}{x.kind === 'primary' ? ' (base)' : ''}</option>)}
                </select>
                <input aria-label={t('equipment.min', { n: i + 1 })} className={`${inputCls} w-24`} type="number" step="any" placeholder="min" value={r.min} onChange={e => setRow(i, { min: e.target.value })} />
                <span>~</span>
                <input aria-label={t('equipment.max', { n: i + 1 })} className={`${inputCls} w-24`} type="number" step="any" placeholder="max" value={r.max} onChange={e => setRow(i, { max: e.target.value })} />
                {s && <span className="text-xs text-[var(--muted)]">{t('equipment.observed', { min: s.min, max: s.max })}</span>}
                <button type="button" aria-label={t('equipment.remove')} className="text-[var(--muted)] hover:text-[var(--bad)]" onClick={() => setRows(rs => rs.filter((_, k) => k !== i))}><X size={16} /></button>
              </div>)
          })}
          <Button size="sm" variant="ghost" disabled={!type} onClick={() => setRows(rs => [...rs, { key: '', min: '', max: '' }])}><Plus size={14} />{t('equipment.addFilter')}</Button>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
          <label className="flex items-center gap-1"><input type="checkbox" checked={soldOnly} onChange={e => setSoldOnly(e.target.checked)} />{t('equipment.soldOnly')}</label>
          <label className="flex items-center gap-2">{t('equipment.depth')}<select className={inputCls} value={depth} onChange={e => setDepth(Number(e.target.value))}>{DEPTHS.map(d => <option key={d} value={d}>{t('equipment.depthOption', { n: d * 50, s: Math.ceil(d * 1.1) })}</option>)}</select></label>
          <Button disabled={!type} loading={query.isPending} onClick={() => query.mutate(buildQuery(type, rarity, rows, soldOnly, depth))}>{t('equipment.run')}</Button>
        </div>
        <p className="mt-2 text-xs text-[var(--muted)]">{t('equipment.soldNote')}</p>
        {query.isError && <p role="alert" className="mt-2 text-sm text-[var(--bad)]">{(query.error as Error).message}</p>}
      </Card>
      {!type ? <EmptyState title={t('equipment.noType')} /> : res && (res.count === 0 ? <EmptyState title={t('equipment.none')} /> : <>
        <p className="text-sm">{t('equipment.result', { count: res.count, from: fmtTs(res.covered_from), to: fmtTs(res.covered_to) })} {res.truncated && <span className="text-[var(--warn)]">{t('equipment.truncated', { count: res.count })}</span>}</p>
        {res.daily.length > 0 && <Card title={t('equipment.chartTitle')}><DailyChart days={res.daily} /></Card>}
        <Card title={t('equipment.table')}>
          <div className="overflow-auto"><table className="w-full text-sm">
            <thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('equipment.time')}</th><th>{t('equipment.rarity')}</th><th className="text-right pr-4">{t('equipment.price')}</th><th>{t('equipment.state')}</th><th>{t('equipment.statsCol')}</th></tr></thead>
            <tbody>{res.listings.map(l => (
              <tr key={l.id} className="border-t border-[var(--line)] align-top">
                <td className="py-1.5 whitespace-nowrap">{fmtTs(l.created_at)}</td><td>{l.rarity}</td>
                <td className="text-right pr-4 tabular-nums font-medium">{fmtNum(l.price_per_unit)}</td>
                <td>{l.has_sold ? <Badge tone="ok">{t('equipment.sold')}</Badge> : l.listing_state === 'active' ? <Badge>{t('equipment.active')}</Badge> : <Badge tone="warn">{t('equipment.gone')}</Badge>}</td>
                <td className="py-1"><div className="flex flex-wrap gap-1">{Object.entries(l.attributes).map(([k, v]) => <span key={k} className={`text-xs px-1.5 py-0.5 rounded ${k.startsWith('secondary') ? 'bg-blue-50 text-blue-800' : 'bg-gray-100'}`}>{formatStat(k, v, labels)}</span>)}</div></td>
              </tr>))}</tbody>
          </table></div>
        </Card>
      </>)}
    </div>
  )
}
