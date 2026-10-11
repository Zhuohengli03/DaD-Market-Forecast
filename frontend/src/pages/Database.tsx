import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Trash2 } from 'lucide-react'
import { useDbDelete, useDbItems, useDbRows, useSeasons } from '../api/hooks'
import type { DbDelete } from '../api/types'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { inputCls } from '../components/ui/Field'
import { useToast } from '../components/ui/Toast'
import { fmtDate, fmtNum, fmtTsUtc } from '../lib/format'
import { itemName, seasonLabel } from '../lib/labels'
import { useUrlState } from '../lib/urlState'

const PAGE = 100
const dayStart = (d: string) => (d ? `${d}T00:00:00Z` : '')
/** The end of a picked day: midnight UTC of the next day (the range is end-exclusive). */
const dayEnd = (d: string) => (d ? new Date(Date.parse(`${d}T00:00:00Z`) + 86_400_000).toISOString() : '')

/** Every listing stored on this machine: look and delete. Editing is deliberately impossible. */
export default function Database() {
  const { t, i18n } = useTranslation()
  const toast = useToast()
  const items = useDbItems(); const seasons = useSeasons(); const del = useDbDelete()
  const list = items.data?.items ?? []
  const [keyParam, setKey] = useUrlState('key', '')
  const key = list.some(i => i.key === keyParam) ? keyParam : list[0]?.key ?? ''
  const item = list.find(i => i.key === key)
  const label = item ? itemName(item, i18n.language) : ''
  const [range, setRange] = useState<{ season: string; from: string; to: string }>({ season: '', from: '', to: '' })
  const [page, setPage] = useState(1)
  const [picked, setPicked] = useState<string[]>([])
  const season = (seasons.data ?? []).find(s => s.id === range.season)
  const start = season ? season.start : dayStart(range.from)
  const end = season ? season.end ?? '' : dayEnd(range.to)
  const rows = useDbRows(key, start, end, page)
  const total = rows.data?.total ?? 0
  const filtered = !!(start || end)
  const reset = () => { setPage(1); setPicked([]) }
  const remove = (body: DbDelete, message: string) => {
    if (!window.confirm(message)) return
    del.mutate(body, { onSuccess: d => { toast.push(t('db.deleted', { n: d }), 'ok'); reset() }, onError: e => toast.push((e as Error).message, 'error') })
  }
  const toggle = (id: string) => setPicked(p => (p.includes(id) ? p.filter(x => x !== id) : [...p, id]))
  const pageIds = (rows.data?.rows ?? []).map(r => r.id)
  const allPicked = pageIds.length > 0 && pageIds.every(id => picked.includes(id))
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('db.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('db.intro', { source: items.data?.source === 'postgres' ? 'Postgres' : t('db.builtin') })}</p></div>
      {!items.isLoading && !list.length ? <EmptyState title={t('db.empty')} /> :
      <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-4">
        <Card title={t('db.items')} className="lg:max-h-[calc(100vh-10rem)] overflow-auto">
          <ul className="space-y-1">{list.map(i => <li key={i.key}>
            <button type="button" onClick={() => { setKey(i.key); reset() }}
              className={`w-full text-left px-2 py-1.5 rounded text-sm ${i.key === key ? 'bg-blue-50 font-medium' : 'hover:bg-gray-50'}`}>
              <span className="block">{itemName(i, i18n.language)} {!i.tracked && <Badge tone="warn">{t('db.untracked')}</Badge>}</span>
              <span className="block text-xs text-[var(--muted)] tabular-nums">{t('db.rowsN', { n: i.rows.toLocaleString() })}{i.first ? ` · ${fmtDate(i.first)} ~ ${fmtDate(i.last)}` : ''}</span>
            </button></li>)}</ul>
        </Card>
        {item && <Card title={label}>
          <div className="flex flex-wrap items-center gap-2 text-sm mb-3">
            <select aria-label={t('common.season')} className={inputCls} value={range.season} onChange={e => { setRange({ season: e.target.value, from: '', to: '' }); reset() }}>
              <option value="">{t('db.allTime')}</option>
              {(seasons.data ?? []).map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}</option>)}</select>
            {!range.season && <>
              <input type="date" aria-label={t('db.from')} className={inputCls} value={range.from} onChange={e => { setRange(r => ({ ...r, from: e.target.value })); reset() }} />
              <span className="text-[var(--muted)]">~</span>
              <input type="date" aria-label={t('db.to')} className={inputCls} value={range.to} onChange={e => { setRange(r => ({ ...r, to: e.target.value })); reset() }} />
            </>}
            <span className="ml-auto text-[var(--muted)] tabular-nums">{t('db.matching', { n: total.toLocaleString() })}</span>
          </div>
          <div className="flex flex-wrap gap-2 mb-3">
            <Button size="sm" variant="ghost" disabled={!picked.length} loading={del.isPending}
              onClick={() => remove({ key, ids: picked }, t('db.confirmPicked', { n: picked.length }))}><Trash2 size={14} />{t('db.deletePicked', { n: picked.length })}</Button>
            {filtered && <Button size="sm" variant="ghost" disabled={!total} loading={del.isPending}
              onClick={() => remove({ key, start: start || undefined, end: end || undefined }, t('db.confirmRange', { n: total.toLocaleString(), name: label }))}><Trash2 size={14} />{t('db.deleteRange', { n: total.toLocaleString() })}</Button>}
            <Button size="sm" variant="ghost" disabled={!item.rows} loading={del.isPending}
              onClick={() => remove({ key, all: true }, t('db.confirmAll', { n: item.rows.toLocaleString(), name: label }))}><Trash2 size={14} />{t('db.deleteAll', { n: item.rows.toLocaleString() })}</Button>
          </div>
          <p className="text-xs text-[var(--muted)] mb-2">{t('db.readOnly')}</p>
          <div className="overflow-x-auto"><table className="w-full text-sm whitespace-nowrap">
            <thead className="text-left text-[var(--muted)]"><tr>
              <th className="py-1 w-6"><input type="checkbox" aria-label={t('db.pickPage')} checked={allPicked}
                onChange={() => setPicked(p => (allPicked ? p.filter(id => !pageIds.includes(id)) : [...new Set([...p, ...pageIds])]))} /></th>
              <th>{t('db.time')}</th><th className="text-right">{t('db.unitPrice')}</th><th className="text-right">{t('db.qty')}</th>
              <th className="text-right">{t('db.price')}</th><th className="pl-3">{t('db.sold')}</th><th className="pl-3">{t('db.id')}</th></tr></thead>
            <tbody>{(rows.data?.rows ?? []).map(r => <tr key={r.id + r.created_at} className="border-t border-[var(--line)]">
              <td className="py-1"><input type="checkbox" aria-label={r.id} checked={picked.includes(r.id)} onChange={() => toggle(r.id)} /></td>
              <td className="tabular-nums">{fmtTsUtc(r.created_at)}</td><td className="text-right tabular-nums">{fmtNum(r.price_per_unit)}</td>
              <td className="text-right tabular-nums">{r.quantity}</td><td className="text-right tabular-nums">{fmtNum(r.price)}</td>
              <td className="pl-3">{r.has_sold ? t('common.yes') : t('common.no')}</td><td className="pl-3 font-mono text-xs text-[var(--muted)]">{r.id}</td></tr>)}</tbody>
          </table></div>
          {total > PAGE && <div className="flex items-center justify-end gap-2 mt-2 text-sm">
            <span className="text-[var(--muted)]">{page} / {Math.ceil(total / PAGE)}</span>
            <Button variant="ghost" size="sm" aria-label={t('common.prev')} disabled={page <= 1} onClick={() => setPage(p => p - 1)}>‹</Button>
            <Button variant="ghost" size="sm" aria-label={t('common.next')} disabled={page * PAGE >= total} onClick={() => setPage(p => p + 1)}>›</Button></div>}
        </Card>}
      </div>}
    </div>
  )
}
