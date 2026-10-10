import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronRight } from 'lucide-react'
import { useCollectAll, useItems, useOverview, useScheduleEstimate, useSetScheduled, useStartCollect } from '../../api/hooks'
import type { ItemInfo, OverviewGroup } from '../../api/types'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { useDebounced } from '../../lib/useDebounced'
import { fmtCount } from '../../lib/quota'
import { groupName, variantLabel } from '../dashboard/TypeCard'
import { ItemScheduleEditor, scheduleText } from './ItemScheduleEditor'

/** Checkbox that can also show "some selected" (for an item type whose qualities are partly ticked). */
function TriCheck({ checked, some, label, onChange }: { checked: boolean; some: boolean; label: string; onChange: (v: boolean) => void }) {
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => { if (ref.current) ref.current.indeterminate = some && !checked }, [some, checked])
  return <input ref={ref} type="checkbox" aria-label={label} checked={checked} onChange={e => onChange(e.target.checked)} className="h-4 w-4" />
}

interface Row { group: OverviewGroup; items: ItemInfo[] }

/** Tracked items grouped by type: tick what is collected on a schedule, and set when, per type or per quality. */
export function ScheduleList({ defaultSeconds, onJob }: { defaultSeconds: number; onJob: (id: string) => void }) {
  const { t, i18n } = useTranslation()
  const overview = useOverview(); const items = useItems(); const estimate = useScheduleEstimate()
  const setScheduled = useSetScheduled(); const collectOne = useStartCollect(); const collectSome = useCollectAll(); const toast = useToast()
  const [open, setOpen] = useState<Set<string>>(new Set())
  const [editing, setEditing] = useState<string | null>(null)
  const [q, setQ] = useState(''); const query = useDebounced(q, 200).trim().toLowerCase()
  const byName = useMemo(() => new Map((items.data ?? []).map(i => [i.name, i])), [items.data])
  const rows: Row[] = useMemo(() => (overview.data ?? []).map(group => ({ group, items: group.variants.map(v => byName.get(v.name)).filter(Boolean) as ItemInfo[] }))
    .filter(r => r.items.length)
    .filter(r => !query || [r.group.name, r.group.name_zh ?? ''].some(s => s.toLowerCase().includes(query))), [overview.data, byName, query])
  const on = (i: ItemInfo) => i.scheduled !== false
  const all = rows.flatMap(r => r.items)
  const ticked = all.filter(on).length
  const onErr = (e: unknown) => toast.push((e as Error).message, 'error')
  const tick = (names: string[], scheduled: boolean) => setScheduled.mutate({ names, scheduled }, { onError: onErr })
  const perMonth = (names: string[]) => names.reduce((n, name) => { const e = estimate.data?.per_item[name]; return n + (e ? e.runs * e.per_run : 0) }, 0)
  const toggleOpen = (k: string) => setOpen(s => { const n = new Set(s); if (n.has(k)) n.delete(k); else n.add(k); return n })
  const label = (r: Row, i: ItemInfo) => { const v = r.group.variants.find(x => x.name === i.name); return v ? variantLabel(v, i18n.language) : i.name }
  const whenOf = (list: ItemInfo[]) => {
    const texts = [...new Set(list.map(i => scheduleText(i.schedule, defaultSeconds, t)))]
    return texts.length === 1 ? texts[0] : t('scheduler.mixed')
  }
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <input className={`${inputCls} w-56`} placeholder={t('dashboard.search')} aria-label={t('dashboard.search')} value={q} onChange={e => setQ(e.target.value)} />
        <Button size="sm" variant="ghost" disabled={!all.length} onClick={() => tick(all.map(i => i.name), true)}>{t('scheduler.tickAll')}</Button>
        <Button size="sm" variant="ghost" disabled={!all.length} onClick={() => tick(all.map(i => i.name), false)}>{t('scheduler.tickNone')}</Button>
        <span className="ml-auto text-xs text-[var(--muted)]">{t('scheduler.tickedCount', { n: ticked, total: all.length })}</span>
      </div>
      <div className="border border-[var(--line)] rounded-lg divide-y divide-[var(--line)]">
        {rows.map(r => {
          const names = r.items.map(i => i.name)
          const nOn = r.items.filter(on).length
          const multi = r.items.length > 1
          const isOpen = open.has(r.group.key)
          const name = groupName(r.group, i18n.language)
          return (
            <Fragment key={r.group.key}>
              <div className="flex flex-wrap items-center gap-3 px-3 py-2 text-sm">
                <TriCheck checked={nOn === r.items.length} some={nOn > 0} label={t('scheduler.tickItem', { name })} onChange={v => tick(names, v)} />
                <button type="button" className="flex items-center gap-2 min-w-0 flex-1 text-left" onClick={() => multi && toggleOpen(r.group.key)} aria-expanded={multi ? isOpen : undefined}>
                  {multi ? <ChevronRight size={14} className={`shrink-0 transition ${isOpen ? 'rotate-90' : ''}`} /> : <span className="w-3.5" />}
                  {r.group.icon_url ? <img src={r.group.icon_url} alt="" loading="lazy" className="w-6 h-6 object-contain" /> : <span className="w-6" />}
                  <span className="font-medium truncate">{name}</span>
                  {multi && <span className="text-xs text-[var(--muted)]">{t('scheduler.qualitiesOn', { on: nOn, n: r.items.length })}</span>}
                </button>
                <span className={`text-xs ${nOn ? '' : 'text-[var(--muted)]'}`}>{nOn ? whenOf(r.items.filter(on)) : t('scheduler.notScheduled')}</span>
                <span className="text-xs text-[var(--muted)] w-40 text-right whitespace-nowrap">{nOn ? t('scheduler.monthShort', { n: fmtCount(perMonth(names)) }) : ''}</span>
                <span className="whitespace-nowrap">
                  <Button size="sm" variant="ghost" onClick={() => setEditing(e => (e === r.group.key ? null : r.group.key))}>{t('scheduler.edit')}</Button>
                  <Button size="sm" variant="ghost" onClick={() => (multi ? collectSome.mutate(names, { onSuccess: onJob, onError: onErr }) : collectOne.mutate(names[0], { onSuccess: onJob, onError: onErr }))}>{t('scheduler.runNow')}</Button>
                </span>
              </div>
              {editing === r.group.key && <div className="bg-gray-50 px-3 py-3">
                {multi && <p className="text-xs text-[var(--muted)] mb-2">{t('scheduler.applyToAll', { n: r.items.length })}</p>}
                <ItemScheduleEditor name={names[0]} names={names} schedule={r.items[0].schedule} defaultSeconds={defaultSeconds} onDone={() => setEditing(null)} />
              </div>}
              {multi && isOpen && r.items.map(i => (
                <Fragment key={i.name}>
                  <div className="flex flex-wrap items-center gap-3 pl-12 pr-3 py-1.5 text-sm bg-gray-50/60">
                    <input type="checkbox" className="h-4 w-4" aria-label={t('scheduler.tickItem', { name: label(r, i) })} checked={on(i)} onChange={e => tick([i.name], e.target.checked)} />
                    <span className="flex-1">{label(r, i)}</span>
                    <span className={`text-xs ${on(i) ? '' : 'text-[var(--muted)]'}`}>{on(i) ? scheduleText(i.schedule, defaultSeconds, t) : t('scheduler.notScheduled')}{on(i) && !i.schedule && <span className="ml-1 text-[var(--muted)]">{t('scheduler.isDefault')}</span>}</span>
                    <span className="text-xs text-[var(--muted)] w-40 text-right whitespace-nowrap">{on(i) ? t('scheduler.monthShort', { n: fmtCount(perMonth([i.name])) }) : ''}</span>
                    <span className="whitespace-nowrap">
                      <Button size="sm" variant="ghost" onClick={() => setEditing(e => (e === i.name ? null : i.name))}>{t('scheduler.edit')}</Button>
                      <Button size="sm" variant="ghost" onClick={() => collectOne.mutate(i.name, { onSuccess: onJob, onError: onErr })}>{t('scheduler.runNow')}</Button>
                    </span>
                  </div>
                  {editing === i.name && <div className="bg-gray-50 pl-12 pr-3 py-3"><ItemScheduleEditor name={i.name} schedule={i.schedule} defaultSeconds={defaultSeconds} onDone={() => setEditing(null)} /></div>}
                </Fragment>))}
            </Fragment>)
        })}
        {!rows.length && <p className="px-3 py-4 text-sm text-[var(--muted)]">{overview.isLoading ? t('common.loading') : t('dashboard.noMatch')}</p>}
      </div>
    </div>
  )
}
