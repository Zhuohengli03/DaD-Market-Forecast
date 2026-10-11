import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Minus, Plus, X } from 'lucide-react'
import { useCollectAll, useCollectAllEstimate, useItems, useQuota, useSeasons, useSetup, useStartCollect, useStatus, useUntrackItem } from '../api/hooks'
import { Badge } from '../components/ui/Badge'
import { fmtCount } from '../lib/quota'
import { useJobStream } from '../api/jobStream'
import { JobList } from '../components/JobList'
import { ProgressPanel } from '../components/ProgressPanel'
import { BatchProgress } from '../components/wizard/InitProgress'
import { AddItemsDialog } from './collect/AddItemsDialog'
import { RemoveItemsDialog } from './collect/RemoveItemsDialog'
import { Notice } from '../components/Notice'
import { Button } from '../components/ui/Button'
import { inputCls } from '../components/ui/Field'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'
import { fmtDate, fmtRelative } from '../lib/format'
import { itemName, seasonLabel } from '../lib/labels'
import { useUrlState } from '../lib/urlState'

const FILTERS = ['all', 'nodata', 'data'] as const
type Filter = typeof FILTERS[number]

/** `embedded`: a section of the collection page (its intro only; the page carries the title). */
export default function Collect({ embedded = false }: { embedded?: boolean } = {}) {
  const { t, i18n } = useTranslation()
  const items = useItems(); const status = useStatus(); const setup = useSetup()
  const start = useStartCollect(); const untrack = useUntrackItem(); const toast = useToast()
  const seasons = useSeasons()
  const latestId = [...(seasons.data ?? [])].sort((a, b) => a.start.localeCompare(b.start)).at(-1)?.id ?? ''
  const [seasonParam, setSeason] = useUrlState('season', '')
  const seasonId = seasonParam || latestId
  const collectAll = useCollectAll(); const quota = useQuota(); const estimate = useCollectAllEstimate(!!items.data?.length && !!seasonId, seasonId || undefined)
  const est = estimate.data
  const [filterRaw, setFilter] = useUrlState('filter', 'all')
  const filter: Filter = (FILTERS as readonly string[]).includes(filterRaw) ? filterRaw as Filter : 'all'
  const all = items.data ?? []
  const counts = { all: all.length, nodata: all.filter(i => !i.rows).length, data: all.filter(i => i.rows > 0).length }
  const shown = filter === 'nodata' ? all.filter(i => !i.rows) : filter === 'data' ? all.filter(i => i.rows > 0) : all
  // the request estimate for exactly the items on screen
  const shownCost = est ? shown.reduce((n, i) => n + (est.per_item[i.name] ?? 0), 0) : null
  const [jobId, setJobId] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [removing, setRemoving] = useState(false)
  const stream = useJobStream(jobId)
  const batch = Array.isArray(stream.progress?.detail?.items)
  const runningFor = new Set((status.data?.jobs_running ?? []).filter(j => j.kind === 'collect').map(j => j.meta.item))
  const onErr = (e: unknown) => toast.push((e as Error).message, 'error')
  return (
    <div className="space-y-4">
      {embedded ? <p className="text-sm text-[var(--muted)] max-w-3xl">{t('collect.intro')}</p>
        : <div><h1 className="text-xl font-semibold">{t('collect.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('collect.intro')}</p></div>}
      {setup.data && !setup.data.api_key_configured && <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noKey')}</Notice>}
      {items.isSuccess && !items.data.length && <Notice to="/catalog" action={t('nav.catalog')}>{t('onboarding.noItems')}</Notice>}
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={() => setAdding(true)}><Plus size={14} />{t('collect.add')}</Button>
        <Button variant="ghost" disabled={!items.data?.length} onClick={() => setRemoving(true)}><Minus size={14} />{t('collect.remove')}</Button>
        <label className="flex items-center gap-2 text-sm">{t('common.season')}
          <select aria-label={t('common.season')} className={inputCls} value={seasonId} onChange={e => setSeason(e.target.value === latestId ? '' : e.target.value)}>
            {[...(seasons.data ?? [])].reverse().map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}{s.id === latestId ? t('collect.currentSeason') : ''}</option>)}</select></label>
        <Button variant={filter === 'all' ? 'ghost' : 'primary'} disabled={!shown.length || !!est?.opens} loading={collectAll.isPending || runningFor.has('*')}
          onClick={() => {
            const n = shownCost ?? shown.length; const left = quota.data?.credits_remaining
            const ask = left != null && n > left ? 'collect.confirmOver' : filter === 'all' ? 'collect.confirmAll' : 'collect.confirmSome'
            if (!window.confirm(t(ask, { n: fmtCount(n), left: fmtCount(left), items: shown.length }))) return
            collectAll.mutate({ names: filter === 'all' ? undefined : shown.map(i => i.name), seasonId }, { onSuccess: id => { setJobId(id); toast.push(t('collect.started')) }, onError: onErr })
          }}>{filter === 'all' ? t('collect.all', { n: shown.length }) : t('collect.these', { n: shown.length })}</Button>
        <span className="text-xs text-[var(--muted)]">{est?.opens ? t('collect.marketClosed', { date: est.opens }) : est?.since ? `${t('collect.window', { from: fmtDate(est.since), to: est.until ? fmtDate(est.until) : t('market.now') })} · ` : ''}{est?.opens ? '' : shownCost != null ? t('collect.shownCost', { n: fmtCount(shownCost), left: fmtCount(quota.data?.credits_remaining) }) : t('cost.estimating')}</span>
      </div>
      <div role="tablist" aria-label={t('collect.filter')} className="flex flex-wrap gap-1.5 text-sm">
        {FILTERS.map(f => (
          <button key={f} type="button" role="tab" aria-selected={filter === f} onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded-full border ${filter === f ? 'bg-[var(--accent)] text-white border-[var(--accent)]' : 'border-[var(--line)] text-[var(--muted)] hover:bg-gray-50'}`}>
            {t(`collect.filters.${f}`, { n: counts[f] })}</button>))}
        {filter === 'nodata' && counts.nodata > 0 && <span className="self-center text-xs text-[var(--muted)]">{t('collect.nodataHint')}</span>}
      </div>
      {adding && <AddItemsDialog onClose={() => setAdding(false)} />}
      {removing && <RemoveItemsDialog onClose={() => setRemoving(false)} />}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {shown.map(i => (
          <Card key={i.name}>
            <div className="flex items-start justify-between gap-2">
              <div className="flex gap-3">
                {i.icon_url ? <img src={i.icon_url} alt="" loading="lazy" className="w-10 h-10 object-contain shrink-0 rounded bg-gray-50" onError={e => { (e.target as HTMLImageElement).style.visibility = 'hidden' }} /> : <div className="w-10 h-10 shrink-0 rounded bg-gray-100" />}
                <div><div className="font-semibold">{itemName(i, i18n.language)}</div>
                  <div className="text-xs text-[var(--muted)]">{t('collect.seasonRows', { n: i.season_rows ?? 0 })} · {t('collect.allRows', { n: i.rows })}</div>
                  <div className="text-xs text-[var(--muted)] mb-3">{t('collect.lastAt', { when: fmtRelative(i.last_ts) })}{status.data?.collecting?.[i.name] && <span className="ml-2"><Badge tone={status.data.collecting[i.name] === 'collecting' ? 'warn' : 'muted'}>{t(`running.state.${status.data.collecting[i.name]}`)}</Badge></span>}</div></div>
              </div>
              <button type="button" title={t('collect.untrack')} aria-label={`${t('collect.untrack')} ${i.name}`} className="text-[var(--muted)] hover:text-[var(--bad)]"
                onClick={() => untrack.mutate(i.name, { onSuccess: () => toast.push(t('collect.untracked', { name: i.name })), onError: onErr })}><X size={16} /></button>
            </div>
            {est?.per_item[i.name] != null && <div className="text-xs text-[var(--muted)] mb-1">{t('collect.itemCost', { n: est.per_item[i.name] })}</div>}
            <Button size="sm" loading={runningFor.has(i.name) || !!status.data?.collecting?.[i.name]} disabled={!!est?.opens} onClick={() => start.mutate({ item: i.name, seasonId }, { onSuccess: id => { setJobId(id); toast.push(t('collect.started')) }, onError: onErr })}>{t('collect.collectNow')}</Button>
          </Card>))}
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card title={t('collect.recent')}><JobList kinds={['collect']} selected={jobId} onSelect={setJobId} /></Card>
        <Card>{jobId && batch ? <BatchProgress status={stream.status} progress={stream.progress} error={stream.error} />
          : jobId ? <ProgressPanel kind="collect" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} /> : <p className="text-sm text-[var(--muted)]">{t('common.pickJob')}</p>}</Card>
      </div>
    </div>
  )
}
