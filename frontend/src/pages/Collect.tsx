import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Plus, X } from 'lucide-react'
import { useCollectAll, useItems, useQuota, useSetup, useStartCollect, useStatus, useUntrackItem } from '../api/hooks'
import { Badge } from '../components/ui/Badge'
import { fmtCount } from '../lib/quota'
import { useJobStream } from '../api/jobStream'
import { JobList } from '../components/JobList'
import { ProgressPanel } from '../components/ProgressPanel'
import { BatchProgress } from '../components/wizard/InitProgress'
import { AddItemsDialog } from './collect/AddItemsDialog'
import { Notice } from '../components/Notice'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'
import { fmtRelative } from '../lib/format'
import { itemName } from '../lib/labels'

export default function Collect() {
  const { t, i18n } = useTranslation()
  const items = useItems(); const status = useStatus(); const setup = useSetup()
  const start = useStartCollect(); const untrack = useUntrackItem(); const toast = useToast()
  const collectAll = useCollectAll(); const quota = useQuota()
  const [jobId, setJobId] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const stream = useJobStream(jobId)
  const batch = Array.isArray(stream.progress?.detail?.items)
  const runningFor = new Set((status.data?.jobs_running ?? []).filter(j => j.kind === 'collect').map(j => j.meta.item))
  const onErr = (e: unknown) => toast.push((e as Error).message, 'error')
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('collect.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('collect.intro')}</p></div>
      {setup.data && !setup.data.api_key_configured && <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noKey')}</Notice>}
      {items.isSuccess && !items.data.length && <Notice to="/catalog" action={t('nav.catalog')}>{t('onboarding.noItems')}</Notice>}
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={() => setAdding(true)}><Plus size={14} />{t('collect.add')}</Button>
        <Button variant="ghost" disabled={!items.data?.length} loading={collectAll.isPending || runningFor.has('*')}
          onClick={() => {
            const n = items.data?.length ?? 0; const left = quota.data?.credits_remaining
            if (left != null && n > left && !window.confirm(t('collect.confirmOver', { n, left }))) return
            collectAll.mutate(undefined, { onSuccess: id => { setJobId(id); toast.push(t('collect.started')) }, onError: onErr })
          }}>{t('collect.all', { n: items.data?.length ?? 0 })}</Button>
        <span className="text-xs text-[var(--muted)]">{t('collect.trackedCount', { n: items.data?.length ?? 0 })} · {t('collect.cost', { n: items.data?.length ?? 0, left: fmtCount(quota.data?.credits_remaining) })}</span>
      </div>
      {adding && <AddItemsDialog onClose={() => setAdding(false)} />}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {(items.data ?? []).map(i => (
          <Card key={i.name}>
            <div className="flex items-start justify-between gap-2">
              <div className="flex gap-3">
                {i.icon_url ? <img src={i.icon_url} alt="" width={40} height={40} loading="lazy" className="shrink-0 rounded bg-gray-50" onError={e => { (e.target as HTMLImageElement).style.visibility = 'hidden' }} /> : <div className="w-10 h-10 shrink-0 rounded bg-gray-100" />}
                <div><div className="font-semibold">{itemName(i, i18n.language)}</div>
                  <div className="text-xs text-[var(--muted)]">{t('collect.seasonRows', { n: i.season_rows ?? 0 })} · {t('collect.allRows', { n: i.rows })}</div>
                  <div className="text-xs text-[var(--muted)] mb-3">{t('collect.lastAt', { when: fmtRelative(i.last_ts) })}{status.data?.collecting?.[i.name] && <span className="ml-2"><Badge tone={status.data.collecting[i.name] === 'collecting' ? 'warn' : 'muted'}>{t(`running.state.${status.data.collecting[i.name]}`)}</Badge></span>}</div></div>
              </div>
              <button type="button" title={t('collect.untrack')} aria-label={`${t('collect.untrack')} ${i.name}`} className="text-[var(--muted)] hover:text-[var(--bad)]"
                onClick={() => untrack.mutate(i.name, { onSuccess: () => toast.push(t('collect.untracked', { name: i.name })), onError: onErr })}><X size={16} /></button>
            </div>
            <Button size="sm" loading={runningFor.has(i.name) || !!status.data?.collecting?.[i.name]} onClick={() => start.mutate(i.name, { onSuccess: id => { setJobId(id); toast.push(t('collect.started')) }, onError: onErr })}>{t('collect.collectNow')}</Button>
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
