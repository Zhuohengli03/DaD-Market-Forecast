import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { useItems, useSetup, useStartCollect, useStatus, useUntrackItem } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { JobList } from '../components/JobList'
import { LogPanel } from '../components/LogPanel'
import { Notice } from '../components/Notice'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'
import { fmtRelative } from '../lib/format'

export default function Collect() {
  const { t } = useTranslation()
  const items = useItems(); const status = useStatus(); const setup = useSetup()
  const start = useStartCollect(); const untrack = useUntrackItem(); const toast = useToast()
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  const runningFor = new Set((status.data?.jobs_running ?? []).filter(j => j.kind === 'collect').map(j => j.meta.item))
  const onErr = (e: unknown) => toast.push((e as Error).message, 'error')
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('collect.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('collect.intro')}</p></div>
      {setup.data && !setup.data.api_key_configured && <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noKey')}</Notice>}
      {items.isSuccess && !items.data.length && <Notice to="/catalog" action={t('nav.catalog')}>{t('onboarding.noItems')}</Notice>}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {(items.data ?? []).map(i => (
          <Card key={i.name}>
            <div className="flex items-start justify-between gap-2">
              <div><div className="font-semibold">{i.name}</div><div className="text-xs text-[var(--muted)] mb-3">{i.rows} {t('dashboard.rows')} · {fmtRelative(i.last_ts)}</div></div>
              <button type="button" title={t('collect.untrack')} aria-label={`${t('collect.untrack')} ${i.name}`} className="text-[var(--muted)] hover:text-[var(--bad)]"
                onClick={() => untrack.mutate(i.name, { onSuccess: () => toast.push(t('collect.untracked', { name: i.name })), onError: onErr })}><X size={16} /></button>
            </div>
            <Button size="sm" loading={runningFor.has(i.name)} onClick={() => start.mutate(i.name, { onSuccess: id => { setJobId(id); toast.push(t('collect.started')) }, onError: onErr })}>{t('collect.collectNow')}</Button>
          </Card>))}
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card title={t('collect.recent')}><JobList kinds={['collect']} selected={jobId} onSelect={setJobId} /></Card>
        <Card>{jobId ? <LogPanel lines={stream.lines} status={stream.status} disconnected={stream.disconnected} /> : <p className="text-sm text-[var(--muted)]">{t('common.log')}: –</p>}</Card>
      </div>
    </div>
  )
}
