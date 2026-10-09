import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItems, useStartCollect, useStatus } from '../api/hooks'
import { useJobEvents } from '../api/sse'
import { JobList } from '../components/JobList'
import { LogPanel } from '../components/LogPanel'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'
import { fmtRelative } from '../lib/format'

export default function Collect() {
  const { t } = useTranslation()
  const items = useItems(); const status = useStatus(); const start = useStartCollect(); const toast = useToast()
  const [jobId, setJobId] = useState<string | null>(null)
  const events = useJobEvents(jobId)
  const runningFor = new Set((status.data?.jobs_running ?? []).filter(j => j.kind === 'collect').map(j => j.meta.item))
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">{t('collect.title')}</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {(items.data ?? []).map(i => (
          <Card key={i.name}><div className="font-semibold">{i.name}</div><div className="text-xs text-[var(--muted)] mb-3">{i.rows} · {fmtRelative(i.last_ts)}</div>
            <Button size="sm" loading={runningFor.has(i.name)} onClick={() => start.mutate(i.name, { onSuccess: id => { setJobId(id); toast.push(t('collect.started')) }, onError: e => toast.push((e as Error).message, 'error') })}>{t('collect.collectNow')}</Button></Card>))}
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card title={t('collect.recent')}><JobList kinds={['collect']} selected={jobId} onSelect={setJobId} /></Card>
        <Card>{jobId ? <LogPanel lines={events.lines} status={events.status} disconnected={events.disconnected} /> : <p className="text-sm text-[var(--muted)]">{t('common.log')}: –</p>}</Card>
      </div>
    </div>
  )
}
