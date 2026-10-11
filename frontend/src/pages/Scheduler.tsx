import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useScheduleEstimate, useScheduler, useSchedulerAction, useSetup } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { JobList } from '../components/JobList'
import { ProgressPanel } from '../components/ProgressPanel'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'
import { fmtCount, taskRunsPerMonth } from '../lib/quota'
import { fmtInterval } from './scheduler/ItemScheduleEditor'
import { ScheduleList } from './scheduler/ScheduleList'
import { DefaultInterval } from './scheduler/DefaultInterval'

/** `embedded`: a section of the collection page (its intro only; the page carries the title). */
export default function Scheduler({ embedded = false }: { embedded?: boolean } = {}) {
  const { t } = useTranslation()
  const setup = useSetup()
  const estimate = useScheduleEstimate()
  const defaultSeconds = setup.data?.collect_interval_seconds ?? 3600
  const sc = useScheduler(); const act = useSchedulerAction(); const toast = useToast()
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  const onErr = (e: unknown) => toast.push((e as Error).message, 'error')
  return (
    <div className="space-y-4">
      {embedded ? <p className="text-sm text-[var(--muted)] max-w-3xl">{t('scheduler.intro')}</p>
        : <div><h1 className="text-xl font-semibold">{t('scheduler.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('scheduler.intro')}</p></div>}
      <Card title={t('scheduler.status')} actions={<>
        <Button size="sm" loading={act.isPending} disabled={sc.data?.running} onClick={() => act.mutate('start', { onError: onErr })}>{t('scheduler.startBtn')}</Button>
        <Button size="sm" variant="danger" loading={act.isPending} disabled={!sc.data?.running} onClick={() => act.mutate('stop', { onError: onErr })}>{t('scheduler.stopBtn')}</Button></>}>
        <div className="space-y-3">
          <Badge tone={sc.data?.running ? 'ok' : 'muted'}>{sc.data?.running ? t('scheduler.on') : t('scheduler.off')}</Badge>
          {setup.data && <DefaultInterval key={setup.data.collect_interval_seconds} seconds={setup.data.collect_interval_seconds} running={!!sc.data?.running} monthlyQuota={setup.data.monthly_quota} />}
        </div>
      </Card>
      <Card>
        <p className="text-xs text-[var(--muted)] mb-2">{t('scheduler.perItemHint', { v: fmtInterval(defaultSeconds, t) })}</p>
        {(() => {
          const perMonth = Math.max(estimate.data?.per_month ?? 0, (sc.data?.tasks ?? []).reduce((n, tk) => n + taskRunsPerMonth(tk.schedule_type, tk.schedule_value), 0))
          const quota = setup.data?.monthly_quota ?? 10000
          return <p className={`text-xs mb-2 ${perMonth > quota ? 'text-[var(--bad)] font-medium' : 'text-[var(--muted)]'}`}>{t('scheduler.month', { n: fmtCount(perMonth), quota: fmtCount(quota) })}{perMonth > quota ? ` ${t('scheduler.monthOver')}` : ''}</p>
        })()}
        <ScheduleList defaultSeconds={defaultSeconds} onJob={setJobId} />
      </Card>
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card title={t('common.jobs')}><JobList kinds={['collect', 'task']} selected={jobId} onSelect={setJobId} /></Card>
        <Card>{jobId ? <ProgressPanel kind="collect" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} /> : <p className="text-sm text-[var(--muted)]">{t('common.pickJob')}</p>}</Card>
      </div>
    </div>
  )
}
