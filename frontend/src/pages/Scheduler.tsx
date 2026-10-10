import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useScheduler, useSchedulerAction, useStartCollect } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { JobList } from '../components/JobList'
import { ProgressPanel } from '../components/ProgressPanel'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'

export default function Scheduler() {
  const { t } = useTranslation()
  const sc = useScheduler(); const act = useSchedulerAction(); const collect = useStartCollect(); const toast = useToast()
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  const onErr = (e: unknown) => toast.push((e as Error).message, 'error')
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('scheduler.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('scheduler.intro')}</p></div>
      <Card title={t('scheduler.status')} actions={<>
        <Button size="sm" loading={act.isPending} disabled={sc.data?.running} onClick={() => act.mutate('start', { onError: onErr })}>{t('scheduler.startBtn')}</Button>
        <Button size="sm" variant="danger" loading={act.isPending} disabled={!sc.data?.running} onClick={() => act.mutate('stop', { onError: onErr })}>{t('scheduler.stopBtn')}</Button></>}>
        <Badge tone={sc.data?.running ? 'ok' : 'muted'}>{sc.data?.running ? t('scheduler.on') : t('scheduler.off')}</Badge>
      </Card>
      <Card>
        <table className="w-full text-sm"><thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('scheduler.task')}</th><th>{t('scheduler.schedule')}</th><th></th></tr></thead>
          <tbody>{(sc.data?.tasks ?? []).map(tk => (
            <tr key={tk.name} className="border-t border-[var(--line)]"><td className="py-1.5">{tk.item ?? tk.name}</td><td>{tk.schedule_value}</td>
              <td className="text-right"><Button size="sm" variant="ghost" onClick={() => collect.mutate(tk.item ?? tk.name, { onSuccess: setJobId, onError: onErr })}>{t('scheduler.runNow')}</Button></td></tr>))}</tbody></table>
      </Card>
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card title={t('common.jobs')}><JobList kinds={['collect', 'task']} selected={jobId} onSelect={setJobId} /></Card>
        <Card>{jobId ? <ProgressPanel kind="collect" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} /> : <p className="text-sm text-[var(--muted)]">{t('common.pickJob')}</p>}</Card>
      </div>
    </div>
  )
}
