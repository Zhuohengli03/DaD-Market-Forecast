import { Fragment, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItems, useItemsConfig, useScheduleEstimate, useScheduler, useSchedulerAction, useSetup, useStartCollect } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { JobList } from '../components/JobList'
import { ProgressPanel } from '../components/ProgressPanel'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useToast } from '../components/ui/Toast'
import { itemName } from '../lib/labels'
import { fmtCount, taskRunsPerMonth } from '../lib/quota'
import { fmtInterval, ItemScheduleEditor } from './scheduler/ItemScheduleEditor'

export default function Scheduler() {
  const { t, i18n } = useTranslation()
  const cfg = useItemsConfig(); const items = useItems(); const setup = useSetup()
  const [editing, setEditing] = useState<string | null>(null)
  const estimate = useScheduleEstimate()
  const defaultSeconds = setup.data?.collect_interval_seconds ?? 3600
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
        <p className="text-xs text-[var(--muted)] mb-2">{t('scheduler.perItemHint', { v: fmtInterval(defaultSeconds, t) })}</p>
        {(() => {
          const perMonth = Math.max(estimate.data?.per_month ?? 0, (sc.data?.tasks ?? []).reduce((n, tk) => n + taskRunsPerMonth(tk.schedule_type, tk.schedule_value), 0))
          const quota = setup.data?.monthly_quota ?? 10000
          return <p className={`text-xs mb-2 ${perMonth > quota ? 'text-[var(--bad)] font-medium' : 'text-[var(--muted)]'}`}>{t('scheduler.month', { n: fmtCount(perMonth), quota: fmtCount(quota) })}{perMonth > quota ? ` ${t('scheduler.monthOver')}` : ''}</p>
        })()}
        <table className="w-full text-sm"><thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('scheduler.task')}</th><th>{t('scheduler.schedule')}</th><th></th></tr></thead>
          <tbody>{(sc.data?.tasks ?? []).map(tk => {
            const name = tk.item ?? tk.name
            const info = items.data?.find(i => i.name === name)
            const entry = cfg.data?.items.find(i => i.name === name)
            const custom = !!entry?.schedule
            const when = tk.schedule_type === 'seconds' ? t('scheduler.everyText', { v: fmtInterval(Number(tk.schedule_value), t) }) : t('scheduler.dailyText', { v: tk.schedule_value })
            return (
              <Fragment key={tk.name}>
                <tr className="border-t border-[var(--line)]"><td className="py-1.5">{info ? itemName(info, i18n.language) : name}</td>
                  <td>{when}{!custom && <span className="ml-2 text-xs text-[var(--muted)]">{t('scheduler.isDefault')}</span>}
                    {estimate.data?.per_item[name] && <div className="text-xs text-[var(--muted)]">{t('scheduler.itemCost', { run: estimate.data.per_item[name].per_run, month: fmtCount(estimate.data.per_item[name].per_run * estimate.data.per_item[name].runs) })}</div>}</td>
                  <td className="text-right whitespace-nowrap">
                    <Button size="sm" variant="ghost" onClick={() => setEditing(e => (e === name ? null : name))}>{t('scheduler.edit')}</Button>
                    <Button size="sm" variant="ghost" onClick={() => collect.mutate(name, { onSuccess: setJobId, onError: onErr })}>{t('scheduler.runNow')}</Button></td></tr>
                {editing === name && <tr className="bg-gray-50"><td colSpan={3} className="px-3 py-3">
                  <ItemScheduleEditor name={name} schedule={entry?.schedule} defaultSeconds={defaultSeconds} onDone={() => setEditing(null)} /></td></tr>}
              </Fragment>)
          })}</tbody></table>
      </Card>
      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <Card title={t('common.jobs')}><JobList kinds={['collect', 'task']} selected={jobId} onSelect={setJobId} /></Card>
        <Card>{jobId ? <ProgressPanel kind="collect" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} /> : <p className="text-sm text-[var(--muted)]">{t('common.pickJob')}</p>}</Card>
      </div>
    </div>
  )
}
