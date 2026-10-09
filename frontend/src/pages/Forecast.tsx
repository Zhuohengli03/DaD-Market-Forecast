import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQueryClient } from '@tanstack/react-query'
import { useJobEvents } from '../api/sse'
import { useJob } from '../api/hooks'
import { LogPanel } from '../components/LogPanel'
import { Card } from '../components/ui/Card'
import { useUrlState } from '../lib/urlState'
import { ForecastForm } from './forecast/ForecastForm'
import { RunList } from './forecast/RunList'
import { RunResult } from './forecast/RunResult'

export default function Forecast() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const [jobId, setJobId] = useState<string | null>(null)
  const [runId, setRunId] = useUrlState('run', '')
  const events = useJobEvents(jobId)
  const job = useJob(events.status === 'succeeded' || events.status === 'failed' ? jobId : null)
  useEffect(() => {
    if (job.data?.status === 'succeeded' && job.data.result_ref) { qc.invalidateQueries({ queryKey: ['runs'] }); setRunId(job.data.result_ref); setJobId(null) }
  }, [job.data, qc, setRunId])
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[280px_1fr] gap-4">
      <Card title={t('forecast.history')} className="xl:h-[calc(100vh-6rem)] overflow-auto"><RunList selected={runId || null} onSelect={setRunId} /></Card>
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">{t('forecast.title')}</h1>
        <Card><ForecastForm onStarted={id => { setJobId(id); setRunId('') }} /></Card>
        {jobId && <Card>{events.status === 'failed' && job.data?.error && <div className="text-sm text-[var(--bad)] mb-2">{t('forecast.failed')}: {job.data.error}</div>}<LogPanel lines={events.lines} status={events.status} disconnected={events.disconnected} /></Card>}
        {runId && <RunResult runId={runId} onDeleted={() => setRunId('')} />}
      </div>
    </div>
  )
}
