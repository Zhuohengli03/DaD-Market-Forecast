import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQueryClient } from '@tanstack/react-query'
import { useSeasons } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { ProgressPanel } from '../components/ProgressPanel'
import { Notice } from '../components/Notice'
import { Card } from '../components/ui/Card'
import { useUrlState } from '../lib/urlState'
import { ForecastForm } from './forecast/ForecastForm'
import { RunList } from './forecast/RunList'
import { RunResult } from './forecast/RunResult'

export default function Forecast() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const seasons = useSeasons()
  const [jobId, setJobId] = useState<string | null>(null)
  const [runId, setRunId] = useUrlState('run', '')
  const stream = useJobStream(jobId)
  useEffect(() => {
    if (stream.status === 'succeeded' && stream.resultRef) { qc.invalidateQueries({ queryKey: ['runs'] }); setRunId(stream.resultRef); setJobId(null) }
  }, [stream.status, stream.resultRef, qc, setRunId])
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[280px_1fr] gap-4">
      <Card title={t('forecast.history')} className="xl:h-[calc(100vh-6rem)] overflow-auto"><RunList selected={runId || null} onSelect={setRunId} /></Card>
      <div className="space-y-4">
        <div><h1 className="text-xl font-semibold">{t('forecast.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('forecast.intro')}</p></div>
        {seasons.isSuccess && !seasons.data.length ? <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noSeasons')}</Notice> :
          <Card><ForecastForm onStarted={id => { setJobId(id); setRunId('') }} /></Card>}
        {jobId && <Card><ProgressPanel kind="forecast" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} /></Card>}
        {runId && <RunResult runId={runId} onDeleted={() => setRunId('')} />}
      </div>
    </div>
  )
}
