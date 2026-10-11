import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Activity } from 'lucide-react'
import { useModels, useRuns, useStartForecast } from '../../api/hooks'
import { useJobStream } from '../../api/jobStream'
import type { Resolution } from '../../api/types'
import { ProgressPanel } from '../../components/ProgressPanel'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { Field, inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { fmtTs } from '../../lib/format'
import { modelHelp, modelLabel, seasonLabel } from '../../lib/labels'
import { useUrlPatch, useUrlState } from '../../lib/urlState'
import { QUICK_FORECAST } from '../market/quickForecast'
import { RunResult } from '../forecast/RunResult'

const BASELINES = ['naive', 'seasonal_naive_24']
const HORIZONS = [24, 48, 72, 168]

/** One-click forecast for this item, its settings, its progress and its results, all in one card. */
export function ItemForecast({ item, season }: { item: string; season: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient(); const toast = useToast()
  const runs = (useRuns().data ?? []).filter(r => r.item === item)
  const models = useModels(); const start = useStartForecast()
  const [jobParam] = useUrlState('job', '')
  const [runParam] = useUrlState('run', '')
  const patch = useUrlPatch()
  const stream = useJobStream(jobParam || null)
  const [horizon, setHorizon] = useState(QUICK_FORECAST.horizon_hours)
  const [coverage, setCoverage] = useState(QUICK_FORECAST.coverage)
  const [resolution, setResolution] = useState<Resolution>('1h')
  const [originStep, setOriginStep] = useState(0)   // 0: automatic, a quarter of the horizon (at least 6h)
  const [chosen, setChosen] = useState<string[]>(QUICK_FORECAST.models)
  const optional = (models.data ?? []).filter(m => !BASELINES.includes(m))
  useEffect(() => {
    if (stream.status === 'succeeded' && stream.resultRef) { qc.invalidateQueries({ queryKey: ['runs'] }); patch({ run: stream.resultRef, job: '' }) }
  }, [stream.status, stream.resultRef, qc, patch])
  const shown = runs.some(r => r.run_id === runParam) ? runParam : runs[0]?.run_id ?? ''
  const running = stream.status === 'running' || stream.status === 'queued'
  const go = () => start.mutate({ item, season_id: season, horizon_hours: horizon, models: chosen.filter(m => optional.includes(m)), coverage,
    backtest_origin_step: originStep || undefined, resolution },
    { onSuccess: id => patch({ job: id, run: '' }), onError: e => toast.push((e as Error).message, 'error') })
  const toggle = (m: string) => setChosen(c => (c.includes(m) ? c.filter(x => x !== m) : [...c, m]))
  return (
    <Card title={t('item.forecast')} actions={<Button loading={start.isPending || running} disabled={!season} onClick={go}><Activity size={14} />{t('market.forecastNow')}</Button>}>
      <div className="space-y-3">
        <details className="text-sm">
          <summary className="cursor-pointer text-[var(--muted)]">{t('item.forecastSettings', { h: horizon, c: Math.round(coverage * 100) })}</summary>
          <div className="mt-3 grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Field label={t('forecast.horizon')}><select aria-label={t('forecast.horizon')} className={inputCls} value={horizon} onChange={e => setHorizon(Number(e.target.value))}>{HORIZONS.map(h => <option key={h} value={h}>{h} {t('common.hours')}</option>)}</select></Field>
            <Field label={t('forecast.coverage')} hint={t('forecast.coverageHint')}><select aria-label={t('forecast.coverage')} className={inputCls} value={coverage} onChange={e => setCoverage(Number(e.target.value))}>{[0.8, 0.9, 0.95].map(c => <option key={c} value={c}>{Math.round(c * 100)}%</option>)}</select></Field>
            <Field label={t('forecast.resolution')} hint={t('forecast.resolutionHint')}><select aria-label={t('forecast.resolution')} className={inputCls} value={resolution} onChange={e => setResolution(e.target.value as Resolution)}>{(['1h', '4h', '1d'] as const).map(r => <option key={r} value={r}>{t(`forecast.resolutions.${r}`)}</option>)}</select></Field>
            <Field label={t('forecast.originStep')} hint={t('forecast.originStepHint')}><select aria-label={t('forecast.originStep')} className={inputCls} value={originStep} onChange={e => setOriginStep(Number(e.target.value))}><option value={0}>{t('forecast.originAuto')}</option>{[3, 6, 12, 24].map(v => <option key={v} value={v}>{v} {t('common.hours')}</option>)}</select></Field>
          </div>
          <Field label={t('forecast.models')} hint={t('forecast.baselineHint')}>
            <div className="flex flex-wrap gap-3">
              {BASELINES.map(m => <label key={m} title={modelHelp(m, t)} className="flex items-center gap-1 text-[var(--muted)]"><input type="checkbox" checked disabled readOnly aria-label={modelLabel(m, t)} />{modelLabel(m, t)}</label>)}
              {optional.map(m => <label key={m} title={modelHelp(m, t)} className="flex items-center gap-1"><input type="checkbox" checked={chosen.includes(m)} onChange={() => toggle(m)} aria-label={modelLabel(m, t)} />{modelLabel(m, t)}</label>)}
            </div>
          </Field>
        </details>
        {jobParam && <ProgressPanel kind="forecast" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} />}
        {runs.length > 0 && <label className="flex items-center gap-2 text-sm"><span className="text-[var(--muted)]">{t('item.runHistory')}</span>
          <select aria-label={t('item.runHistory')} className={inputCls} value={shown} onChange={e => patch({ run: e.target.value })}>
            {runs.map(r => <option key={r.run_id} value={r.run_id}>{fmtTs(r.created_at)} · {seasonLabel(r.season_name, t)} · {r.horizon_hours} {t('common.hours')}</option>)}</select></label>}
        {shown ? <RunResult runId={shown} onDeleted={() => patch({ run: '' })} /> : !jobParam && <p className="text-sm text-[var(--muted)]">{t('item.noRun')}</p>}
      </div>
    </Card>
  )
}
