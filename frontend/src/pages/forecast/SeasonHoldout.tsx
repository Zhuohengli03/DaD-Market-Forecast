import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Trash2 } from 'lucide-react'
import { useCollectEstimate, useDeleteHoldout, useHoldout, useHoldouts, useItems, useSeasons, useStartHoldout } from '../../api/hooks'
import { useJobStream } from '../../api/jobStream'
import { ProgressPanel } from '../../components/ProgressPanel'
import { Button } from '../../components/ui/Button'
import { Field, inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { fmtDate, fmtPct } from '../../lib/format'
import { itemName, modelLabel, seasonLabel } from '../../lib/labels'
import { useUrlState } from '../../lib/urlState'
import { RefreshData } from '../market/RefreshData'
import { HoldoutResult } from './HoldoutResult'

const TRAIN_FRACS = [0.6, 0.7, 0.8]
const RESOLUTIONS = ['1h', '4h', '1d'] as const

/** Season holdout: how well would the forecaster have done on a season it had not seen the end of? */
export function SeasonHoldout() {
  const { t, i18n } = useTranslation()
  const qc = useQueryClient(); const toast = useToast()
  const seasons = useSeasons(); const items = useItems()
  const ended = (seasons.data ?? []).filter(s => s.end && new Date(s.end).getTime() <= Date.now())
    .sort((a, b) => (a.start < b.start ? 1 : -1))
  const [item, setItem] = useState('')
  const [seasonId, setSeasonId] = useState('')
  const [trainFrac, setTrainFrac] = useState(0.7)
  const [resolution, setResolution] = useState<(typeof RESOLUTIONS)[number]>('1h')
  const [jobId, setJobId] = useState<string | null>(null)
  const [runId, setRunId] = useUrlState('holdout', '')
  const tracked = items.data ?? []
  const pickedItem = tracked.some(i => i.name === item) ? item : tracked.find(i => i.rows > 0)?.name ?? tracked[0]?.name ?? ''
  const season = ended.find(s => s.id === seasonId) ?? ended[0]
  const estimate = useCollectEstimate(pickedItem, season?.id)
  const progress = estimate.data?.progress
  const start = useStartHoldout(); const stream = useJobStream(jobId)
  const history = useHoldouts(); const run = useHoldout(runId || null); const del = useDeleteHoldout()
  useEffect(() => {
    if (stream.status === 'succeeded' && stream.resultRef) { qc.invalidateQueries({ queryKey: ['holdouts'] }); setRunId(stream.resultRef); setJobId(null) }
  }, [stream.status, stream.resultRef, qc, setRunId])
  if (!ended.length) return <p className="text-sm text-[var(--muted)]">{t('holdout.noEnded')}</p>
  const seasonName = seasonLabel(season.name, t)
  const submit = () => start.mutate({ item: pickedItem, season_id: season.id, train_frac: trainFrac, resolution },
    { onSuccess: setJobId, onError: e => toast.push((e as Error).message, 'error') })
  return (
    <div className="space-y-3">
      <p className="text-sm text-[var(--muted)] max-w-3xl">{t('holdout.intro', { pct: `${Math.round(trainFrac * 100)}%` })}</p>
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        <Field label={t('common.item')}><select aria-label={t('common.item')} className={inputCls} value={pickedItem} onChange={e => setItem(e.target.value)}>
          {tracked.map(i => <option key={i.name} value={i.name}>{itemName(i, i18n.language)}</option>)}</select></Field>
        <Field label={t('holdout.season')}><select aria-label={t('holdout.season')} className={inputCls} value={season.id} onChange={e => setSeasonId(e.target.value)}>
          {ended.map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}</option>)}</select></Field>
        <Field label={t('holdout.trainFrac')}><select aria-label={t('holdout.trainFrac')} className={inputCls} value={trainFrac} onChange={e => setTrainFrac(Number(e.target.value))}>
          {TRAIN_FRACS.map(f => <option key={f} value={f}>{Math.round(f * 100)}%</option>)}</select></Field>
        <Field label={t('forecast.resolution')} hint={t('forecast.resolutionHint')}><select aria-label={t('forecast.resolution')} className={inputCls} value={resolution} onChange={e => setResolution(e.target.value as typeof resolution)}>
          {RESOLUTIONS.map(r => <option key={r} value={r}>{t(`forecast.resolutions.${r}`)}</option>)}</select></Field>
      </div>
      <div className="flex flex-wrap items-start justify-between gap-3 text-sm">
        <span className="text-[var(--muted)]">{progress?.complete ? t('holdout.complete')
          : progress?.through ? t('holdout.progress', { through: fmtDate(progress.through), pct: `${Math.round((progress.share ?? 0) * 100)}%` })
          : t('holdout.progressNone')}</span>
        {!progress?.complete && pickedItem && <RefreshData item={pickedItem} seasonId={season.id} latest={false} seasonName={seasonName} />}
      </div>
      <Button onClick={submit} disabled={!pickedItem} loading={start.isPending || stream.status === 'running' || stream.status === 'queued'}>{t('holdout.run')}</Button>
      {jobId && <ProgressPanel kind="forecast" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} />}
      {!!history.data?.length && <div className="text-sm">
        <div className="text-xs text-[var(--muted)] mb-1">{t('holdout.history')}</div>
        <ul className="flex flex-wrap gap-2">{history.data.map(h => <li key={h.run_id}>
          <button type="button" onClick={() => setRunId(h.run_id)}
            className={`px-2 py-1 rounded border text-xs ${h.run_id === runId ? 'border-[var(--accent)]' : 'border-[var(--border)]'}`}>
            {itemName(tracked.find(i => i.name === h.item) ?? { name: h.item }, i18n.language)} · {seasonLabel(h.season_name, t)} · {modelLabel(h.selected, t)} {fmtPct(h.skill)}</button></li>)}</ul>
      </div>}
      {run.data && <div className="space-y-2">
        <div className="flex justify-end"><Button variant="ghost" title={t('holdout.delete')}
          onClick={() => { if (window.confirm(t('holdout.confirmDelete'))) del.mutate(run.data!.run_id, { onSuccess: () => setRunId('') }) }}><Trash2 size={14} /></Button></div>
        <HoldoutResult run={run.data} />
      </div>}
    </div>
  )
}
