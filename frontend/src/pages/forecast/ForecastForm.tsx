import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useModels, useStartForecast } from '../../api/hooks'
import { useSelection } from '../../components/SelectorBar'
import { Button } from '../../components/ui/Button'
import { Field, inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { itemName, modelHelp, modelLabel } from '../../lib/labels'

const BASELINES = ['naive', 'seasonal_naive_24']
const HORIZONS = [24, 48, 72, 168]

export function ForecastForm({ onStarted }: { onStarted: (jobId: string) => void }) {
  const { t, i18n } = useTranslation()
  const sel = useSelection()
  const models = useModels()
  const start = useStartForecast()
  const toast = useToast()
  const [horizon, setHorizon] = useState(72)
  const [coverage, setCoverage] = useState(0.9)
  const [originStep, setOriginStep] = useState(6)
  const [chosen, setChosen] = useState<string[]>(['ridge', 'hgb', 'rf', 'ets', 'arima'])
  const [error, setError] = useState<string | null>(null)
  const optional = (models.data ?? []).filter(m => !BASELINES.includes(m))
  const toggle = (m: string) => setChosen(c => (c.includes(m) ? c.filter(x => x !== m) : [...c, m]))
  const submit = () => {
    setError(null)
    if (!sel.item || !sel.season) { setError(`${t('common.item')} / ${t('common.season')}`); return }
    start.mutate({ item: sel.item, season_id: sel.season, horizon_hours: horizon, models: chosen.filter(m => optional.includes(m)), coverage, backtest_origin_step: originStep },
      { onSuccess: id => { toast.push(t('forecast.pending'), 'ok'); onStarted(id) }, onError: e => { setError((e as Error).message); toast.push((e as Error).message, 'error') } })
  }
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <Field label={t('common.item')}><select className={inputCls} value={sel.item} onChange={e => sel.setItem(e.target.value)}>{sel.items.map(i => <option key={i.name} value={i.name}>{itemName(i, i18n.language)}</option>)}</select></Field>
        <Field label={t('common.season')}><select className={inputCls} value={sel.season} onChange={e => sel.setSeason(e.target.value)}>{sel.seasons.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select></Field>
        <Field label={t('forecast.horizon')}><select className={inputCls} value={horizon} onChange={e => setHorizon(Number(e.target.value))}>{HORIZONS.map(h => <option key={h} value={h}>{h} {t('common.hours')}</option>)}</select></Field>
        <Field label={t('forecast.coverage')} hint={t('forecast.coverageHint')}><select className={inputCls} value={coverage} onChange={e => setCoverage(Number(e.target.value))}>{[0.8, 0.9, 0.95].map(c => <option key={c} value={c}>{Math.round(c * 100)}%</option>)}</select></Field>
      </div>
      <Field label={t('forecast.models')} hint={t('forecast.baselineHint')} error={error}>
        <div className="flex flex-wrap gap-3 text-sm">
          {BASELINES.map(m => <label key={m} title={modelHelp(m, t)} className="flex items-center gap-1 text-[var(--muted)]"><input type="checkbox" checked disabled readOnly aria-label={modelLabel(m, t)} />{modelLabel(m, t)}</label>)}
          {optional.map(m => <label key={m} title={modelHelp(m, t)} className="flex items-center gap-1"><input type="checkbox" checked={chosen.includes(m)} onChange={() => toggle(m)} aria-label={modelLabel(m, t)} />{modelLabel(m, t)}</label>)}
        </div>
      </Field>
      <details className="text-sm"><summary className="cursor-pointer text-[var(--muted)]">{t('forecast.advanced')}</summary>
        <div className="mt-2 max-w-xs"><Field label={t('forecast.originStep')} hint={t('forecast.originStepHint')}>
          <select className={inputCls} value={originStep} onChange={e => setOriginStep(Number(e.target.value))}>{[3, 6, 12].map(v => <option key={v} value={v}>{v} {t('common.hours')}</option>)}</select></Field></div>
      </details>
      <Button onClick={submit} loading={start.isPending}>{t('forecast.start')}</Button>
    </div>
  )
}
