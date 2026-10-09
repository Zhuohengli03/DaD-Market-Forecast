import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useModels, useStartForecast } from '../../api/hooks'
import { useSelection } from '../../components/SelectorBar'
import { Button } from '../../components/ui/Button'
import { Field, inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'

const BASELINES = ['naive', 'seasonal_naive_24']
const HORIZONS = [24, 48, 72, 168]

export function ForecastForm({ onStarted }: { onStarted: (jobId: string) => void }) {
  const { t } = useTranslation()
  const sel = useSelection()
  const models = useModels()
  const start = useStartForecast()
  const toast = useToast()
  const [horizon, setHorizon] = useState(72)
  const [coverage, setCoverage] = useState(0.9)
  const [chosen, setChosen] = useState<string[]>(['ridge', 'hgb', 'rf', 'ets', 'arima'])
  const [error, setError] = useState<string | null>(null)
  const optional = (models.data ?? []).filter(m => !BASELINES.includes(m))
  const toggle = (m: string) => setChosen(c => (c.includes(m) ? c.filter(x => x !== m) : [...c, m]))
  const submit = () => {
    setError(null)
    if (!sel.item || !sel.season) { setError(`${t('common.item')} / ${t('common.season')}`); return }
    start.mutate({ item: sel.item, season_id: sel.season, horizon_hours: horizon, models: chosen.filter(m => optional.includes(m)), coverage },
      { onSuccess: id => { toast.push(t('forecast.pending'), 'ok'); onStarted(id) }, onError: e => { setError((e as Error).message); toast.push((e as Error).message, 'error') } })
  }
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <Field label={t('common.item')}><select className={inputCls} value={sel.item} onChange={e => sel.setItem(e.target.value)}>{sel.items.map(i => <option key={i.name} value={i.name}>{i.name}</option>)}</select></Field>
        <Field label={t('common.season')}><select className={inputCls} value={sel.season} onChange={e => sel.setSeason(e.target.value)}>{sel.seasons.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select></Field>
        <Field label={t('forecast.horizon')}><select className={inputCls} value={horizon} onChange={e => setHorizon(Number(e.target.value))}>{HORIZONS.map(h => <option key={h} value={h}>{h} h</option>)}</select></Field>
        <Field label={t('forecast.coverage')}><select className={inputCls} value={coverage} onChange={e => setCoverage(Number(e.target.value))}>{[0.8, 0.9, 0.95].map(c => <option key={c} value={c}>{Math.round(c * 100)}%</option>)}</select></Field>
      </div>
      <Field label={t('forecast.models')} error={error}>
        <div className="flex flex-wrap gap-3 text-sm">
          {BASELINES.map(m => <label key={m} className="flex items-center gap-1 text-[var(--muted)]"><input type="checkbox" checked disabled readOnly aria-label={m} />{m}</label>)}
          {optional.map(m => <label key={m} className="flex items-center gap-1"><input type="checkbox" checked={chosen.includes(m)} onChange={() => toggle(m)} aria-label={m} />{m}</label>)}
        </div>
      </Field>
      <Button onClick={submit} loading={start.isPending}>{t('forecast.start')}</Button>
    </div>
  )
}
