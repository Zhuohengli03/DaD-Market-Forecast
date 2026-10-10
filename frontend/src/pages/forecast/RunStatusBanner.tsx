import { useTranslation } from 'react-i18next'
import type { ForecastRun } from '../../api/types'
import { fmtPct } from '../../lib/format'
import { modelLabel } from '../../lib/labels'

const STALE_HOURS = 24

export function RunStatusBanner({ run }: { run: ForecastRun }) {
  const { t } = useTranslation()
  const good = run.beats_baseline
  const age = run.data_age_hours ?? 0
  return (
    <div className="space-y-2">
      <div className={`rounded-lg px-4 py-3 border ${good ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
        <div className="font-semibold">{good ? t('forecast.beats') : t('forecast.notBeats')}</div>
        {!good && run.baseline_reason && <div className="text-xs text-[var(--muted)] mt-0.5">{run.baseline_reason}</div>}
        <div className="text-sm mt-1 flex flex-wrap gap-x-4 gap-y-1">
          <span>{t('forecast.skill')}: <b>{fmtPct(run.skill_vs_naive)}</b></span>
          <span>{t('forecast.selected')}: <b>{modelLabel(run.selected.name, t)}</b>{run.selected.members.length > 1 && <span className="text-[var(--muted)]"> ({run.selected.members.map(m => `${modelLabel(m.name, t)} ${Math.round(m.weight * 100)}%`).join(', ')})</span>}</span>
          <span className="text-[var(--muted)]">{t('forecast.origins', { n: run.backtest?.origins ?? 0 })}</span>
        </div>
      </div>
      {age > STALE_HOURS && <div className="rounded-lg px-4 py-2 border border-red-200 bg-red-50 text-sm">{t('forecast.stale', { days: Math.round(age / 24) })}</div>}
    </div>
  )
}
