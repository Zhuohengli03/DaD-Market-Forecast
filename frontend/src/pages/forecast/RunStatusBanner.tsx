import { useTranslation } from 'react-i18next'
import type { ForecastRun } from '../../api/types'
import { fmtPct } from '../../lib/format'
import { localizeWarning, modelLabel, seasonLabel } from '../../lib/labels'

/** Finished seasons get no new listings, so "collect fresh data" does not apply. */
export function isPastSeason(run: ForecastRun): boolean {
  const s = run.season as { end?: string | null; ended?: boolean } | undefined
  return s?.ended ?? (!!s?.end && new Date(s.end).getTime() <= Date.now())
}

const STALE_HOURS = 24

export function RunStatusBanner({ run }: { run: ForecastRun }) {
  const { t, i18n } = useTranslation()
  const good = run.beats_baseline
  const age = run.data_age_hours ?? 0
  const ended = isPastSeason(run)
  return (
    <div className="space-y-2">
      <div className={`rounded-lg px-4 py-3 border ${good ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
        <div className="font-semibold">{good ? t('forecast.beats') : t('forecast.notBeats')}</div>
        {!good && run.baseline_reason && <div className="text-xs text-[var(--muted)] mt-0.5">{localizeWarning(run.baseline_reason, i18n.language)}</div>}
        <div className="text-sm mt-1 flex flex-wrap gap-x-4 gap-y-1">
          <span>{run.skill_vs_naive >= 0 ? t('forecast.skillBetter', { p: fmtPct(run.skill_vs_naive).replace('+', '') }) : t('forecast.skillWorse', { p: fmtPct(-run.skill_vs_naive).replace('+', '') })}</span>
          <span>{t('forecast.selected')}: <b>{modelLabel(run.selected.name, t)}</b>{run.selected.members.length > 1 && <span className="text-[var(--muted)]"> ({run.selected.members.map(m => `${modelLabel(m.name, t)} ${Math.round(m.weight * 100)}%`).join(', ')})</span>}</span>
          <span className="text-[var(--muted)]">{t('forecast.origins', { n: run.backtest?.origins ?? 0 })}</span>
        </div>
      </div>
      {ended && <div className="rounded-lg px-4 py-2 border border-[var(--line)] bg-gray-50 text-sm text-[var(--muted)]">{t('forecast.pastSeason', { season: seasonLabel(run.season?.name, t) })}</div>}
      {!ended && age > STALE_HOURS && <div className="rounded-lg px-4 py-2 border border-red-200 bg-red-50 text-sm">{t('forecast.stale', { days: Math.round(age / 24) })}</div>}
    </div>
  )
}
