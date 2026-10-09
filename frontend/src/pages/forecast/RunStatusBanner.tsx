import { useTranslation } from 'react-i18next'
import type { ForecastRun } from '../../api/types'
import { fmtPct } from '../../lib/format'

export function RunStatusBanner({ run }: { run: ForecastRun }) {
  const { t } = useTranslation()
  const good = run.beats_baseline
  return (
    <div className={`rounded-lg px-4 py-3 border ${good ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
      <div className="font-semibold">{good ? t('forecast.beats') : t('forecast.notBeats')}</div>
      <div className="text-sm mt-1 flex gap-4"><span>{t('forecast.skill')}: <b>{fmtPct(run.skill_vs_naive)}</b></span><span>{t('forecast.selected')}: <b>{run.selected.name}</b>{run.selected.members.length > 1 && <span className="text-[var(--muted)]"> ({run.selected.members.map(m => `${m.name} ${Math.round(m.weight * 100)}%`).join(', ')})</span>}</span></div>
    </div>
  )
}
