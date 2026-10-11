import { useTranslation } from 'react-i18next'
import type { ForecastRun, ModelMetric } from '../../api/types'
import { fmtNum } from '../../lib/format'
import { modelHelp, modelLabel } from '../../lib/labels'
import { localizeError } from '../../lib/errors'

const BASELINES = ['naive', 'seasonal_naive_24']

const fmtShare = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(1)}%`)

export function BacktestTable({ run }: { run: ForecastRun }) {
  const { t, i18n } = useTranslation()
  const members = new Set([run.selected.name, ...run.selected.members.map(m => m.name)])
  const rows = [...run.backtest.models].sort((a, b) => {
    const ab = BASELINES.indexOf(a.name), bb = BASELINES.indexOf(b.name)
    if (ab !== -1 || bb !== -1) return (ab === -1 ? 99 : ab) - (bb === -1 ? 99 : bb)
    return (a.mae_mean ?? 1e9) - (b.mae_mean ?? 1e9)
  })
  const cols: [string, keyof ModelMetric][] = [[t('forecast.mae'), 'mae_mean'], [t('forecast.maeH1'), 'mae_h1'], [t('forecast.maeH24'), 'mae_h24'], [t('forecast.rmse'), 'rmse_mean'], [t('forecast.mase'), 'mase_mean'], [t('forecast.mdape'), 'mdape_mean'], [t('forecast.dir'), 'direction_acc']]
  return (
    <div className="space-y-2">
      <div className="text-xs text-[var(--muted)]">{t('forecast.origins', { n: run.backtest.origins })}{run.backtest.independent_windows != null && run.backtest.independent_windows < run.backtest.origins && t('forecast.independent', { n: run.backtest.independent_windows })} · {t('forecast.evalHorizonN', { h: run.backtest.horizon })} · {t('forecast.coverageN', { actual: run.backtest.coverage == null ? '–' : Math.round(run.backtest.coverage * 100) + '%', target: Math.round(run.backtest.nominal_coverage * 100) + '%' })}</div>
      {run.backtest.in_sample ? <p className="text-xs text-[var(--muted)]">{t('forecast.inSample')}</p>
        : run.backtest.in_sample === false && <p className="text-xs text-[var(--muted)]">{t('forecast.splitJudged', { pick: run.backtest.selection_origins, judge: run.backtest.evaluation_origins })}</p>}
      <div className="overflow-x-auto"><table className="w-full text-sm whitespace-nowrap">
        <thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('forecast.model')}</th>{cols.map(([label, key]) => <th key={label} className="py-1 pl-3 text-right cursor-help underline decoration-dotted" title={t(`forecast.metricHelp.${key}`)}>{label}</th>)}</tr></thead>
        <tbody>{rows.map(r => (
          <tr key={r.name} className={`border-t border-[var(--line)] ${members.has(r.name) ? 'bg-blue-50 font-medium' : ''} ${BASELINES.includes(r.name) ? 'text-[var(--muted)]' : ''}`}>
            <td className="py-1" title={modelHelp(r.name, t)}>{modelLabel(r.name, t)}</td>{cols.map(([label, key]) => <td key={label} className="py-1 pl-3 text-right tabular-nums">{key === 'mdape_mean' ? fmtShare(r.mdape_mean) : fmtNum(r[key] as number | null, key === 'direction_acc' || key === 'mase_mean' ? 2 : 3)}</td>)}
          </tr>))}</tbody>
      </table></div>
      <p className="text-xs text-[var(--muted)]">{t('forecast.tableHint')}</p>
      {Object.entries(run.backtest.failures).map(([n, e]) => <div key={n} className="text-xs text-[var(--bad)]">{modelLabel(n, t)}: {localizeError(e, i18n.language)}</div>)}
    </div>
  )
}
