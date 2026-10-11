import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import type { HoldoutRun } from '../../api/types'
import { fmtInZone, fmtNum, fmtPct } from '../../lib/format'
import { localizeWarning, modelLabel } from '../../lib/labels'

const ZONES = [{ key: 'ny', tz: 'America/New_York' }, { key: 'utc', tz: 'UTC' }] as const
const pct0 = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? '–' : `${Math.round(x * 100)}%`)

/** One sentence: did the frozen method beat "last hour" on the part of the season it never saw? */
export function holdoutVerdict(run: HoldoutRun, t: TFunction): string {
  const m = run.metrics
  const args = { test: run.test_hours, method: modelLabel(run.selected.name, t), p: pct0(Math.abs(m.skill)), mape: pct0(m.mape) }
  return m.skill > 0 ? t('holdout.verdictBetter', args) : t('holdout.verdictWorse', args)
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded border border-[var(--border)] p-3" title={hint}>
      <div className="text-xs text-[var(--muted)]">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
    </div>
  )
}

function HoldoutChart({ run, tz }: { run: HoldoutRun; tz: string }) {
  const { t } = useTranslation()
  const N = { train: t('holdout.train'), actual: t('holdout.actual'), pred: t('holdout.pred'), naive: t('holdout.naive'), band: t('chart.band') }
  const nT = run.train_tail.length, pts = run.points
  const x = [...run.train_tail.map(p => p.ts), ...pts.map(p => p.ts)]
  const pad = (arr: (number | null)[]) => [...Array(nT).fill(null), ...arr]
  const series = [
    { name: N.train, type: 'line', color: '#9ca3af', data: [...run.train_tail.map(p => p.actual), ...Array(pts.length).fill(null)], symbol: 'none', lineStyle: { width: 1 } },
    { name: N.band, type: 'line', data: pad(pts.map(p => p.lower)), lineStyle: { opacity: 0 }, stack: 'ci', symbol: 'none', silent: true, tooltip: { show: false } },
    { name: N.band, type: 'line', color: '#93c5fd', data: pad(pts.map(p => Math.max(0, p.upper - p.lower))), lineStyle: { opacity: 0 }, stack: 'ci', areaStyle: { color: '#2563eb', opacity: 0.12 }, symbol: 'none', silent: true, tooltip: { show: false } },
    { name: N.actual, type: 'line', color: '#16a34a', data: pad(pts.map(p => p.actual)), symbol: 'none', lineStyle: { width: 1.5 } },
    { name: N.pred, type: 'line', color: '#2563eb', data: pad(pts.map(p => p.pred)), symbol: 'none', lineStyle: { width: 1.5, type: 'dashed' } },
    { name: N.naive, type: 'line', color: '#d97706', data: pad(pts.map(p => p.naive)), symbol: 'none', lineStyle: { width: 1, type: 'dotted' } },
  ]
  const option = { animation: false, tooltip: { trigger: 'axis', valueFormatter: (v: number) => (v == null ? '–' : Number(v).toFixed(2)) },
    legend: { top: 0, data: [N.train, N.actual, N.pred, N.naive, { name: N.band, icon: 'roundRect' }] }, grid: { left: 56, right: 24, top: 36, bottom: 60 },
    xAxis: { type: 'category', data: x, axisLabel: { formatter: (v: string) => fmtInZone(v, tz) } }, yAxis: { type: 'value', scale: true, name: t('chart.unitPrice') },
    dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 4 }], series }
  return <ReactECharts option={option} notMerge style={{ height: 380, width: '100%' }} />
}

export function HoldoutResult({ run }: { run: HoldoutRun }) {
  const { t, i18n } = useTranslation()
  const [zone, setZone] = useState<(typeof ZONES)[number]['key']>('ny')
  const tz = ZONES.find(z => z.key === zone)!.tz
  const m = run.metrics
  return (
    <div className="space-y-3">
      <p className={`text-sm font-medium ${m.skill > 0 ? 'text-[var(--good)]' : 'text-[var(--warn)]'}`}>{holdoutVerdict(run, t)}</p>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--muted)]">
        <span>{t('holdout.split', { train: run.train_hours, test: run.test_hours, at: fmtInZone(run.split_ts, tz) })} · {t('holdout.selected', { name: modelLabel(run.selected.name, t), p: fmtPct(run.train_skill) })}</span>
        <span className="flex items-center gap-1">{t('holdout.zone')}
          {ZONES.map(z => <button key={z.key} type="button" aria-pressed={zone === z.key} onClick={() => setZone(z.key)}
            className={`px-2 py-0.5 rounded border ${zone === z.key ? 'border-[var(--accent)] text-[var(--fg)]' : 'border-[var(--border)]'}`}>{t(`holdout.zones.${z.key}`)}</button>)}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2">
        <Metric label={t('holdout.mae')} value={fmtNum(m.mae)} hint={t('holdout.maeHint')} />
        <Metric label={t('holdout.naiveMae')} value={fmtNum(m.naive_mae)} hint={t('holdout.naiveMaeHint')} />
        <Metric label={t('holdout.skill')} value={fmtPct(m.skill)} hint={t('holdout.skillHint')} />
        <Metric label={t('holdout.mdape')} value={m.mdape == null ? '–' : `${(m.mdape * 100).toFixed(1)}%`} hint={t('holdout.mdapeHint')} />
        <Metric label={t('holdout.mape')} value={pct0(m.mape)} hint={t('holdout.mapeHint')} />
        <Metric label={t('holdout.direction')} value={pct0(m.direction_acc)} hint={t('holdout.direction')} />
        <Metric label={t('holdout.coverage')} value={pct0(m.coverage)} hint={t('holdout.coverageHint', { target: pct0(run.nominal_coverage) })} />
      </div>
      <HoldoutChart run={run} tz={tz} />
      <details className="text-sm"><summary className="cursor-pointer text-[var(--muted)]">{t('holdout.daily')}</summary>
        <p className="text-xs text-[var(--muted)] my-1">{t('holdout.dayHint')}</p>
        <table className="w-full text-xs tabular-nums">
          <thead><tr className="text-left text-[var(--muted)]"><th>{t('holdout.day')}</th><th>{t('holdout.dayMae')}</th><th>{t('holdout.dayNaive')}</th><th>{t('holdout.skill')}</th></tr></thead>
          <tbody>{run.windows.map(w => <tr key={w.origin} className="border-t border-[var(--border)]">
            <td>{fmtInZone(w.origin, tz)}</td><td>{fmtNum(w.mae)}</td><td>{fmtNum(w.naive_mae)}</td>
            <td className={w.skill > 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}>{fmtPct(w.skill)}</td></tr>)}</tbody>
        </table>
      </details>
      {run.warnings.length > 0 && <ul className="text-xs text-[var(--warn)] list-disc pl-4">{run.warnings.map(w => <li key={w}>{localizeWarning(w, i18n.language)}</li>)}</ul>}
    </div>
  )
}
