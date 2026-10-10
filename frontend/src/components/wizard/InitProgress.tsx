import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CheckCircle2, AlertCircle, Loader2, Circle } from 'lucide-react'
import type { InitItem, JobProgress, JobStatus } from '../../api/types'
import { itemName } from '../../lib/labels'

const STEPS = ['catalog', 'seasons', 'collect', 'forecast', 'done'] as const

export function ItemLine({ it }: { it: InitItem }) {
  const { t, i18n } = useTranslation()
  const collect = it.status === 'waiting' ? t('wizard.progress.waiting')
    : it.status === 'collecting' ? t('wizard.progress.collecting', { pages: it.pages, n: it.collected })
    : it.status === 'collect_failed' ? (it.error === 'auth' ? t('wizard.progress.collectFailedAuth') : t('wizard.progress.collectFailed'))
    : t('wizard.progress.collected', { n: it.collected })
  const forecast = it.status === 'forecasting' ? <span>{t('wizard.progress.forecasting')}</span>
    : it.forecast === 'ok' ? <span className="text-[var(--good)]">{t('wizard.progress.forecastOk')} · <Link to={`/forecast?run=${it.run_id}`} className="underline">{t('wizard.progress.view')}</Link></span>
    : it.forecast === 'not_enough_data' ? <span className="text-[var(--muted)]">{it.hours !== null ? t('wizard.progress.notEnough', { hours: it.hours }) : t('wizard.progress.notEnoughNoHours')}</span>
    : it.forecast === 'failed' ? <span className="text-[var(--warn)]">{t('wizard.progress.forecastFailed')}</span> : null
  const icon = it.status === 'collect_failed' ? <AlertCircle size={16} className="text-[var(--bad)]" />
    : it.status === 'done' || it.status === 'collected' ? <CheckCircle2 size={16} className="text-[var(--good)]" />
    : it.status === 'waiting' ? <Circle size={16} className="text-gray-300" /> : <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
  return (
    <li className="flex gap-3 text-sm py-1.5">
      <span className="pt-0.5">{icon}</span>
      <div className="flex-1">
        <div className="font-medium">{itemName(it, i18n.language)}</div>
        <div className={it.status === 'collect_failed' ? 'text-[var(--bad)]' : 'text-[var(--muted)]'}>{collect}</div>
        {forecast && <div>{forecast}</div>}
      </div>
    </li>
  )
}

export function InitProgress({ status, progress, error }: { status: JobStatus | null; progress: JobProgress | null; error: string | null }) {
  const { t } = useTranslation()
  const stage = progress?.stage ?? 'catalog'
  const at = STEPS.indexOf(stage as typeof STEPS[number])
  const items = ((progress?.detail?.items as InitItem[] | undefined) ?? [])
  const pct = status === 'succeeded' ? 100 : Math.round((Math.max(at, 0) / 4) * 100)
  return (
    <div className="space-y-4">
      <ol className="flex flex-wrap gap-2 text-xs">
        {STEPS.slice(0, 4).map((s, i) => (
          <li key={s} className={`px-2 py-1 rounded-full border ${i < at || status === 'succeeded' ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : i === at ? 'bg-blue-50 border-blue-200 text-blue-700' : 'border-[var(--line)] text-[var(--muted)]'}`}>{i + 1}. {t(`wizard.progress.steps.${s}`)}</li>
        ))}
      </ol>
      <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} className="h-2 rounded-full bg-gray-100 overflow-hidden">
        <div className="h-full bg-[var(--accent)] transition-all" style={{ width: `${pct}%` }} />
      </div>
      {status === 'failed' && <p role="alert" className="text-sm text-[var(--bad)]">{error ?? t('progress.error.catalog')}</p>}
      {status === 'succeeded' && <p className="text-sm font-medium text-[var(--good)]">{t('wizard.progress.allDone')}</p>}
      <ul className="divide-y divide-[var(--line)]">{items.map(it => <ItemLine key={it.item_id} it={it} />)}</ul>
    </div>
  )
}

/** Per-item rows for a job that collects several items in a row (the "collect all" button). */
export function BatchProgress({ status, progress, error }: { status: JobStatus | null; progress: JobProgress | null; error: string | null }) {
  const { t } = useTranslation()
  const items = ((progress?.detail?.items as InitItem[] | undefined) ?? [])
  const finished = items.filter(i => i.status === 'collected' || i.status === 'collect_failed').length
  const pct = status === 'succeeded' ? 100 : items.length ? Math.round((finished / items.length) * 100) : 0
  return (
    <div className="space-y-3">
      <div className="text-sm font-medium">{t('collect.allProgress', { done: finished, total: items.length })}</div>
      <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} className="h-2 rounded-full bg-gray-100 overflow-hidden">
        <div className="h-full bg-[var(--accent)] transition-all" style={{ width: `${pct}%` }} />
      </div>
      {status === 'failed' && <p role="alert" className="text-sm text-[var(--bad)]">{error ?? t('progress.error.collect')}</p>}
      {status === 'succeeded' && <p className="text-sm font-medium text-[var(--good)]">{t('collect.allDone')}</p>}
      <ul className="divide-y divide-[var(--line)] max-h-96 overflow-auto">{items.map(it => <ItemLine key={it.item_id} it={it} />)}</ul>
    </div>
  )
}
