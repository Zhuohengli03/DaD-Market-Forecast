import { useTranslation } from 'react-i18next'
import { CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import type { JobKind, JobProgress, JobStatus } from '../api/types'
import { localizeMessage, modelLabel } from '../lib/labels'

interface Props { kind: JobKind; status: JobStatus | null; progress: JobProgress | null; error: string | null; disconnected: boolean }

/** One plain sentence describing where a job is, for people who never want to read a log. */
export function describe(kind: JobKind, p: JobProgress | null, t: (k: string, o?: Record<string, unknown>) => string): string {
  if (!p) return t('progress.waiting')
  const d = p.detail ?? {}
  const n = (k: string) => (d[k] as number | undefined) ?? 0
  switch (`${kind}:${p.stage}`) {
    case 'forecast:load': return t('progress.forecast.load')
    case 'forecast:backtest': return p.done ? t('progress.forecast.backtest', { done: p.done, total: p.total, model: modelLabel(String(d.model ?? ''), t as never) }) : t('progress.forecast.backtestStart')
    case 'forecast:select': return t('progress.forecast.select')
    case 'forecast:fit': return t('progress.forecast.fit', { model: modelLabel(String(d.model ?? ''), t as never) })
    case 'forecast:references': return t('progress.forecast.references')
    case 'forecast:done': return t('progress.forecast.done')
    case 'collect:collect': case 'task:collect': return t('progress.collect.page', { page: p.done, new: n('new') })
    case 'collect:done': case 'task:done': return n('new') ? t('progress.collect.done', { new: n('new') }) : t('progress.collect.none')
    case 'catalog:items': return t('progress.catalog.items', { n: p.done })
    case 'catalog:names': return t('progress.catalog.names')
    case 'catalog:check': return t('progress.catalog.check', { done: p.done, total: p.total })
    case 'catalog:done': return d.tradable !== undefined ? t('progress.catalog.checked', { checked: n('checked'), tradable: n('tradable') }) : t('progress.catalog.synced', { items: n('items'), types: n('types') })
    default: return t('progress.working')
  }
}

function failure(kind: JobKind, p: JobProgress | null, error: string | null, lang: string, t: (k: string) => string): string {
  const code = p?.stage === 'error' ? String(p.detail?.code ?? '') : ''
  if (code === 'auth') return t('progress.error.auth')
  if (code === 'api') return t('progress.error.api')
  if (error) {
    const msg = error.replace(/^\w+(Error|Exception): /, '')
    if (/^KeyError/.test(error) && /no CSV data|not tracked/.test(error)) return t('progress.error.noData')
    // our own messages are Chinese or "中文 / English"; anything else is internal detail
    if (/[\u4e00-\u9fff]/.test(msg)) return localizeMessage(msg, lang)
  }
  return t(`progress.error.${kind}`)
}

export function ProgressPanel({ kind, status, progress, error, disconnected }: Props) {
  const { t, i18n } = useTranslation()
  const done = status === 'succeeded'
  const failed = status === 'failed'
  const pct = progress?.total && progress.done !== null ? Math.round((progress.done / progress.total) * 100) : null
  // collection page budgets are an upper bound, not a forecast of the work, so its bar is indeterminate
  const determinate = pct !== null && !(kind === 'collect' || kind === 'task')
  return (
    <div className="space-y-3" aria-live="polite">
      <div className="flex items-center gap-2 text-sm">
        {done ? <CheckCircle2 className="text-[var(--good)]" size={18} /> : failed ? <AlertCircle className="text-[var(--bad)]" size={18} /> : <Loader2 className="animate-spin text-[var(--accent)]" size={18} />}
        {failed ? <span role="alert" className="text-[var(--bad)]">{failure(kind, progress, error, i18n.language, t)}</span>
          : <span className={done ? 'font-medium' : ''}>{describe(kind, progress, t)}</span>}
      </div>
      {!done && !failed && (
        <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={determinate ? pct! : undefined} className="h-2 rounded-full bg-gray-100 overflow-hidden">
          {determinate ? <div className="h-full bg-[var(--accent)] transition-all" style={{ width: `${pct}%` }} />
            : <div className="h-full w-1/3 bg-[var(--accent)] rounded-full animate-[progress_1.2s_ease-in-out_infinite]" />}
        </div>
      )}
      {disconnected && !done && !failed && <p className="text-xs text-[var(--muted)]">{t('progress.reconnecting')}</p>}
    </div>
  )
}
