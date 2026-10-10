import { useTranslation } from 'react-i18next'
import { Gauge } from 'lucide-react'
import { useQuota } from '../../api/hooks'
import { fmtRelative } from '../../lib/format'
import { fmtCount } from '../../lib/quota'

/** Credits left on the DarkerDB plan, as reported by the last request this app made (checking costs nothing). */
export function QuotaWidget() {
  const { t } = useTranslation()
  const q = useQuota().data
  const left = q?.credits_remaining ?? null
  const pct = left !== null && q ? Math.max(0, Math.min(100, Math.round((left / q.monthly_quota) * 100))) : null
  const tone = pct === null ? 'bg-gray-500' : pct < 10 ? 'bg-red-500' : pct < 30 ? 'bg-amber-400' : 'bg-emerald-400'
  return (
    <div className="px-4 py-2 text-xs text-gray-400" aria-label={t('quota.title')}>
      <div className="flex items-center gap-1.5 text-gray-300"><Gauge size={12} />{t('quota.title')}</div>
      {left === null ? <div className="mt-1">{t('quota.unknown')}</div> : <>
        <div className="mt-1"><span className="text-white font-medium tabular-nums">{fmtCount(left)}</span> / {fmtCount(q?.monthly_quota)} {t('quota.left')}</div>
        <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? 0} className="mt-1 h-1.5 rounded-full bg-white/10 overflow-hidden"><div className={`h-full ${tone}`} style={{ width: `${pct}%` }} /></div>
        <div className="mt-1 text-gray-500">{t('quota.updated', { when: fmtRelative(q?.updated_at) })}{q?.rate_limit ? ` · ${t('quota.perMin', { n: q.rate_limit })}` : ''}</div>
      </>}
    </div>
  )
}
