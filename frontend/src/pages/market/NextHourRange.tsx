import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNextHour } from '../../api/hooks'
import { Card } from '../../components/ui/Card'
import { inputCls } from '../../components/ui/Field'
import { fmtNum, fmtTsUtc } from '../../lib/format'
import { fmtWait } from '../../lib/collectEta'

type Prices = { p10: number; p25: number; p50: number; p75: number; p90: number }
export type Verdict = 'very_cheap' | 'cheap' | 'normal' | 'pricey' | 'very_pricey'
const pct = (x: number | null | undefined) => (x == null ? '–' : `${Math.round(x * 100)}%`)
const TONE: Record<Verdict, string> = { very_cheap: 'text-[var(--good)]', cheap: 'text-[var(--good)]', normal: '', pricey: 'text-[var(--warn)]', very_pricey: 'text-[var(--bad)]' }

/** Where a price sits against next hour's sales: below P10, below P25, inside P25-P75, up to P90, above. */
export function priceVerdict(price: number, p: Prices): Verdict {
  if (price < p.p10) return 'very_cheap'
  if (price < p.p25) return 'cheap'
  if (price <= p.p75) return 'normal'
  return price <= p.p90 ? 'pricey' : 'very_pricey'
}

/** Next hour's sale-price range, for deciding whether a listing is a good buy or where to list. */
export function NextHourRange({ item, seasonId }: { item: string; seasonId?: string }) {
  const { t } = useTranslation()
  const q = useNextHour(item, seasonId)
  const [typed, setTyped] = useState('')
  const d = q.data
  const price = Number(typed)
  const verdict = d && typed && price > 0 ? priceVerdict(price, d.prices) : null
  const span = d ? d.prices.p90 - d.prices.p10 || 1 : 1
  const at = (v: number) => `${((v - (d?.prices.p10 ?? 0)) / span) * 100}%`
  return (
    <Card title={t('nextHour.title')}>
      {q.isLoading ? <p className="text-sm">{t('common.loading')}</p>
        : q.isError || !d ? <p className="text-sm text-[var(--muted)]">{(q.error as Error | null)?.message ?? t('common.none')}</p>
        : <div className="space-y-3 text-sm">
          <p>{t('nextHour.range', { hour: fmtTsUtc(d.next_hour), lo: fmtNum(d.prices.p25), hi: fmtNum(d.prices.p75), wlo: fmtNum(d.prices.p10), whi: fmtNum(d.prices.p90) })}</p>
          <div className="relative h-6 max-w-xl" aria-hidden>
            <div className="absolute inset-y-2 left-0 right-0 rounded bg-blue-100" />
            <div className="absolute inset-y-1 rounded bg-blue-400/70" style={{ left: at(d.prices.p25), width: `calc(${at(d.prices.p75)} - ${at(d.prices.p25)})` }} />
            <div className="absolute inset-y-0 w-0.5 bg-[var(--ink)]" style={{ left: at(d.prices.p50) }} />
          </div>
          <div className="flex justify-between max-w-xl text-xs text-[var(--muted)] tabular-nums">
            <span>{fmtNum(d.prices.p10)}</span><span>{fmtNum(d.prices.p25)}</span><span>{fmtNum(d.prices.p50)}</span><span>{fmtNum(d.prices.p75)}</span><span>{fmtNum(d.prices.p90)}</span>
          </div>
          <label className="flex flex-wrap items-center gap-2">
            <span className="text-[var(--muted)]">{t('nextHour.typed')}</span>
            <input type="number" min={0} step="0.01" aria-label={t('nextHour.typed')} className={`${inputCls} w-28`} value={typed} onChange={e => setTyped(e.target.value)} />
            {verdict && <span className={`font-medium ${TONE[verdict]}`}>{t(`nextHour.verdict.${verdict}`)}</span>}
          </label>
          <p className="text-xs text-[var(--muted)]">{t('nextHour.how', { level: fmtNum(d.level), days: d.calib_days })}{' '}
            {d.check.hours ? t('nextHour.check', { days: d.check_days, mid: pct(d.check.mid), wide: pct(d.check.wide) }) : ''}</p>
          {d.stale_hours > 24 && <p className="text-xs text-[var(--warn)]">{t('nextHour.stale', { ago: fmtWait(d.stale_hours * 3600) })}</p>}
          <p className="text-xs text-[var(--muted)]">{t('nextHour.notDirection')}</p>
        </div>}
    </Card>
  )
}
