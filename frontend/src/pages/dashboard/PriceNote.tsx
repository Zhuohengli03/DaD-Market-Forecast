import { useTranslation } from 'react-i18next'
import { fmtTsUtc } from '../../lib/format'

type Priced = { latest: number | null; latest_ts?: string | null; latest_listings?: number | null }

/** What a shown price is: the median unit price of one hour's listings, which hour and how many. */
export function PriceNote({ v, compact = false }: { v: Priced; compact?: boolean }) {
  const { t } = useTranslation()
  if (v.latest == null) return null
  return (
    <div className="text-xs text-[var(--muted)] leading-snug" title={t('price.hint')}>
      <div>{t('price.what')}</div>
      {!compact && v.latest_ts && <div>{t('price.when', { at: fmtTsUtc(v.latest_ts), n: v.latest_listings ?? 0 })}</div>}
    </div>
  )
}
