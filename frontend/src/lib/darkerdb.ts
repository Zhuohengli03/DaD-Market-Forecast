import type { TFunction } from 'i18next'
import { fmtDate } from './format'

/** DarkerDB keeps marketplace listings from this day on (checked: nothing earlier, ~40M listings since). */
export const RECORDS_START = '2026-07-31'

/** "2026-07-31（记录起点）" when the first listing is just where DarkerDB's records begin; otherwise the date. */
export function firstRecordLabel(iso: string, t: TFunction): string {
  const d = fmtDate(iso)
  return d === RECORDS_START ? `${d}${t('catalog.recordsStartTag')}` : d
}
