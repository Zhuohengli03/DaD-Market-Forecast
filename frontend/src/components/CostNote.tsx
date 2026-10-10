import { useTranslation } from 'react-i18next'
import type { JobCost } from '../api/types'
import { fmtCount } from '../lib/quota'

/** "Estimated ≤ N requests · used M" for a task that calls DarkerDB. */
export function CostNote({ cost, running }: { cost?: JobCost | null; running?: boolean }) {
  const { t } = useTranslation()
  if (!cost || (cost.estimate == null && !cost.used)) return null
  const parts = []
  if (cost.estimate != null) parts.push(t('cost.estimate', { n: fmtCount(cost.estimate) }))
  parts.push(t(running ? 'cost.usedSoFar' : 'cost.used', { n: fmtCount(cost.used) }))
  return <span>{parts.join(' · ')}</span>
}
