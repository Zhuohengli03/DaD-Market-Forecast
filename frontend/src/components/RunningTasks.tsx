import { useIsMutating } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { useStatus } from '../api/hooks'
import { isIdle, summarizeRunning } from '../lib/running'

/** "Data collection: 2 collecting · 5 waiting", "Forecast: 1", … one line per area that has work running. */
export function RunningTasks({ compact = false }: { compact?: boolean }) {
  const { t } = useTranslation()
  const status = useStatus()
  const equipment = useIsMutating({ mutationKey: ['equipmentQuery'] })
  const s = summarizeRunning(status.data, equipment)
  if (isIdle(s)) return <span className={compact ? 'text-gray-500' : 'text-sm text-[var(--muted)]'}>{t('dashboard.noJobs')}</span>
  const lines: [string, string][] = []
  if (s.collecting || s.waiting) lines.push([t('running.collect'), [s.collecting && t('running.collecting', { n: s.collecting }), s.waiting && t('running.waiting', { n: s.waiting })].filter(Boolean).join(' · ')])
  if (s.init) lines.push([t('running.init'), t('running.inProgress')])
  if (s.forecast) lines.push([t('running.forecast'), t('running.n', { n: s.forecast })])
  if (s.equipment) lines.push([t('running.equipment'), t('running.querying')])
  if (s.catalog) lines.push([t('running.catalog'), t('running.n', { n: s.catalog })])
  return (
    <ul className={compact ? 'space-y-0.5' : 'text-sm space-y-1'} aria-label={t('dashboard.runningJobs')}>
      {lines.map(([area, what]) => (
        <li key={area} className="flex justify-between gap-2">
          <span className={compact ? 'text-gray-300' : ''}>{area}</span>
          <span className={compact ? 'text-amber-300 tabular-nums' : 'text-[var(--warn)] tabular-nums'}>{what}</span>
        </li>))}
    </ul>
  )
}
