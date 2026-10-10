import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CheckCircle2, Circle } from 'lucide-react'
import { useItems, useSetup } from '../api/hooks'
import { Card } from '../components/ui/Card'

/** Ordered first-run checklist; hidden once the user can view prices and forecasts. */
export function GettingStarted() {
  const { t } = useTranslation()
  const setup = useSetup(); const items = useItems()
  if (!setup.isSuccess || !items.isSuccess) return null
  const s = setup.data
  const steps = [
    { key: 'key', done: s.api_key_configured, to: '/settings' },
    { key: 'seasons', done: s.seasons > 0, to: '/settings' },
    { key: 'items', done: s.tracked_items > 0, to: '/catalog' },
    { key: 'data', done: (items.data ?? []).some(i => (i.rows ?? 0) > 0), to: '/collect' },
  ]
  if (steps.every(x => x.done)) return null
  const next = steps.find(x => !x.done)
  return (
    <Card title={t('start.title')}>
      <p className="text-sm text-[var(--muted)] mb-3">{t('start.intro')}</p>
      <ol className="space-y-2">
        {steps.map((x, i) => (
          <li key={x.key} className="flex items-center gap-3 text-sm">
            {x.done ? <CheckCircle2 size={18} className="text-[var(--good)]" /> : <Circle size={18} className="text-gray-300" />}
            <span className={x.done ? 'text-[var(--muted)] line-through' : 'font-medium'}>{i + 1}. {t(`start.${x.key}`)}</span>
            {x === next && <Link to={x.to} className="ml-auto text-[var(--accent)] font-medium hover:underline">{t(`start.${x.key}Go`)}</Link>}
          </li>
        ))}
      </ol>
    </Card>
  )
}
