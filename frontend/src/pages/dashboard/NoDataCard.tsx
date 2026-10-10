import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { useUntrackBulk } from '../../api/hooks'
import type { ItemInfo } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { useToast } from '../../components/ui/Toast'
import { itemName } from '../../lib/labels'

/** Tracked items with no listings yet (collection interrupted, or not reached yet), in one place with ways out. */
export function NoDataCard({ items, states }: { items: ItemInfo[]; states: Record<string, 'collecting' | 'waiting'> }) {
  const { t, i18n } = useTranslation()
  const remove = useUntrackBulk(); const toast = useToast()
  const idle = items.filter(i => !states[i.name])
  const drop = (names: string[]) => remove.mutate(names, {
    onSuccess: r => toast.push(t('collect.removed', { n: r.removed })),
    onError: e => toast.push((e as Error).message, 'error'),
  })
  return (
    <Card title={t('dashboard.emptyItems.title', { n: items.length })} actions={<>
      <Link to="/collect" className="text-sm font-medium text-[var(--accent)] hover:underline">{t('dashboard.emptyItems.collect')}</Link>
      <Button size="sm" variant="ghost" disabled={!idle.length} loading={remove.isPending}
        onClick={() => window.confirm(t('dashboard.emptyItems.confirm', { n: idle.length })) && drop(idle.map(i => i.name))}>{t('dashboard.emptyItems.removeAll', { n: idle.length })}</Button></>}>
      <p className="text-sm text-[var(--muted)] mb-3">{t('dashboard.emptyItems.intro')}</p>
      <ul className="flex flex-wrap gap-2 max-h-64 overflow-auto">
        {items.map(i => (
          <li key={i.name} className="flex items-center gap-1.5 rounded-full border border-[var(--line)] pl-1 pr-2 py-0.5 text-sm">
            {i.icon_url ? <img src={i.icon_url} alt="" loading="lazy" className="w-5 h-5 object-contain" /> : <span className="w-5" />}
            <span>{itemName(i, i18n.language)}</span>
            {states[i.name] ? <Badge tone={states[i.name] === 'collecting' ? 'warn' : 'muted'}>{t(`running.state.${states[i.name]}`)}</Badge>
              : <button type="button" aria-label={`${t('collect.untrack')} ${itemName(i, i18n.language)}`} title={t('collect.untrack')}
                  className="text-[var(--muted)] hover:text-[var(--bad)]" onClick={() => drop([i.name])}><X size={14} /></button>}
          </li>))}
      </ul>
    </Card>
  )
}
