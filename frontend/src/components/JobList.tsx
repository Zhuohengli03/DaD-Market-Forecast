import { useTranslation } from 'react-i18next'
import { useItems, useJobs } from '../api/hooks'
import type { Job } from '../api/types'
import { Badge } from './ui/Badge'
import { EmptyState } from './ui/EmptyState'
import { describe } from './ProgressPanel'
import { CostNote } from './CostNote'
import { fmtTs } from '../lib/format'
import { itemName, localizeMessage, seasonLabel } from '../lib/labels'

const tone = (s: Job['status']) => (s === 'succeeded' ? 'ok' : s === 'failed' ? 'bad' : 'warn')

export function JobList({ kinds, selected, onSelect }: { kinds: Job['kind'][]; selected: string | null; onSelect: (id: string) => void }) {
  const { t, i18n } = useTranslation()
  const jobs = (useJobs().data ?? []).filter(j => kinds.includes(j.kind))
  const items = useItems().data ?? []
  const label = (j: Job) => {
    let text = localizeMessage(j.label, i18n.language)
    const item = items.find(i => i.name === j.meta.item)
    if (item) text = text.replace(item.name, itemName(item, i18n.language))
    return text.split(' · ').map(part => /^\d+h$/.test(part)
      ? t('forecast.hoursN', { n: Number(part.slice(0, -1)) }) : seasonLabel(part, t)).join(' · ')
  }
  if (!jobs.length) return <EmptyState title={t('common.none')} />
  return <ul className="divide-y divide-[var(--line)]">{jobs.map(j => (
    <li key={j.id}><button type="button" onClick={() => onSelect(j.id)} className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-50 flex justify-between items-center gap-2 ${selected === j.id ? 'bg-blue-50' : ''}`}>
      <span><div>{label(j)}</div>
        <div className="text-xs text-[var(--muted)]">{fmtTs(j.created_at)}{j.status === 'succeeded' && j.progress ? ` · ${describe(j.kind, j.progress, t)}` : ''}</div>
        <div className="text-xs text-[var(--muted)]"><CostNote cost={j.cost} running={j.status === 'running' || j.status === 'queued'} /></div></span>
      <Badge tone={tone(j.status)}>{t(`progress.status.${j.status}`)}</Badge>
    </button></li>))}</ul>
}
