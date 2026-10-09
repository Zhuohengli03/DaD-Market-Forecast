import { useTranslation } from 'react-i18next'
import { useJobs } from '../api/hooks'
import type { Job } from '../api/types'
import { Badge } from './ui/Badge'
import { EmptyState } from './ui/EmptyState'
import { fmtTs } from '../lib/format'

const tone = (s: Job['status']) => (s === 'succeeded' ? 'ok' : s === 'failed' ? 'bad' : 'warn')

export function JobList({ kinds, selected, onSelect }: { kinds: Job['kind'][]; selected: string | null; onSelect: (id: string) => void }) {
  const { t } = useTranslation()
  const jobs = (useJobs().data ?? []).filter(j => kinds.includes(j.kind))
  if (!jobs.length) return <EmptyState title={t('common.none')} />
  return <ul className="divide-y divide-[var(--line)]">{jobs.map(j => (
    <li key={j.id}><button type="button" onClick={() => onSelect(j.id)} className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-50 flex justify-between items-center ${selected === j.id ? 'bg-blue-50' : ''}`}>
      <span><div>{j.label}</div><div className="text-xs text-[var(--muted)]">{fmtTs(j.created_at)}{j.error ? ` · ${j.error}` : ''}</div></span><Badge tone={tone(j.status)}>{j.status}</Badge>
    </button></li>))}</ul>
}
