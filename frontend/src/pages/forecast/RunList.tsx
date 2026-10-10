import { useTranslation } from 'react-i18next'
import { useRuns } from '../../api/hooks'
import { Badge } from '../../components/ui/Badge'
import { EmptyState } from '../../components/ui/EmptyState'
import { fmtTs } from '../../lib/format'
import { modelLabel } from '../../lib/labels'

export function RunList({ selected, onSelect }: { selected: string | null; onSelect: (id: string) => void }) {
  const { t } = useTranslation()
  const runs = useRuns()
  if (!runs.data?.length) return <EmptyState title={t('forecast.noRuns')} />
  return (
    <ul className="divide-y divide-[var(--line)]">
      {runs.data.map(r => (
        <li key={r.run_id}><button type="button" onClick={() => onSelect(r.run_id)} className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-50 ${selected === r.run_id ? 'bg-blue-50' : ''}`}>
          <div className="flex justify-between"><span className="font-medium">{r.item}</span><Badge tone={r.beats_baseline ? 'ok' : 'warn'}>{modelLabel(r.selected, t)}</Badge></div>
          <div className="text-xs text-[var(--muted)]">{r.season_name} · {r.horizon_hours}h · {fmtTs(r.created_at)}{r.warnings ? ` · ⚠ ${r.warnings}` : ''}</div>
        </button></li>))}
    </ul>
  )
}
