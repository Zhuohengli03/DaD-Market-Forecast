import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import type { OverviewGroup, OverviewVariant } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Card } from '../../components/ui/Card'
import { Sparkline } from '../../components/charts/Sparkline'
import { fmtNum, fmtPct, fmtRelative } from '../../lib/format'

const MAX_ROWS = 4

export const groupName = (g: OverviewGroup, lang: string) => (lang.startsWith('zh') && g.name_zh ? g.name_zh : g.name)
export const variantLabel = (v: OverviewVariant, lang: string) => (lang.startsWith('zh') ? v.label : v.label_en)
/** The quality with the most data this season stands for the type in the sparkline. */
export const leadVariant = (g: OverviewGroup) => [...g.variants].sort((a, b) => b.season_rows - a.season_rows || b.rows - a.rows)[0]
export const groupLink = (g: OverviewGroup) => (g.archetype ? `/catalog/${encodeURIComponent(g.archetype)}` : `/market?item=${encodeURIComponent(g.variants[0].name)}`)

function Change({ v }: { v: number | null }) {
  const { t } = useTranslation()
  if (v === null) return <span className="text-xs text-[var(--muted)]" title={t('dashboard.under24h')}>–</span>
  return <span className={`tabular-nums ${v >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}`}>{fmtPct(v)}</span>
}

/** One card per item type: a single quality shows big numbers, several qualities show a compact price table. */
export function TypeCard({ group, states }: { group: OverviewGroup; states: Record<string, 'collecting' | 'waiting'> }) {
  const { t, i18n } = useTranslation()
  const withData = group.variants.filter(v => v.rows > 0)
  const lead = leadVariant(group)
  const busy = group.variants.map(v => states[v.name]).find(Boolean)
  const single = group.variants.length === 1
  const newest = withData.map(v => v.last_ts).filter(Boolean).sort().at(-1)
  return (
    <Link to={groupLink(group)} className="block" aria-label={groupName(group, i18n.language)}>
      <Card className="hover:shadow-md transition h-full">
        <div className="flex items-start gap-3">
          {group.icon_url ? <img src={group.icon_url} alt="" loading="lazy" className="w-10 h-10 object-contain shrink-0" /> : <div className="w-10 h-10 rounded bg-gray-100 shrink-0" />}
          <div className="flex-1 min-w-0">
            <div className="font-semibold truncate">{groupName(group, i18n.language)}</div>
            <div className="text-xs text-[var(--muted)]" title={t('dashboard.rowsHint', { total: group.rows })}>{single ? t('dashboard.seasonRows', { n: group.season_rows, total: group.rows }) : t('dashboard.qualities', { n: group.variants.length, rows: group.season_rows, total: group.rows })}</div>
          </div>
          {busy && <Badge tone={busy === 'collecting' ? 'warn' : 'muted'}>{t(`running.state.${busy}`)}</Badge>}
          {!busy && group.season_rows === 0 && <span title={t('dashboard.pastOnlyHint')}><Badge tone="warn">{t('dashboard.pastOnly')}</Badge></span>}
        </div>
        {single ? (
          <div className="mt-3 flex items-end justify-between">
            <div><div className="text-xs text-[var(--muted)]" title={t('dashboard.latestHint')}>{t('dashboard.latestPrice')}</div><div className="text-2xl font-semibold tabular-nums">{fmtNum(lead.latest)}</div></div>
            <div className="text-right"><div className="text-xs text-[var(--muted)]">{t('dashboard.change24')}</div><Change v={lead.change24} /></div>
          </div>
        ) : (
          <table className="mt-3 w-full text-sm">
            <thead className="text-xs text-[var(--muted)]"><tr><th className="text-left font-normal">{t('dashboard.quality')}</th><th className="text-right font-normal">{t('dashboard.latestPrice')}</th><th className="text-right font-normal">{t('dashboard.change24Short')}</th></tr></thead>
            <tbody>{withData.slice(0, MAX_ROWS).map(v => (
              <tr key={v.name}><td className="py-0.5">{variantLabel(v, i18n.language)}</td><td className="text-right tabular-nums">{fmtNum(v.latest)}</td><td className="text-right"><Change v={v.change24} /></td></tr>))}</tbody>
          </table>
        )}
        {!single && withData.length > MAX_ROWS && <div className="text-xs text-[var(--muted)] mt-1">{t('dashboard.moreQualities', { n: withData.length - MAX_ROWS })}</div>}
        <div className="mt-2"><Sparkline values={lead.spark} /></div>
        <div className="text-xs text-[var(--muted)] mt-1">{single ? t('dashboard.sparkHint') : t('dashboard.sparkOf', { q: variantLabel(lead, i18n.language) })} · {t('dashboard.updated', { when: fmtRelative(newest) })}</div>
      </Card>
    </Link>
  )
}
