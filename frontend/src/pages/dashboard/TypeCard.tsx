import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useOverview } from '../../api/hooks'
import type { OverviewGroup, OverviewVariant, Season } from '../../api/types'
import { inputCls } from '../../components/ui/Field'
import { seasonLabel } from '../../lib/labels'
import { Badge } from '../../components/ui/Badge'
import { Card } from '../../components/ui/Card'
import { PriceNote } from './PriceNote'
import { Sparkline } from '../../components/charts/Sparkline'
import { fmtNum, fmtPct, fmtRelative, fmtDate } from '../../lib/format'

const MAX_ROWS = 4
const STALE_HOURS = 2   // older data says nothing about the next hour: hide the line (the item page explains)

export const groupName = (g: OverviewGroup, lang: string) => (lang.startsWith('zh') && g.name_zh ? g.name_zh : g.name)
export const variantLabel = (v: OverviewVariant, lang: string) => (lang.startsWith('zh') ? v.label : v.label_en)
/** The quality with the most data in the chosen season stands for the type in the sparkline. */
export const leadVariant = (g: OverviewGroup) => [...g.variants].sort((a, b) => b.season_rows - a.season_rows || b.rows - a.rows)[0]
export const groupLink = (g: OverviewGroup) => (g.archetype ? `/item/${encodeURIComponent(g.archetype)}` : `/market?item=${encodeURIComponent(g.variants[0].name)}`)

function Change({ v }: { v: number | null }) {
  const { t } = useTranslation()
  if (v === null) return <span className="text-xs text-[var(--muted)]" title={t('dashboard.under24h')}>–</span>
  return <span className={`tabular-nums ${v >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}`}>{fmtPct(v)}</span>
}

/** One card per item type: a single quality shows big numbers, several qualities show a compact price table. */
/** An item type with no data in a season: its qualities, every number blank. */
const blank = (g: OverviewGroup): OverviewGroup => ({ ...g, season_rows: 0,
  variants: g.variants.map(v => ({ ...v, season_rows: 0, latest: null, change24: null, spark: v.spark.map(() => null), next_hour: null, latest_ts: null, latest_listings: null })) })

/** The card with its own season picker: it opens on the item's newest trading season, any other one on demand. */
export function TypeCard({ group, states, seasons = [] }: { group: OverviewGroup; states: Record<string, 'collecting' | 'waiting'>; seasons?: Season[] }) {
  const { t, i18n } = useTranslation()
  const own = leadVariant(group)?.season_id ?? ''
  const [picked, setPicked] = useState(own)
  const other = useOverview(picked || undefined, false, !!picked && picked !== own)
  const shown = !picked || picked === own ? group : other.data?.find(g => g.key === group.key) ?? blank(group)
  return (
    <div className="relative h-full">
      {seasons.length > 0 && <select aria-label={`${groupName(group, i18n.language)} ${t('common.season')}`} value={picked} onChange={e => setPicked(e.target.value)}
        className={`${inputCls} absolute top-3 right-3 z-10 !py-0.5 !px-1 text-xs`}>
        {!own && <option value="">–</option>}
        {[...seasons].sort((a, b) => b.start.localeCompare(a.start)).map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}</option>)}</select>}
      <CardBody group={shown} states={states} />
    </div>
  )
}

function CardBody({ group, states }: { group: OverviewGroup; states: Record<string, 'collecting' | 'waiting'> }) {
  const { t, i18n } = useTranslation()
  const withData = group.variants.filter(v => v.season_rows > 0)
  const lead = leadVariant(group)
  const busy = group.variants.map(v => states[v.name]).find(Boolean)
  const single = group.variants.length === 1
  const newest = withData.map(v => v.last_ts).filter(Boolean).sort().at(-1)
  return (
    <Link to={groupLink(group)} className="block" aria-label={groupName(group, i18n.language)}>
      <Card className="hover:shadow-md transition h-full">
        <div className="flex items-start gap-3 pr-24">
          {group.icon_url ? <img src={group.icon_url} alt="" loading="lazy" className="w-10 h-10 object-contain shrink-0" /> : <div className="w-10 h-10 rounded bg-gray-100 shrink-0" />}
          <div className="flex-1 min-w-0">
            <div className="font-semibold truncate">{groupName(group, i18n.language)}</div>
            <div className="text-xs text-[var(--muted)]" title={t('dashboard.rowsHint', { total: group.rows })}>{single ? t('dashboard.seasonRows', { n: group.season_rows, total: group.rows }) : t('dashboard.qualities', { n: group.variants.length, rows: group.season_rows, total: group.rows })}</div>
          </div>
          {busy && <Badge tone={busy === 'collecting' ? 'warn' : 'muted'}>{t(`running.state.${busy}`)}</Badge>}
          {group.recent_update && <span title={t('dashboard.updateHint', { date: fmtDate(group.recent_update.date) })}><Badge tone="muted">{group.recent_update.title.replace(/^Early Access /, '')}</Badge></span>}
        </div>
        {single ? (
          <div className="mt-3 flex items-end justify-between">
            <div><div className="text-2xl font-semibold tabular-nums">{fmtNum(lead.latest)}</div><PriceNote v={lead} /></div>
            <div className="text-right"><div className="text-xs text-[var(--muted)]">{t('dashboard.change24')}</div><Change v={lead.change24} /></div>
          </div>
        ) : (
          <table className="mt-3 w-full text-sm">
            <thead className="text-xs text-[var(--muted)]"><tr><th className="text-left font-normal">{t('dashboard.quality')}</th><th className="text-right font-normal">{t('dashboard.latestPrice')}</th><th className="text-right font-normal">{t('dashboard.change24Short')}</th></tr></thead>
            <tbody>{(withData.length ? withData : group.variants).slice(0, MAX_ROWS).map(v => (
              <tr key={v.name}><td className="py-0.5">{variantLabel(v, i18n.language)}</td><td className="text-right tabular-nums">{fmtNum(v.latest)}</td><td className="text-right"><Change v={v.change24} /></td></tr>))}</tbody>
          </table>
        )}
        {!single && <div className="mt-1"><PriceNote v={lead} compact /></div>}
        {!single && withData.length > MAX_ROWS && <div className="text-xs text-[var(--muted)] mt-1">{t('dashboard.moreQualities', { n: withData.length - MAX_ROWS })}</div>}
        {lead.next_hour && (lead.next_hour.stale_hours ?? 0) <= STALE_HOURS && <div className="mt-2 text-xs tabular-nums" title={t('dashboard.nextHourHint')}>{t('dashboard.nextHour', { lo: fmtNum(lead.next_hour.p25), hi: fmtNum(lead.next_hour.p75) })}</div>}
        <div className="mt-2"><Sparkline values={lead.spark} /></div>
        <div className="text-xs text-[var(--muted)] mt-1">{single ? t('dashboard.sparkHint') : t('dashboard.sparkOf', { q: variantLabel(lead, i18n.language) })} · {t('dashboard.updated', { when: fmtRelative(newest) })}</div>
      </Card>
    </Link>
  )
}
