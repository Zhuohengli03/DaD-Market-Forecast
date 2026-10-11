import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useParams } from 'react-router-dom'
import { useAnnouncements, useCatalogType, useItemAnnouncements, useItems, useSeasons } from '../../api/hooks'
import { Card } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import { rarityLabel, seasonLabel } from '../../lib/labels'
import { useUrlState } from '../../lib/urlState'
import { ListingStats, Params, VariantTable } from '../catalog/ItemDetail'
import { QualityPrices } from '../catalog/QualityPrices'
import { ListingsTable } from '../market/ListingsTable'
import { NextHourRange } from '../market/NextHourRange'
import { ItemHeader } from './ItemHeader'
import { PriceSection } from './PriceSection'
import { SeasonCompare } from './SeasonCompare'
import { RelatedUpdates } from './RelatedUpdates'
import { ItemForecast } from './ItemForecast'

/** Everything about one item type on one page: price by season, next hour, seasons compared, updates, facts. */
export default function ItemPage() {
  const { t } = useTranslation()
  const { archetype = '' } = useParams()
  const q = useCatalogType(archetype); const items = useItems(); const seasons = useSeasons()
  const [qParam, setQ] = useUrlState('q', '')
  const [seasonParam, setSeason] = useUrlState('season', '')
  const [focusTs, setFocusTs] = useState<string | null>(null)
  const patches = useAnnouncements().data?.items ?? []
  const row = q.data
  const mentions = useItemAnnouncements(archetype).data?.mentions ?? []
  if (q.isLoading) return <p className="text-sm">{t('common.loading')}</p>
  if (!row) return <EmptyState title={t('catalog.detail.missing')} />
  const ids = new Set(row.variants.map(v => v.id))
  const tracked = (items.data ?? []).filter(i => i.item_id && ids.has(i.item_id))
  const item = tracked.find(i => i.name === qParam) ?? [...tracked].sort((a, b) => b.rows - a.rows)[0]
  const allSeasons = seasons.data ?? []
  const byStart = [...allSeasons].sort((a, b) => a.start.localeCompare(b.start))
  const latest = byStart.at(-1)
  // open on the newest season whose market has opened: a new season trades only after its solo weeks
  const trading = byStart.filter(s => Date.parse(s.market_open ? `${s.market_open}T00:00:00Z` : s.start) <= Date.now()).at(-1) ?? latest
  const season = allSeasons.some(s => s.id === seasonParam) ? seasonParam : trading?.id ?? ''
  const isLatest = season === latest?.id
  const seasonName = seasonLabel(allSeasons.find(s => s.id === season)?.name, t)
  const trackedIds = new Set((items.data ?? []).map(i => i.item_id ?? i.name))
  const collected = tracked.filter(i => i.rows > 0)
  const localSeason = collected.some(i => i.season_rows != null) ? collected.reduce((n, i) => n + (i.season_rows ?? 0), 0) : null
  const top = row.variants.at(-1)
  // every patch as a line on the chart; the ones that name this item stand out
  const named = new Set(mentions.map(m => m.id))
  const marks = patches.map(a => ({ ts: a.date, label: a.title.replace(/^Early Access /, ''), strong: named.has(a.id) }))
  const pick = (ts: string) => {
    const owner = [...allSeasons].sort((a, b) => b.start.localeCompare(a.start)).find(s => s.start <= ts)
    if (owner && owner.id !== season) setSeason(owner.id)   // an update from another season opens that season
    setFocusTs(ts)
  }
  return (
    <div className="space-y-4">
      <ItemHeader row={row} tracked={tracked} item={item} onItem={setQ} season={season} latestSeason={isLatest} seasonName={seasonName} seasons={allSeasons} onSeason={setSeason} />
      {item && <PriceSection item={item.name} seasons={allSeasons} season={season} marks={marks} focusTs={focusTs} />}
      {item && isLatest && <NextHourRange item={item.name} seasonId={season || undefined} />}
      {item && <SeasonCompare item={item.name} />}
      {item && <RelatedUpdates archetype={row.archetype} item={item.name} onPick={pick} />}
      {item && <ItemForecast item={item.name} season={season} />}
      {!item && <Card title={t('catalog.detail.rarities')}><VariantTable row={row} tracked={trackedIds} shown={top?.id ?? ''} onShow={() => {}} /></Card>}
      <details className="bg-[var(--panel)] border border-[var(--line)] rounded-xl p-4" open={!item}>
        <summary className="cursor-pointer font-semibold text-sm">{t('item.facts')}</summary>
        <div className="mt-3 space-y-4">
          <ListingStats row={row} localSeason={localSeason} />
          <QualityPrices archetype={row.archetype} season={season} />
          {item && <VariantTable row={row} tracked={trackedIds} shown={top?.id ?? ''} onShow={() => {}} />}
          {top && <div><div className="text-sm font-semibold mb-2">{t('catalog.params')} · {rarityLabel(top.rarity, t)}</div><Params raw={top.raw} /></div>}
        </div>
      </details>
      {item && <details className="bg-[var(--panel)] border border-[var(--line)] rounded-xl p-4">
        <summary className="cursor-pointer font-semibold text-sm">{t('item.listings')}</summary>
        <div className="mt-3"><ListingsTable item={item.name} season_id={season} onPick={() => {}} /></div>
      </details>}
    </div>
  )
}
