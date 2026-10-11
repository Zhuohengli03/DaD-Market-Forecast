import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { useOverview } from '../../api/hooks'
import type { CatalogType, ItemInfo, Season } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Card } from '../../components/ui/Card'
import { inputCls } from '../../components/ui/Field'
import { fmtNum, fmtPct } from '../../lib/format'
import { itemName, seasonLabel, typeLabel } from '../../lib/labels'
import { RefreshData } from '../market/RefreshData'
import { variantLabel } from '../dashboard/TypeCard'
import { PriceNote } from '../dashboard/PriceNote'

/** Name, quality switch, latest price, and the two actions: get data and one-click forecast. */
export function ItemHeader({ row, tracked, item, onItem, season, latestSeason, seasonName, seasons = [], onSeason }: {
  row: CatalogType; tracked: ItemInfo[]; item: ItemInfo | undefined; onItem: (name: string) => void
  season: string; latestSeason: boolean; seasonName: string; seasons?: Season[]; onSeason?: (id: string) => void }) {
  const { t, i18n } = useTranslation()
  const group = useOverview(season || undefined).data?.find(g => g.archetype === row.archetype)
  const v = group?.variants.find(x => x.name === item?.name)
  return (
    <Card>
      <div className="flex flex-wrap items-start gap-4">
        {row.icon_url ? <img src={row.icon_url} alt="" className="w-20 h-20 object-contain rounded-lg bg-gray-50" /> : <div className="w-20 h-20 rounded-lg bg-gray-100" />}
        <div className="flex-1 min-w-[240px] space-y-2">
          <h1 className="text-xl font-semibold">{itemName(row, i18n.language)}</h1>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge>{row.category && row.category !== row.item_type ? t(`category.${row.category}`, { defaultValue: typeLabel(row.item_type, t) }) : typeLabel(row.item_type, t)}</Badge>
            {row.has_stats && <Link to={`/equipment?type=${row.archetype}`} className="text-[var(--accent)] hover:underline">{t('catalog.viewMarket')}</Link>}
          </div>
          {/* the one season picker of the page: price, data fetching, next hour and listings follow it */}
          {item && onSeason && seasons.length > 0 && <select aria-label={t('common.season')} className={`${inputCls} mr-2`} value={season} onChange={e => onSeason(e.target.value)}>
            {[...seasons].sort((a, b) => b.start.localeCompare(a.start)).map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}</option>)}</select>}
          {tracked.length > 1 && <select aria-label={t('item.quality')} className={inputCls} value={item?.name ?? ''} onChange={e => onItem(e.target.value)}>
            {tracked.map(i => { const gv = group?.variants.find(x => x.name === i.name); return <option key={i.name} value={i.name}>{gv ? variantLabel(gv, i18n.language) : i.name}</option> })}</select>}
        </div>
        {item && <div className="text-right space-y-1">
          <div className="text-2xl font-semibold tabular-nums">{fmtNum(v?.latest)}</div>
          {v && <PriceNote v={v} />}
          <div className="text-xs text-[var(--muted)]">{t('dashboard.change24')} <span className={(v?.change24 ?? 0) >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}>{fmtPct(v?.change24)}</span></div>
          <div className="text-xs text-[var(--muted)]">{t('dashboard.seasonRows', { n: v?.season_rows ?? item.season_rows ?? 0 })}</div>
        </div>}
      </div>
      {item && <div className="mt-3 flex flex-wrap items-start justify-end gap-3">
        <RefreshData item={item.name} seasonId={season || undefined} latest={latestSeason} seasonName={seasonName} />
      </div>}
    </Card>
  )
}
