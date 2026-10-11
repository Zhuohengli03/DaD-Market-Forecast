import { Fragment } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import type { CatalogType } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { seasonFirstLabel } from '../../lib/season'
import { itemName, rarityLabel, typeLabel } from '../../lib/labels'

/** localSeason: this-season listings collected on this machine for tracked items (item id → count).
 *  marketOpens: the current season's trading-open date while it is still ahead (no listing can be "this season" yet). */
export function CatalogTable({ rows, tracked, localSeason = {}, marketOpens }: { rows: CatalogType[]; tracked: Set<string>; localSeason?: Record<string, number>; marketOpens?: string | null }) {
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const go = (arch: string) => navigate(`/item/${encodeURIComponent(arch)}`)
  return (
    <div className="overflow-auto border border-[var(--line)] rounded-lg">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-left text-[var(--muted)]"><tr>
          <th className="px-3 py-2 w-12"></th><th className="px-3 py-2">{t('catalog.name')}</th><th className="px-3 py-2">{t('catalog.type')}</th>
          <th className="px-3 py-2">{t('catalog.rarities')}</th><th className="px-3 py-2 text-right" title={t('catalog.seasonListingsHint')}>{t('catalog.seasonListings')}</th><th className="px-3 py-2 whitespace-nowrap" title={t('catalog.seasonFirstHint')}>{t('catalog.seasonFirst')}</th><th className="px-3 py-2 text-right" title={t('catalog.listingsHint')}>{t('catalog.listings')}</th><th className="px-3 py-2"></th>
        </tr></thead>
        <tbody>{rows.map(r => {
          const single = r.variants.length === 1 ? r.variants[0] : null
          return (
            <Fragment key={r.archetype}>
              <tr className="border-t border-[var(--line)] hover:bg-gray-50 cursor-pointer" onClick={() => go(r.archetype)}>
                <td className="px-3 py-1.5">{r.icon_url && <img src={r.icon_url} alt="" loading="lazy" className="w-8 h-8 object-contain" onError={e => { (e.target as HTMLImageElement).style.visibility = 'hidden' }} />}</td>
                <td className="px-3 py-1.5"><Link to={`/item/${encodeURIComponent(r.archetype)}`} className="font-medium hover:underline" onClick={e => e.stopPropagation()}>{itemName(r, i18n.language)}</Link>
                  {r.has_stats && <span className="ml-2"><Badge tone="warn">{t('catalog.withStats')}</Badge></span>}</td>
                <td className="px-3 py-1.5">{typeLabel(r.item_type, t)}</td>
                <td className="px-3 py-1.5 text-xs text-[var(--muted)]">{r.rarities.map(x => rarityLabel(x, t)).join(' · ')}</td>
                <td className="px-3 py-1.5 text-right tabular-nums font-medium">{(() => {
                  if (marketOpens) return <span className="text-xs font-normal text-[var(--muted)]">{t('catalog.marketOpens', { date: marketOpens })}</span>
                  if (r.season_listings != null) return r.season_listings
                  const local = r.variants.filter(v => v.id in localSeason)
                  return local.length ? <span title={t('catalog.localHint')}>{local.reduce((n, v) => n + localSeason[v.id], 0)}<span className="ml-1 text-[10px] text-[var(--muted)]">{t('catalog.local')}</span></span> : '–'
                })()}</td>
                <td className="px-3 py-1.5 tabular-nums text-[var(--muted)] whitespace-nowrap">{seasonFirstLabel(r, t)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums text-[var(--muted)]">{r.listings_total ?? '–'}</td>
                <td className="px-3 py-1.5 text-right">
                  {r.has_stats ? <Link to={`/equipment?type=${r.archetype}`} onClick={e => e.stopPropagation()} className="text-sm font-medium text-[var(--accent)] hover:underline">{t('catalog.viewMarket')}</Link>
                    : single && tracked.has(single.id) ? <Badge tone="ok">{t('catalog.tracked')}</Badge> : null}
                </td>
              </tr>
            </Fragment>)
        })}</tbody>
      </table>
    </div>
  )
}
