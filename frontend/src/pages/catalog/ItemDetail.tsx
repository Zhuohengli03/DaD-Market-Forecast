import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { useCatalogMeta, useTrackItem, useTypeHistory } from '../../api/hooks'
import type { CatalogType, CatalogVariant } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { seasonFirstLabel, seasonLookedUp } from '../../lib/season'
import { paramLabel, paramValue, rarityLabel } from '../../lib/labels'

// in Chinese, list the Chinese name and description before the English ones
const ZH_FIRST = ['name_zh', 'flavor_zh']

export function Params({ raw }: { raw: Record<string, unknown> }) {
  const { t, i18n } = useTranslation()
  const entries = Object.entries(raw)
  const ordered = i18n.language.startsWith('zh')
    ? [...entries.filter(([k]) => ZH_FIRST.includes(k)), ...entries.filter(([k]) => !ZH_FIRST.includes(k))]
    : entries
  return (
    <dl className="grid grid-cols-[minmax(160px,max-content)_1fr] gap-x-6 gap-y-1 text-sm">
      {ordered.map(([k, v]) => (
        <Fragment key={k}>
          <dt className="text-[var(--muted)]" title={k}>{paramLabel(k, t)}{paramLabel(k, t) !== k && <span className="ml-1 font-mono text-[10px] opacity-60">{k}</span>}</dt>
          <dd className="break-words">{v !== null && typeof v === 'object' ? <pre className="font-mono text-xs whitespace-pre-wrap">{JSON.stringify(v, null, 2)}</pre> : paramValue(k, v, t)}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

export function ListingStats({ row, localSeason }: { row: CatalogType; localSeason: number | null }) {
  const { t } = useTranslation()
  const history = useTypeHistory(); const toast = useToast()
  const opens = useCatalogMeta().data?.market_opens   // the season's market is still closed: nothing to look up yet
  // looked up: the season count is known, and the first listing too (or there never was one)
  const known = seasonLookedUp(row)
  const season = row.season_listings ?? localSeason
  return (
    <section aria-label={t('catalog.detail.listings')} className="lg:border-l lg:border-[var(--line)] lg:pl-6 min-w-[300px]">
      <div className="text-sm font-semibold mb-2">{t('catalog.detail.listings')}</div>
      <dl className="grid grid-cols-3 gap-4 text-sm">
        <div><dt className="text-[var(--muted)]">{t('catalog.seasonListings')}</dt>
          <dd className="text-xl font-semibold tabular-nums">{opens ? <span className="text-sm font-normal text-[var(--muted)]">{t('catalog.detail.marketClosed', { date: opens })}</span> : season ?? '–'}{!opens && row.season_listings == null && localSeason != null && <span className="ml-1 text-xs font-normal text-[var(--muted)]">{t('catalog.local')}</span>}</dd></div>
        <div><dt className="text-[var(--muted)]" title={t('catalog.seasonFirstHint')}>{t('catalog.seasonFirst')}</dt><dd className="text-xl font-semibold tabular-nums">{seasonFirstLabel(row, t)}</dd></div>
        <div><dt className="text-[var(--muted)]">{t('catalog.listings')}</dt><dd className="text-xl font-semibold tabular-nums">{row.listings_total ?? '–'}</dd></div>
      </dl>
      <p className="mt-2 text-xs text-[var(--muted)] max-w-md">{t('catalog.listingsHint')}</p>
      {/* always available: this season's numbers keep growing, so a lookup can be refreshed */}
      {!opens && <Button className="mt-2" size="sm" variant="ghost" loading={history.isPending}
        onClick={() => history.mutate({ archetype: row.archetype, refresh: known }, { onSuccess: d => toast.push(t('catalog.detail.fetched', { n: d.credits_used ?? 0 })), onError: e => toast.push((e as Error).message, 'error') })}>{known ? t('catalog.refetchHistory') : t('catalog.fetchHistory')}</Button>}
    </section>
  )
}

export function VariantTable({ row, tracked, shown, onShow }: { row: CatalogType; tracked: Set<string>; shown: string; onShow: (id: string) => void }) {
  const { t } = useTranslation()
  const track = useTrackItem(); const toast = useToast()
  const gs = row.variants.some(v => v.gear_score != null), vp = row.variants.some(v => v.vendor_price != null)
  const trackBtn = (v: CatalogVariant) => tracked.has(v.id) ? <Badge tone="ok">{t('catalog.tracked')}</Badge> :
    <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); track.mutate(v.id, { onSuccess: d => toast.push(t('catalog.trackedOk', { name: d.name })), onError: err => toast.push((err as Error).message, 'error') }) }}>{t('catalog.track')}</Button>
  return (
    <table className="w-full text-sm"><thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('catalog.rarity')}</th>{gs && <th>{t('catalog.gearScore')}</th>}{vp && <th>{t('catalog.vendor')}</th>}<th></th></tr></thead>
      <tbody>{row.variants.map(v => (
        <tr key={v.id} className={`border-t border-[var(--line)] cursor-pointer ${v.id === shown ? 'bg-blue-50' : 'hover:bg-gray-50'}`} onClick={() => onShow(v.id)}>
          <td className="py-1.5">{v.rarity ? rarityLabel(v.rarity, t) : v.id}</td>
          {gs && <td className="tabular-nums">{v.gear_score ?? ''}</td>}{vp && <td className="tabular-nums">{v.vendor_price ?? ''}</td>}
          <td className="text-right">{!row.has_stats && trackBtn(v)}</td>
        </tr>))}</tbody></table>
  )
}

/** Everything about one item type: names, listings, every rarity and all of its DarkerDB parameters. */
