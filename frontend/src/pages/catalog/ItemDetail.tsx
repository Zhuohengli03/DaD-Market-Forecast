import { Fragment, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { ArrowLeft } from 'lucide-react'
import { useCatalogType, useItems, useTrackItem, useTypeHistory } from '../../api/hooks'
import type { CatalogType, CatalogVariant } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import { useToast } from '../../components/ui/Toast'
import { fmtDate } from '../../lib/format'
import { QualityPrices } from './QualityPrices'
import { itemName, paramLabel, paramValue, rarityLabel, typeLabel } from '../../lib/labels'

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

function ListingStats({ row, localSeason }: { row: CatalogType; localSeason: number | null }) {
  const { t } = useTranslation()
  const history = useTypeHistory(); const toast = useToast()
  // looked up: the season count is known, and the first listing too (or there never was one)
  const known = row.season_listings != null && (!!row.first_listed || row.listings_total === 0)
  const season = row.season_listings ?? localSeason
  const first = row.first_listed ? fmtDate(row.first_listed) : known ? t('catalog.noListing') : '–'
  return (
    <section aria-label={t('catalog.detail.listings')} className="lg:border-l lg:border-[var(--line)] lg:pl-6 min-w-[300px]">
      <div className="text-sm font-semibold mb-2">{t('catalog.detail.listings')}</div>
      <dl className="grid grid-cols-3 gap-4 text-sm">
        <div><dt className="text-[var(--muted)]">{t('catalog.seasonListings')}</dt>
          <dd className="text-xl font-semibold tabular-nums">{season ?? '–'}{row.season_listings == null && localSeason != null && <span className="ml-1 text-xs font-normal text-[var(--muted)]">{t('catalog.local')}</span>}</dd></div>
        <div><dt className="text-[var(--muted)]">{t('catalog.listings')}</dt><dd className="text-xl font-semibold tabular-nums">{row.listings_total ?? '–'}</dd></div>
        <div><dt className="text-[var(--muted)]">{t('catalog.firstListed')}</dt><dd className="text-xl font-semibold tabular-nums">{first}</dd></div>
      </dl>
      <p className="mt-2 text-xs text-[var(--muted)] max-w-md">{t('catalog.listingsHint')}</p>
      {!known && <Button className="mt-2" size="sm" variant="ghost" loading={history.isPending}
        onClick={() => history.mutate(row.archetype, { onSuccess: d => toast.push(t('catalog.detail.fetched', { n: d.credits_used ?? 0 })), onError: e => toast.push((e as Error).message, 'error') })}>{t('catalog.fetchHistory')}</Button>}
    </section>
  )
}

function VariantTable({ row, tracked, shown, onShow }: { row: CatalogType; tracked: Set<string>; shown: string; onShow: (id: string) => void }) {
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
export default function ItemDetail() {
  const { t, i18n } = useTranslation()
  const { archetype = '' } = useParams()
  const q = useCatalogType(archetype); const items = useItems()
  const row = q.data
  const [picked, setPicked] = useState<string | null>(null)
  if (q.isLoading) return <p className="text-sm">{t('common.loading')}</p>
  if (!row) return <div className="space-y-3"><Link to="/catalog" className="text-sm text-[var(--accent)]">← {t('catalog.detail.back')}</Link><EmptyState title={t('catalog.detail.missing')} /></div>
  const tracked = new Set((items.data ?? []).map(i => i.item_id ?? i.name))
  const ids = new Set(row.variants.map(v => v.id))
  const mine = (items.data ?? []).filter(i => i.item_id && ids.has(i.item_id))
  const collected = mine.filter(i => i.rows > 0)
  const localSeason = collected.some(i => i.season_rows != null) ? collected.reduce((n, i) => n + (i.season_rows ?? 0), 0) : null
  const shown = picked ?? row.variants.at(-1)?.id ?? ''   // highest rarity shows the most parameters
  const current = row.variants.find(v => v.id === shown) ?? row.variants[0]
  const zh = i18n.language.startsWith('zh')
  return (
    <div className="space-y-4">
      <Link to="/catalog" className="inline-flex items-center gap-1 text-sm text-[var(--accent)] hover:underline"><ArrowLeft size={14} />{t('catalog.detail.back')}</Link>
      <Card>
        <div className="flex flex-wrap items-start gap-4">
          {current?.icon_url || row.icon_url ? <img src={current?.icon_url || row.icon_url!} alt="" className="w-24 h-24 object-contain rounded-lg bg-gray-50" /> : <div className="w-24 h-24 rounded-lg bg-gray-100" />}
          <div className="flex-1 min-w-[260px] space-y-1">
            <h1 className="text-xl font-semibold">{itemName(row, i18n.language)}{zh && row.name_zh && <span className="ml-2 text-sm font-normal text-[var(--muted)]">{row.name}</span>}</h1>
            <div className="flex flex-wrap gap-2 text-xs">
              <Badge>{row.category && row.category !== row.item_type ? t(`category.${row.category}`, { defaultValue: typeLabel(row.item_type, t) }) : typeLabel(row.item_type, t)}</Badge>
              {row.has_stats ? <Badge tone="warn">{t('catalog.withStats')}</Badge> : <Badge tone="muted">{t('catalog.noStats')}</Badge>}
            </div>
            <div className="text-xs text-[var(--muted)]">{t('catalog.rarity')}：{row.rarities.map(x => rarityLabel(x, t)).join(' · ')}</div>
            <div className="text-xs text-[var(--muted)] font-mono">{row.archetype}</div>
            {row.has_stats && <Link to={`/equipment?type=${row.archetype}`} className="inline-block text-sm font-medium text-[var(--accent)] hover:underline">{t('catalog.viewMarket')}</Link>}
          </div>
          <ListingStats row={row} localSeason={localSeason} />
        </div>
      </Card>
      <QualityPrices archetype={row.archetype} />
      <Card title={t('catalog.detail.rarities')}>
        {row.has_stats && <p className="text-xs text-[var(--muted)] mb-2">{t('catalog.detail.statsNote')}</p>}
        <VariantTable row={row} tracked={tracked} shown={shown} onShow={setPicked} />
      </Card>
      {current && <Card title={`${t('catalog.params')} · ${rarityLabel(current.rarity, t)}`}><Params raw={current.raw} /></Card>}
    </div>
  )
}
