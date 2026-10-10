import { Fragment, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useTrackItem } from '../../api/hooks'
import type { CatalogType, CatalogVariant } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { itemName, paramLabel, rarityLabel, typeLabel } from '../../lib/labels'

function Params({ raw }: { raw: Record<string, unknown> }) {
  const { t } = useTranslation()
  return (
    <dl className="grid grid-cols-[minmax(160px,max-content)_1fr] gap-x-6 gap-y-1 text-xs">
      {Object.entries(raw).map(([k, v]) => (
        <Fragment key={k}>
          <dt className="text-[var(--muted)]" title={k}>{paramLabel(k, t)}{paramLabel(k, t) !== k && <span className="ml-1 font-mono text-[10px] opacity-60">{k}</span>}</dt>
          <dd>{v !== null && typeof v === 'object' ? <pre className="font-mono whitespace-pre-wrap">{JSON.stringify(v, null, 2)}</pre> : String(v)}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

function Variants({ row, tracked }: { row: CatalogType; tracked: Set<string> }) {
  const { t } = useTranslation()
  const track = useTrackItem(); const toast = useToast()
  const [shown, setShown] = useState<string>(row.variants.at(-1)?.id ?? "")  // highest rarity shows the most parameters
  const current = row.variants.find(v => v.id === shown) ?? row.variants[0]
  const trackBtn = (v: CatalogVariant) => tracked.has(v.id) ? <Badge tone="ok">{t('catalog.tracked')}</Badge> :
    <Button size="sm" variant="ghost" onClick={() => track.mutate(v.id, { onSuccess: d => toast.push(t('catalog.trackedOk', { name: d.name })), onError: e => toast.push((e as Error).message, 'error') })}>{t('catalog.track')}</Button>
  return (
    <div className="space-y-3">
      <table className="text-sm"><thead className="text-left text-[var(--muted)]"><tr><th className="pr-6 py-1">{t('catalog.rarity')}</th><th className="pr-6">{t('catalog.gearScore')}</th><th className="pr-6">{t('catalog.vendor')}</th><th></th></tr></thead>
        <tbody>{row.variants.map(v => (
          <tr key={v.id} className={v.id === current?.id ? 'bg-blue-50' : ''}>
            <td className="pr-6 py-1"><button type="button" className="underline decoration-dotted" onClick={() => setShown(v.id)}>{v.rarity ? rarityLabel(v.rarity, t) : v.id}</button></td>
            <td className="pr-6 tabular-nums">{v.gear_score ?? ''}</td><td className="pr-6 tabular-nums">{v.vendor_price ?? ''}</td>
            <td>{!row.has_stats && trackBtn(v)}</td>
          </tr>))}</tbody></table>
      {current && <div><div className="text-xs font-semibold mb-2">{t('catalog.params')} · {rarityLabel(current.rarity, t)}</div><Params raw={current.raw} /></div>}
    </div>
  )
}

export function CatalogTable({ rows, tracked }: { rows: CatalogType[]; tracked: Set<string> }) {
  const { t, i18n } = useTranslation()
  const [open, setOpen] = useState<string | null>(null)
  return (
    <div className="overflow-auto border border-[var(--line)] rounded-lg">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-left text-[var(--muted)]"><tr>
          <th className="px-3 py-2 w-12"></th><th className="px-3 py-2">{t('catalog.name')}</th><th className="px-3 py-2">{t('catalog.type')}</th>
          <th className="px-3 py-2">{t('catalog.rarities')}</th><th className="px-3 py-2 text-right">{t('catalog.listings')}</th><th className="px-3 py-2"></th>
        </tr></thead>
        <tbody>{rows.map(r => {
          const single = r.variants.length === 1 ? r.variants[0] : null
          return (
            <Fragment key={r.archetype}>
              <tr className="border-t border-[var(--line)] hover:bg-gray-50">
                <td className="px-3 py-1.5">{r.icon_url && <img src={r.icon_url} alt="" width={32} height={32} loading="lazy" onError={e => { (e.target as HTMLImageElement).style.visibility = 'hidden' }} />}</td>
                <td className="px-3 py-1.5"><button type="button" className="font-medium text-left hover:underline" onClick={() => setOpen(o => (o === r.archetype ? null : r.archetype))}>{itemName(r, i18n.language)}</button>
                  {r.has_stats && <span className="ml-2"><Badge tone="warn">{t('catalog.withStats')}</Badge></span>}</td>
                <td className="px-3 py-1.5">{typeLabel(r.item_type, t)}</td>
                <td className="px-3 py-1.5 text-xs text-[var(--muted)]">{r.rarities.map(x => rarityLabel(x, t)).join(' · ')}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{r.listings_total ?? '–'}</td>
                <td className="px-3 py-1.5 text-right">
                  {r.has_stats ? <Link to={`/equipment?type=${r.archetype}`} className="text-sm font-medium text-[var(--accent)] hover:underline">{t('catalog.viewMarket')}</Link>
                    : single && tracked.has(single.id) ? <Badge tone="ok">{t('catalog.tracked')}</Badge> : null}
                </td>
              </tr>
              {open === r.archetype && <tr className="bg-gray-50"><td></td><td colSpan={5} className="px-3 py-3"><Variants row={r} tracked={tracked} /></td></tr>}
            </Fragment>)
        })}</tbody>
      </table>
    </div>
  )
}
