import { Fragment, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useTrackItem } from '../../api/hooks'
import type { CatalogRow } from '../../api/types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { fmtNum } from '../../lib/format'

const ICON = (id: string) => `https://api.darkerdb.com/v2/items/${encodeURIComponent(id)}/icon?size=48&dpr=2`
const typeOf = (r: CatalogRow) => r.item_type || r.slot_type || r.weapon_type || r.armor_type || r.hand_type || ''

function Params({ raw }: { raw: Record<string, unknown> }) {
  return (
    <dl className="grid grid-cols-[minmax(140px,max-content)_1fr] gap-x-6 gap-y-1 text-xs">
      {Object.entries(raw).map(([k, v]) => (
        <Fragment key={k}>
          <dt className="text-[var(--muted)] font-mono">{k}</dt>
          <dd>{v !== null && typeof v === 'object' ? <pre className="font-mono whitespace-pre-wrap">{JSON.stringify(v, null, 2)}</pre> : String(v)}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

export function CatalogTable({ rows, tracked }: { rows: CatalogRow[]; tracked: Set<string> }) {
  const { t } = useTranslation()
  const track = useTrackItem()
  const toast = useToast()
  const [open, setOpen] = useState<string | null>(null)
  return (
    <div className="overflow-auto border border-[var(--line)] rounded-lg">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-left text-[var(--muted)]"><tr>
          <th className="px-3 py-2 w-12"></th><th className="px-3 py-2">{t('catalog.name')}</th><th className="px-3 py-2">{t('catalog.rarity')}</th><th className="px-3 py-2">{t('catalog.type')}</th>
          <th className="px-3 py-2 text-right">{t('catalog.gearScore')}</th><th className="px-3 py-2 text-right">{t('catalog.vendor')}</th>
          <th className="px-3 py-2 text-right">{t('catalog.listings')}</th><th className="px-3 py-2 text-right">{t('catalog.low')}</th><th className="px-3 py-2 text-right">{t('catalog.median')}</th><th className="px-3 py-2"></th>
        </tr></thead>
        <tbody>{rows.map(r => (
          <Fragment key={r.id}>
            <tr className="border-t border-[var(--line)] hover:bg-gray-50">
              <td className="px-3 py-1.5"><img src={ICON(r.id)} alt="" width={32} height={32} loading="lazy" onError={e => { (e.target as HTMLImageElement).style.visibility = 'hidden' }} /></td>
              <td className="px-3 py-1.5"><button type="button" className="font-medium text-left hover:underline" onClick={() => setOpen(o => (o === r.id ? null : r.id))}>{r.name}</button><div className="text-xs text-[var(--muted)] font-mono">{r.id}</div></td>
              <td className="px-3 py-1.5">{r.rarity && <Badge>{r.rarity}</Badge>}</td>
              <td className="px-3 py-1.5">{typeOf(r)}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.gear_score ?? ''}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.vendor_price ?? ''}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.market?.listings ?? '–'}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{r.market ? fmtNum(r.market.min_price_per_unit) : '–'}</td>
              <td className="px-3 py-1.5 text-right tabular-nums font-medium">{r.market ? fmtNum(r.market.median_price_per_unit) : '–'}</td>
              <td className="px-3 py-1.5 text-right">{tracked.has(r.id) ? <Badge tone="ok">{t('catalog.tracked')}</Badge> :
                <Button size="sm" variant="ghost" onClick={() => track.mutate(r.id, { onSuccess: d => toast.push(t('catalog.trackedOk', { name: d.name })), onError: e => toast.push((e as Error).message, 'error') })}>{t('catalog.track')}</Button>}</td>
            </tr>
            {open === r.id && <tr className="bg-gray-50"><td></td><td colSpan={9} className="px-3 py-3"><div className="text-xs font-semibold mb-2">{t('catalog.params')}</div><Params raw={r.raw} /></td></tr>}
          </Fragment>))}</tbody>
      </table>
    </div>
  )
}
