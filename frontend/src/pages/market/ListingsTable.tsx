import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useListings } from '../../api/hooks'
import { Button } from '../../components/ui/Button'
import { EmptyState } from '../../components/ui/EmptyState'
import { inputCls } from '../../components/ui/Field'
import { fmtNum, fmtTs } from '../../lib/format'

interface Props { item: string; season_id?: string; start?: string; end?: string; onPick: (ts: string) => void }
const PAGE = 50

export function ListingsTable({ item, season_id, start, end, onPick }: Props) {
  const { t } = useTranslation()
  const [page, setPage] = useState(1)
  const [sold, setSold] = useState<'' | 'true' | 'false'>('')
  const [minQty, setMinQty] = useState(''); const [maxQty, setMaxQty] = useState('')
  useEffect(() => { setPage(1) }, [item, season_id, start, end])
  const q = useListings(item, { season_id: season_id || undefined, start: start || undefined, end: end || undefined, sold: sold === '' ? undefined : sold === 'true', min_qty: minQty ? Number(minQty) : undefined, max_qty: maxQty ? Number(maxQty) : undefined, page, page_size: PAGE })
  const total = q.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2 items-center text-sm">
        <select className={inputCls} value={sold} onChange={e => { setSold(e.target.value as typeof sold); setPage(1) }}><option value="">{t('market.sold')}: {t('common.all')}</option><option value="true">{t('common.yes')}</option><option value="false">{t('common.no')}</option></select>
        <input className={inputCls} placeholder={t('market.minQty')} type="number" min={0} value={minQty} onChange={e => { setMinQty(e.target.value); setPage(1) }} />
        <input className={inputCls} placeholder={t('market.maxQty')} type="number" min={0} value={maxQty} onChange={e => { setMaxQty(e.target.value); setPage(1) }} />
        <span className="ml-auto text-[var(--muted)]">{t('market.page', { page, pages, total })}</span>
        <Button variant="ghost" size="sm" aria-label={t('common.prev')} disabled={page <= 1} onClick={() => setPage(p => p - 1)}>‹</Button>
        <Button variant="ghost" size="sm" aria-label={t('common.next')} disabled={page >= pages} onClick={() => setPage(p => p + 1)}>›</Button>
      </div>
      {q.isLoading ? <p className="text-sm">{t('common.loading')}</p> : !q.data?.rows.length ? <EmptyState title={t('market.emptyListings')} /> :
        <div className="overflow-auto border border-[var(--line)] rounded-lg">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-left text-[var(--muted)]"><tr><th className="px-3 py-2">{t('market.time')}</th><th className="px-3 py-2">{t('market.qty')}</th><th className="px-3 py-2">{t('market.price')}</th><th className="px-3 py-2">{t('market.total')}</th><th className="px-3 py-2">{t('market.sold')}</th></tr></thead>
            <tbody>{q.data.rows.map((r, i) => (
              <tr key={`${r.item_id}-${r.created_at}-${i}`} className="border-t border-[var(--line)] hover:bg-blue-50 cursor-pointer" onClick={() => onPick(r.created_at)}>
                <td className="px-3 py-1.5">{fmtTs(r.created_at)}</td><td className="px-3 py-1.5">{r.quantity}</td><td className="px-3 py-1.5 font-medium">{fmtNum(r.price_per_unit)}</td><td className="px-3 py-1.5">{fmtNum(r.price)}</td><td className="px-3 py-1.5">{r.has_sold ? '✓' : ''}</td>
              </tr>))}</tbody>
          </table>
        </div>}
    </div>
  )
}
