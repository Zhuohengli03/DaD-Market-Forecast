import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItems, useItemsEstimate, useTrackBulk } from '../../api/hooks'
import { useDebounced } from '../../lib/useDebounced'
import { fmtCount } from '../../lib/quota'
import { ItemPicker, usePickOptions } from '../../components/ItemPicker'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'

/** Track many items without random stats at once; items already tracked are left out of the list. */
export function AddItemsDialog({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation()
  const items = useItems(); const bulk = useTrackBulk(); const toast = useToast()
  const { options, loading } = usePickOptions()
  const tracked = new Set((items.data ?? []).flatMap(i => [i.item_id, i.name]).filter(Boolean) as string[])
  const fresh = options.filter(o => !tracked.has(o.id))
  const [chosen, setChosen] = useState<string[]>([])
  const estimate = useItemsEstimate(useDebounced(chosen, 400), 200)
  const add = () => bulk.mutate(chosen, {
    onSuccess: r => { toast.push(t('collect.added', { n: r.added })); onClose() },
    onError: e => toast.push((e as Error).message, 'error'),
  })
  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center p-4" onKeyDown={e => { if (e.key === 'Escape') onClose() }}>
      <div role="dialog" aria-modal="true" aria-labelledby="add-items-title" className="bg-white rounded-xl shadow-xl max-w-2xl w-full p-6 space-y-4 max-h-[90vh] overflow-auto">
        <div><h2 id="add-items-title" className="text-lg font-semibold">{t('collect.addTitle')}</h2>
          <p className="text-sm text-[var(--muted)]">{t('collect.addIntro', { n: fresh.length })}</p></div>
        {loading ? <p className="text-sm">{t('common.loading')}</p> : !fresh.length ? <p className="text-sm">{options.length ? t('collect.allTracked') : t('collect.needCatalog')}</p>
          : <ItemPicker options={fresh} chosen={chosen} onChange={setChosen} height="max-h-[50vh]" />}
        <div className="flex justify-end items-center gap-2 border-t border-[var(--line)] pt-4">
          {chosen.length > 0 && <span className="mr-auto text-xs text-[var(--muted)]">{t('collect.addCost', { n: fmtCount(estimate.data?.requests ?? chosen.length * 201) })}</span>}
          <Button variant="ghost" onClick={onClose}>{t('common.cancel')}</Button>
          <Button disabled={!chosen.length} loading={bulk.isPending} onClick={add}>{t('collect.addN', { n: chosen.length })}</Button>
        </div>
      </div>
    </div>
  )
}
