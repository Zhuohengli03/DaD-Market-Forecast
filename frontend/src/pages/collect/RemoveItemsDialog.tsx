import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItems, useUntrackBulk } from '../../api/hooks'
import type { CatalogType } from '../../api/types'
import { ItemPicker, usePickOptions, type PickOption } from '../../components/ItemPicker'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { itemName } from '../../lib/labels'

/** Stop tracking many items at once, with the same shortcuts as adding; collected listings are kept. */
export function RemoveItemsDialog({ onClose }: { onClose: () => void }) {
  const { t, i18n } = useTranslation()
  const items = useItems(); const remove = useUntrackBulk(); const toast = useToast()
  const { options: catalog } = usePickOptions()
  const options = useMemo<PickOption[]>(() => {
    const byId = new Map(catalog.map(o => [o.id, o]))
    return (items.data ?? []).map(i => {
      const o = i.item_id ? byId.get(i.item_id) : undefined
      // option ids are tracked names here, since that is what the server removes by
      return { id: i.name, type: o?.type ?? ({ name: i.name } as CatalogType), category: o?.category ?? 'other', label: o?.label ?? itemName(i, i18n.language) }
    })
  }, [items.data, catalog, i18n.language])
  const [chosen, setChosen] = useState<string[]>([])
  const submit = () => remove.mutate(chosen, {
    onSuccess: r => { toast.push(t('collect.removed', { n: r.removed })); onClose() },
    onError: e => toast.push((e as Error).message, 'error'),
  })
  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center p-4" onKeyDown={e => { if (e.key === 'Escape') onClose() }}>
      <div role="dialog" aria-modal="true" aria-labelledby="remove-items-title" className="bg-white rounded-xl shadow-xl max-w-2xl w-full p-6 space-y-4 max-h-[90vh] overflow-auto">
        <div><h2 id="remove-items-title" className="text-lg font-semibold">{t('collect.removeTitle')}</h2>
          <p className="text-sm text-[var(--muted)]">{t('collect.removeIntro', { n: options.length })}</p></div>
        {!options.length ? <p className="text-sm">{t('onboarding.noItems')}</p>
          : <ItemPicker options={options} chosen={chosen} onChange={setChosen} height="max-h-[50vh]" />}
        <div className="flex justify-end gap-2 border-t border-[var(--line)] pt-4">
          <Button variant="ghost" onClick={onClose}>{t('common.cancel')}</Button>
          <Button variant="danger" disabled={!chosen.length} loading={remove.isPending} onClick={submit}>{t('collect.removeN', { n: chosen.length })}</Button>
        </div>
      </div>
    </div>
  )
}
