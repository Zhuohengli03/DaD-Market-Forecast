import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItemsConfig, useSaveItemsConfig } from '../../api/hooks'
import type { ItemConfigEntry, ItemsConfig } from '../../api/types'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { itemName } from '../../lib/labels'

export function ItemsConfigForm() {
  const { t, i18n } = useTranslation()
  const cfg = useItemsConfig(); const save = useSaveItemsConfig(); const toast = useToast()
  const [draft, setDraft] = useState<ItemsConfig | null>(null)
  useEffect(() => { if (cfg.data) setDraft(cfg.data) }, [cfg.data])
  if (!draft) return <p className="text-sm">{t('common.loading')}</p>
  const setRow = (i: number, patch: Partial<ItemConfigEntry>) => setDraft({ ...draft, items: draft.items.map((it, k) => (k === i ? { ...it, ...patch } : it)) })
  return (
    <div className="space-y-3">
      <table className="w-full text-sm"><thead className="text-left text-[var(--muted)]"><tr>
        <th className="py-1">{t('settings.itemName')}</th><th>{t('settings.itemId')}</th><th>{t('settings.description')}</th><th>{t('settings.enabled')}</th></tr></thead>
        <tbody>{draft.items.map((it, i) => (
          <tr key={it.name} className="border-t border-[var(--line)]">
            <td className="py-1.5 font-medium">{itemName(it, i18n.language)}</td>
            <td className="font-mono text-xs text-[var(--muted)]">{it.item_id ?? it.file ?? ''}</td>
            <td className="pr-2"><input className={`${inputCls} w-full`} value={it.description} onChange={e => setRow(i, { description: e.target.value })} /></td>
            <td><input type="checkbox" checked={it.enabled} onChange={e => setRow(i, { enabled: e.target.checked })} aria-label={`${t('settings.enabled')} ${itemName(it, i18n.language)}`} /></td>
          </tr>))}</tbody></table>
      <Button loading={save.isPending} onClick={() => save.mutate(draft, { onSuccess: () => toast.push(t('settings.saved')), onError: e => toast.push((e as Error).message, 'error') })}>{t('common.save')}</Button>
    </div>
  )
}
