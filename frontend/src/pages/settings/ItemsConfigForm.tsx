import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItemsConfig, useSaveItemsConfig } from '../../api/hooks'
import type { ItemConfigEntry, ItemsConfig } from '../../api/types'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'

const blank: ItemConfigEntry = { name: '', file: '', csv: '', category: 'ore', description: '', enabled: true }
const LABEL_KEYS: Record<string, string> = { name: 'settings.itemName', file: 'settings.file', csv: 'settings.csv', category: 'settings.category', description: 'settings.description' }

export function ItemsConfigForm() {
  const { t } = useTranslation()
  const cfg = useItemsConfig(); const save = useSaveItemsConfig(); const toast = useToast()
  const [draft, setDraft] = useState<ItemsConfig | null>(null)
  useEffect(() => { if (cfg.data) setDraft(cfg.data) }, [cfg.data])
  if (!draft) return <p className="text-sm">{t('common.loading')}</p>
  const setRow = (i: number, patch: Partial<ItemConfigEntry>) => setDraft({ ...draft, items: draft.items.map((it, k) => (k === i ? { ...it, ...patch } : it)) })
  const cols: (keyof ItemConfigEntry)[] = ['name', 'file', 'csv', 'category', 'description']
  return (
    <div className="space-y-3">
      <table className="w-full text-sm"><thead className="text-left text-[var(--muted)]"><tr>{cols.map(c => <th key={c} className="py-1">{t(LABEL_KEYS[c])}</th>)}<th>{t('settings.enabled')}</th><th></th></tr></thead>
        <tbody>{draft.items.map((it, i) => (
          <tr key={i} className="border-t border-[var(--line)]">
            {cols.map(c => <td key={c} className="py-1 pr-2">{c === 'category'
              ? <select className={inputCls} value={it.category} onChange={e => setRow(i, { category: e.target.value })}>{Object.entries(draft.categories).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select>
              : <input className={`${inputCls} w-full`} value={it[c] as string} onChange={e => setRow(i, { [c]: e.target.value })} />}</td>)}
            <td><input type="checkbox" checked={it.enabled} onChange={e => setRow(i, { enabled: e.target.checked })} /></td>
            <td><Button size="sm" variant="ghost" onClick={() => setDraft({ ...draft, items: draft.items.filter((_, k) => k !== i) })}>✕</Button></td>
          </tr>))}</tbody></table>
      <div className="flex gap-2">
        <Button variant="ghost" onClick={() => setDraft({ ...draft, items: [...draft.items, blank] })}>{t('common.add')}</Button>
        <Button loading={save.isPending} onClick={() => save.mutate(draft, { onSuccess: () => toast.push(t('settings.saved')), onError: e => toast.push((e as Error).message, 'error') })}>{t('common.save')}</Button>
      </div>
    </div>
  )
}
