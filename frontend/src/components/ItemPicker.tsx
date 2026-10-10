import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useCatalog } from '../api/hooks'
import type { CatalogType } from '../api/types'
import { itemName, rarityLabel } from '../lib/labels'
import { useDebounced } from '../lib/useDebounced'
import { Button } from './ui/Button'
import { inputCls } from './ui/Field'

// shortcut buttons: what players usually trade first, then everything else by size
const CATEGORY_ORDER = ['ore', 'ingot', 'powder', 'gem', 'drink', 'herb', 'wood', 'consumable', 'throwable', 'light_source', 'food', 'treasure', 'gold', 'currency']

export interface PickOption { id: string; type: CatalogType; category: string; label: string }

/** Every tradable item without random stats, one option per rarity, grouped by category. */
export function usePickOptions(): { options: PickOption[]; loading: boolean } {
  const { t, i18n } = useTranslation()
  const cat = useCatalog({ has_stats: false, page: 1, page_size: 1000 })
  const options = useMemo(() => (cat.data?.rows ?? []).flatMap(r => r.variants.map(v => ({
    id: v.id, type: r, category: r.category || 'misc',
    label: r.variants.length > 1 ? `${itemName(r, i18n.language)}（${rarityLabel(v.rarity, t)}）` : itemName(r, i18n.language),
  }))), [cat.data, i18n.language, t])
  return { options, loading: cat.isLoading }
}

interface Props { options: PickOption[]; chosen: string[]; onChange: (ids: string[]) => void; height?: string }

/** Checkbox list with "select all", "select none" and "select a whole category" shortcuts. */
export function ItemPicker({ options, chosen, onChange, height = 'max-h-64' }: Props) {
  const { t } = useTranslation()
  const [q, setQ] = useState(''); const query = useDebounced(q, 300)
  const groups = useMemo(() => {
    const m = new Map<string, PickOption[]>()
    for (const o of options) m.set(o.category, [...(m.get(o.category) ?? []), o])
    const rank = (c: string) => { const i = CATEGORY_ORDER.indexOf(c); return i < 0 ? CATEGORY_ORDER.length : i }
    return [...m.entries()].sort((a, b) => rank(a[0]) - rank(b[0]) || a[1].length - b[1].length)
  }, [options])
  const toggle = (id: string) => onChange(chosen.includes(id) ? chosen.filter(x => x !== id) : [...chosen, id])
  const toggleGroup = (ids: string[]) => {
    const all = ids.every(id => chosen.includes(id))
    onChange(all ? chosen.filter(id => !ids.includes(id)) : [...new Set([...chosen, ...ids])])
  }
  const visible = (o: PickOption) => !query || o.label.includes(query) || o.type.name.toLowerCase().includes(query.toLowerCase())
  return (
    <div className="space-y-3 text-sm">
      <div className="flex flex-wrap gap-1.5">
        <Button size="sm" onClick={() => onChange(options.map(o => o.id))}>{t('wizard.init.all', { n: options.length })}</Button>
        <Button size="sm" variant="ghost" onClick={() => onChange([])}>{t('wizard.init.none')}</Button>
        {groups.map(([c, list]) => (
          <Button key={c} size="sm" variant="ghost" onClick={() => toggleGroup(list.map(o => o.id))}>{t('wizard.init.allOf', { cat: t(`category.${c}`, { defaultValue: c }), n: list.length })}</Button>))}
      </div>
      <input className={`${inputCls} w-full`} placeholder={t('wizard.init.search')} value={q} onChange={e => setQ(e.target.value)} />
      <div className={`${height} overflow-auto border border-[var(--line)] rounded-md p-2 space-y-2`}>
        {groups.map(([c, list]) => {
          const shown = list.filter(visible)
          if (!shown.length) return null
          return (
            <div key={c}>
              <div className="text-xs font-semibold text-[var(--muted)] mb-1">{t(`category.${c}`, { defaultValue: c })}</div>
              <div className="grid grid-cols-2 gap-1">{shown.map(o => <label key={o.id} className="flex items-center gap-2"><input type="checkbox" aria-label={o.label} checked={chosen.includes(o.id)} onChange={() => toggle(o.id)} />{o.label}</label>)}</div>
            </div>)
        })}
      </div>
    </div>
  )
}
