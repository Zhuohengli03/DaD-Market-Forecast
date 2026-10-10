import { useTranslation } from 'react-i18next'
import { useItems, useSeasons } from '../api/hooks'
import type { Resolution } from '../api/types'
import { useUrlState } from '../lib/urlState'
import { inputCls } from './ui/Field'
import { itemName } from '../lib/labels'

export function useSelection() {
  const items = useItems(); const seasons = useSeasons()
  const [item, setItem] = useUrlState('item', '')
  const [season, setSeason] = useUrlState('season', '')
  const [start, setStart] = useUrlState('start', '')
  const [end, setEnd] = useUrlState('end', '')
  const [res, setRes] = useUrlState('res', '1h')
  const effectiveItem = item || items.data?.[0]?.name || ''
  const effectiveSeason = season || (start || end ? '' : seasons.data?.at(-1)?.id ?? '')
  return { item: effectiveItem, season: effectiveSeason, start, end, res: res as Resolution, setItem, setSeason, setStart, setEnd, setRes: (r: Resolution) => setRes(r), items: items.data ?? [], seasons: seasons.data ?? [] }
}

export function SelectorBar({ showRange = true, showResolution = true }: { showRange?: boolean; showResolution?: boolean }) {
  const { t, i18n } = useTranslation()
  const s = useSelection()
  const toLocalInput = (iso: string) => iso ? new Date(iso).toISOString().slice(0, 16) : ''
  const fromLocalInput = (v: string) => v ? new Date(v + 'Z').toISOString() : ''
  return (
    <div className="flex flex-wrap items-end gap-3 bg-[var(--panel)] border border-[var(--line)] rounded-xl p-3">
      <label className="text-sm flex flex-col gap-1"><span className="text-[var(--muted)]">{t('common.item')}</span>
        <select className={inputCls} value={s.item} onChange={e => s.setItem(e.target.value)}>{s.items.map(i => <option key={i.name} value={i.name}>{itemName(i, i18n.language)}</option>)}</select></label>
      <label className="text-sm flex flex-col gap-1"><span className="text-[var(--muted)]">{t('common.season')}</span>
        <select className={inputCls} value={s.season} onChange={e => { s.setSeason(e.target.value); s.setStart(''); s.setEnd('') }}>
          <option value="">{t('market.customRange')}</option>
          {s.seasons.map(se => <option key={se.id} value={se.id}>{se.name}</option>)}</select></label>
      {showRange && !s.season && <>
        <label className="text-sm flex flex-col gap-1"><span className="text-[var(--muted)]">{t('common.start')} (UTC)</span><input type="datetime-local" className={inputCls} value={toLocalInput(s.start)} onChange={e => s.setStart(fromLocalInput(e.target.value))} /></label>
        <label className="text-sm flex flex-col gap-1"><span className="text-[var(--muted)]">{t('common.end')} (UTC)</span><input type="datetime-local" className={inputCls} value={toLocalInput(s.end)} onChange={e => s.setEnd(fromLocalInput(e.target.value))} /></label>
      </>}
      {showResolution && <div className="text-sm flex flex-col gap-1"><span className="text-[var(--muted)]">{t('common.resolution')}</span>
        <div className="inline-flex border border-[var(--line)] rounded-md overflow-hidden">{(['1h', '4h', '1d'] as const).map(r => <button key={r} type="button" onClick={() => s.setRes(r)} className={`px-3 py-1.5 text-sm ${s.res === r ? 'bg-[var(--accent)] text-white' : 'bg-white'}`}>{r}</button>)}</div></div>}
    </div>
  )
}
