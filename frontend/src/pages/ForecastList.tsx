import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useItems, useRuns, useSeasons } from '../api/hooks'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { inputCls } from '../components/ui/Field'
import { fmtTs } from '../lib/format'
import { itemName, modelLabel, seasonLabel } from '../lib/labels'
import { useUrlState } from '../lib/urlState'
import ForecastRedirect from './ForecastRedirect'
import { itemPageOf } from './market/quickForecast'

const BASELINES = ['naive', 'seasonal_naive_24']

/** Items forecast in one season, newest first, like the catalog; a row opens the item page on its latest forecast. */
export default function ForecastList() {
  const [params] = useSearchParams()
  if (params.get('run')) return <ForecastRedirect />      // an old link to one forecast
  return <List />
}

function List() {
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const runs = useRuns().data ?? []; const items = useItems().data ?? []; const seasons = useSeasons().data ?? []
  const [seasonParam, setSeason] = useUrlState('season', '')
  const [query, setQuery] = useState('')
  const newest = [...runs].sort((a, b) => b.created_at.localeCompare(a.created_at))
  const season = seasons.some(s => s.id === seasonParam) ? seasonParam : newest[0]?.season_id ?? seasons.at(-1)?.id ?? ''
  const byItem = new Map<string, typeof runs>()
  for (const r of newest.filter(r => r.season_id === season)) byItem.set(r.item, [...(byItem.get(r.item) ?? []), r])
  const q = query.trim().toLowerCase()
  const rows = [...byItem.entries()].map(([name, rs]) => ({ name, runs: rs, info: items.find(i => i.name === name) }))
    .filter(r => !q || [r.name, r.info?.name_zh ?? ''].some(s => s.toLowerCase().includes(q)))
  const seasonName = seasonLabel(seasons.find(s => s.id === season)?.name, t)
  const open = (name: string, runId: string, itemId?: string | null) => itemId &&
    navigate(`${itemPageOf(itemId)}?${new URLSearchParams({ q: name, season, run: runId })}`)
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('forecastList.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('forecastList.intro')}</p></div>
      <div className="flex flex-wrap gap-2 items-center text-sm">
        <select aria-label={t('common.season')} className={inputCls} value={season} onChange={e => setSeason(e.target.value)}>
          {[...seasons].sort((a, b) => b.start.localeCompare(a.start)).map(s => <option key={s.id} value={s.id}>{seasonLabel(s.name, t)}</option>)}</select>
        <input aria-label={t('dashboard.search')} placeholder={t('dashboard.search')} className={`${inputCls} w-56`} value={query} onChange={e => setQuery(e.target.value)} />
        <span className="ml-auto text-[var(--muted)]">{t('forecastList.count', { n: byItem.size })}</span>
      </div>
      {!byItem.size ? <EmptyState title={t('forecastList.empty', { season: seasonName })} />
        : !rows.length ? <EmptyState title={t('dashboard.noMatch')} />
        : <Card><table className="w-full text-sm">
          <thead className="text-left text-[var(--muted)] text-xs"><tr>
            <th className="px-3 py-2 w-12"></th><th className="px-3 py-2">{t('common.item')}</th><th className="px-3 py-2 text-right">{t('forecastList.runs')}</th>
            <th className="px-3 py-2">{t('forecastList.latest')}</th><th className="px-3 py-2">{t('forecast.horizon')}</th><th className="px-3 py-2">{t('forecastList.model')}</th></tr></thead>
          <tbody>{rows.map(({ name, runs: rs, info }) => { const last = rs[0]; return (
            <tr key={name} className="border-t border-[var(--line)] hover:bg-gray-50 cursor-pointer" onClick={() => open(name, last.run_id, info?.item_id)}>
              <td className="px-3 py-1.5">{info?.icon_url ? <img src={info.icon_url} alt="" loading="lazy" className="w-8 h-8 object-contain" /> : <div className="w-8 h-8 rounded bg-gray-100" />}</td>
              <td className="px-3 py-1.5 font-medium">{info ? itemName(info, i18n.language) : name}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{rs.length}</td>
              <td className="px-3 py-1.5 tabular-nums text-[var(--muted)]">{fmtTs(last.created_at)}</td>
              <td className="px-3 py-1.5">{last.horizon_hours} {t('common.hours')}</td>
              <td className="px-3 py-1.5" title={t('forecastList.modelHint')}>{modelLabel(last.selected, t)}{BASELINES.includes(last.selected) ? t('forecastList.baseline') : ''}</td>
            </tr>) })}</tbody></table></Card>}
    </div>
  )
}
