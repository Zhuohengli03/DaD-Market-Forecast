import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useItems, useOverview, useSeasons, useStatus } from '../api/hooks'
import type { OverviewGroup } from '../api/types'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { inputCls } from '../components/ui/Field'
import { GettingStarted } from './GettingStarted'
import { RunningTasks } from '../components/RunningTasks'
import { NoDataCard } from './dashboard/NoDataCard'
import { TypeCard, groupName, variantLabel } from './dashboard/TypeCard'
import { useDebounced } from '../lib/useDebounced'

const SORTS = ['change', 'rows', 'name'] as const
type Sort = typeof SORTS[number]
// sections in the order players usually care about; anything else follows alphabetically
const CATEGORY_ORDER = ['ore', 'ingot', 'powder', 'gem', 'drink', 'herb', 'wood', 'consumable', 'throwable', 'light_source', 'food', 'treasure', 'gold', 'currency']

const biggestMove = (g: OverviewGroup) => Math.max(0, ...g.variants.map(v => Math.abs(v.change24 ?? 0)))

export default function Dashboard() {
  const { t, i18n } = useTranslation()
  const items = useItems(); const status = useStatus(); const seasons = useSeasons()
  // every item at its own newest trading season; each card picks another season by itself
  const overview = useOverview(undefined, true)
  const allSeasons = seasons.data ?? []
  const states = status.data?.collecting ?? {}
  const [q, setQ] = useState(''); const query = useDebounced(q, 200).trim().toLowerCase()
  const [sort, setSort] = useState<Sort>('change')
  const [filter, setFilter] = useState<'all' | 'data' | 'nodata'>('all')
  // every tracked item gets a card; those without trading data show dashes, after the ones with data
  const groups = overview.data ?? []
  const traded = (g: OverviewGroup) => g.variants.some(v => v.season_id)
  const sections = useMemo(() => {
    const match = (g: OverviewGroup) => !query || [g.name, g.name_zh ?? '', ...g.variants.map(v => variantLabel(v, i18n.language))].some(s => s.toLowerCase().includes(query))
    const cmp = (a: OverviewGroup, b: OverviewGroup) => Number(traded(b)) - Number(traded(a)) || cmpBy(a, b)
    const cmpBy = (a: OverviewGroup, b: OverviewGroup) => sort === 'change' ? biggestMove(b) - biggestMove(a)
      : sort === 'rows' ? b.season_rows - a.season_rows : groupName(a, i18n.language).localeCompare(groupName(b, i18n.language), i18n.language)
    const by = new Map<string, OverviewGroup[]>()
    const kept = (g: OverviewGroup) => filter === 'all' || (filter === 'data') === traded(g)
    for (const g of groups.filter(g => match(g) && kept(g))) by.set(g.category || 'other', [...(by.get(g.category || 'other') ?? []), g])
    const rank = (c: string) => { const i = CATEGORY_ORDER.indexOf(c); return i < 0 ? CATEGORY_ORDER.length : i }
    return [...by.entries()].map(([c, list]) => [c, list.sort(cmp)] as const).sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0]))
  }, [groups, query, sort, filter, i18n.language])
  const empty = (items.data ?? []).filter(i => !i.rows)
  const shown = sections.reduce((n, [, l]) => n + l.length, 0)
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">{t('dashboard.title')}</h1>
      <GettingStarted />
      {groups.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <input className={`${inputCls} w-64`} placeholder={t('dashboard.search')} aria-label={t('dashboard.search')} value={q} onChange={e => setQ(e.target.value)} />
          <select className={inputCls} aria-label={t('dashboard.filter')} value={filter} onChange={e => setFilter(e.target.value as typeof filter)}>
            <option value="all">{t('dashboard.filters.all', { n: groups.length })}</option>
            <option value="data">{t('dashboard.filters.data', { n: groups.filter(traded).length })}</option>
            <option value="nodata">{t('dashboard.filters.nodata', { n: groups.filter(g => !traded(g)).length })}</option></select>
          <label className="flex items-center gap-2 text-[var(--muted)]">{t('dashboard.sortBy')}
            <select className={inputCls} value={sort} onChange={e => setSort(e.target.value as Sort)}>{SORTS.map(s => <option key={s} value={s}>{t(`dashboard.sorts.${s}`)}</option>)}</select></label>
          <span className="ml-auto text-xs text-[var(--muted)]">{t('dashboard.summary', { types: shown, items: groups.reduce((n, g) => n + g.variants.filter(v => v.season_rows > 0).length, 0) })}</span>
        </div>)}
      {overview.isLoading ? <p>{t('common.loading')}</p> : !(overview.data ?? []).length ? <EmptyState title={t('common.none')} /> :

        sections.length === 0 ? <EmptyState title={t('dashboard.noMatch')} /> :
        sections.map(([cat, list]) => (
          <details key={cat} open className="group">
            <summary className="cursor-pointer select-none text-sm font-semibold text-[var(--muted)] mb-2 list-none flex items-center gap-2">
              <span className="inline-block transition group-open:rotate-90">›</span>{t(`category.${cat}`, { defaultValue: cat })}<span className="font-normal">（{list.length}）</span></summary>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">{list.map(g => <TypeCard key={g.key} group={g} states={states} seasons={allSeasons} />)}</div>
          </details>))}
      {empty.length > 0 && <NoDataCard items={empty} states={states} />}
      <Card title={t('dashboard.runningJobs')}><RunningTasks /></Card>
    </div>
  )
}
