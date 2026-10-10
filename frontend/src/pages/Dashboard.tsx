import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useItems, useSeries, useStatus } from '../api/hooks'
import type { ItemInfo } from '../api/types'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { Sparkline } from '../components/charts/Sparkline'
import { GettingStarted } from './GettingStarted'
import { RunningTasks } from '../components/RunningTasks'
import { NoDataCard } from './dashboard/NoDataCard'
import { fmtNum, fmtPct, fmtRelative, fmtTs } from '../lib/format'
import { change24h } from '../lib/market'
import { itemName } from '../lib/labels'

function ItemCard({ item, state }: { item: ItemInfo; state?: 'collecting' | 'waiting' }) {
  const { t, i18n } = useTranslation()
  const last = item.last_ts ? new Date(item.last_ts).getTime() : null
  // the API range end is exclusive, so extend one hour past the newest listing
  const end = last !== null ? new Date(last + 3.6e6).toISOString() : undefined
  const start = last !== null ? new Date(last - 48 * 3.6e6).toISOString() : undefined
  const series = useSeries(item.name, { start, end, resolution: '1h' }, !!end)
  const pts = series.data?.points ?? []
  const prices = pts.map(p => p.median_price)
  const { latest, change } = change24h(pts)
  return (
    <Link to={`/market?item=${encodeURIComponent(item.name)}`} className="block">
      <Card className="hover:shadow-md transition">
        <div className="flex items-start justify-between"><div><div className="font-semibold">{itemName(item, i18n.language)}</div><div className="text-xs text-[var(--muted)]">{item.description}</div></div><div className="flex flex-col items-end gap-1"><Badge tone={item.rows ? 'muted' : 'warn'}>{item.rows} {t('dashboard.rows')}</Badge>{state && <Badge tone={state === 'collecting' ? 'warn' : 'muted'}>{t(`running.state.${state}`)}</Badge>}</div></div>
        <div className="mt-3 flex items-end justify-between">
          <div><div className="text-xs text-[var(--muted)]">{t('dashboard.latest')}</div><div className="text-2xl font-semibold">{fmtNum(latest)}</div></div>
          <div className="text-right"><div className="text-xs text-[var(--muted)]">{t('dashboard.change24')}</div><div className={`font-medium ${change === null ? '' : change >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}`}>{fmtPct(change)}</div></div>
        </div>
        <div className="mt-2">{pts.length ? <Sparkline values={prices} /> : <p className="text-xs text-[var(--muted)] h-10">{t('dashboard.noData')}</p>}</div>
        <div className="text-xs text-[var(--muted)] mt-2">{t('dashboard.lastSeen')}: {fmtTs(item.last_ts)} ({fmtRelative(item.last_ts)})</div>
      </Card>
    </Link>
  )
}

export default function Dashboard() {
  const { t } = useTranslation()
  const items = useItems()
  const status = useStatus()
  const states = status.data?.collecting ?? {}
  // items with no listings yet (e.g. an interrupted first collection) have nothing to chart: list them together
  const withData = (items.data ?? []).filter(i => i.rows > 0)
  const empty = (items.data ?? []).filter(i => !i.rows)
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">{t('dashboard.title')}</h1>
      <GettingStarted />
      {items.isLoading ? <p>{t('common.loading')}</p> : !items.data?.length ? <EmptyState title={t('common.none')} /> : <>
        {withData.length > 0 && <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{withData.map(i => <ItemCard key={i.name} item={i} state={states[i.name]} />)}</div>}
        {empty.length > 0 && <NoDataCard items={empty} states={states} />}
      </>}
      <Card title={t('dashboard.runningJobs')}>
        <RunningTasks />
      </Card>
    </div>
  )
}
