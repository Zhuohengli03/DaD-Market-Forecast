import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useItems, useSeries, useStatus } from '../api/hooks'
import type { ItemInfo } from '../api/types'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { Sparkline } from '../components/charts/Sparkline'
import { fmtNum, fmtPct, fmtRelative, fmtTs } from '../lib/format'

function ItemCard({ item }: { item: ItemInfo }) {
  const { t } = useTranslation()
  const end = item.last_ts ?? undefined
  const start = end ? new Date(new Date(end).getTime() - 48 * 3.6e6).toISOString() : undefined
  const series = useSeries(item.name, { start, end, resolution: '1h' }, !!end)
  const pts = series.data?.points ?? []
  const prices = pts.map(p => p.median_price)
  const valid = prices.filter((v): v is number => v !== null)
  const latest = valid.at(-1) ?? null
  const dayAgo = valid.length > 24 ? valid[valid.length - 25] : valid[0] ?? null
  const change = latest !== null && dayAgo ? latest / dayAgo - 1 : null
  return (
    <Link to={`/market?item=${encodeURIComponent(item.name)}`} className="block">
      <Card className="hover:shadow-md transition">
        <div className="flex items-start justify-between"><div><div className="font-semibold">{item.name}</div><div className="text-xs text-[var(--muted)]">{item.description}</div></div><Badge tone={item.rows ? 'muted' : 'warn'}>{item.rows} {t('dashboard.rows')}</Badge></div>
        <div className="mt-3 flex items-end justify-between">
          <div><div className="text-xs text-[var(--muted)]">{t('dashboard.latest')}</div><div className="text-2xl font-semibold">{fmtNum(latest)}</div></div>
          <div className="text-right"><div className="text-xs text-[var(--muted)]">{t('dashboard.change24')}</div><div className={`font-medium ${change === null ? '' : change >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}`}>{fmtPct(change)}</div></div>
        </div>
        <div className="mt-2"><Sparkline values={prices} /></div>
        <div className="text-xs text-[var(--muted)] mt-2">{t('dashboard.lastSeen')}: {fmtTs(item.last_ts)} ({fmtRelative(item.last_ts)})</div>
      </Card>
    </Link>
  )
}

export default function Dashboard() {
  const { t } = useTranslation()
  const items = useItems()
  const status = useStatus()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">{t('dashboard.title')}</h1>
      {items.isLoading ? <p>{t('common.loading')}</p> : !items.data?.length ? <EmptyState title={t('common.none')} /> :
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{items.data.map(i => <ItemCard key={i.name} item={i} />)}</div>}
      <Card title={t('dashboard.runningJobs')}>
        {status.data?.jobs_running.length ? <ul className="text-sm space-y-1">{status.data.jobs_running.map(j => <li key={j.id} className="flex justify-between"><span>{j.label}</span><Badge tone="warn">{j.status}</Badge></li>)}</ul> : <p className="text-sm text-[var(--muted)]">{t('dashboard.noJobs')}</p>}
      </Card>
    </div>
  )
}
