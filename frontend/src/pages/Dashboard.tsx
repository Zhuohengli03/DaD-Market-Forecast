import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useItems, useSeries, useStatus } from '../api/hooks'
import type { ItemInfo } from '../api/types'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { Sparkline } from '../components/charts/Sparkline'
import { fmtNum, fmtPct, fmtRelative, fmtTs } from '../lib/format'
import { change24h } from '../lib/market'
import { itemName, localizeMessage } from '../lib/labels'

function ItemCard({ item }: { item: ItemInfo }) {
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
        <div className="flex items-start justify-between"><div><div className="font-semibold">{itemName(item, i18n.language)}</div><div className="text-xs text-[var(--muted)]">{item.description}</div></div><Badge tone={item.rows ? 'muted' : 'warn'}>{item.rows} {t('dashboard.rows')}</Badge></div>
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
  const { t, i18n } = useTranslation()
  const items = useItems()
  const status = useStatus()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">{t('dashboard.title')}</h1>
      {items.isLoading ? <p>{t('common.loading')}</p> : !items.data?.length ? <EmptyState title={t('common.none')} /> :
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{items.data.map(i => <ItemCard key={i.name} item={i} />)}</div>}
      <Card title={t('dashboard.runningJobs')}>
        {status.data?.jobs_running.length ? <ul className="text-sm space-y-1">{status.data.jobs_running.map(j => <li key={j.id} className="flex justify-between"><span>{localizeMessage(j.label, i18n.language)}</span><Badge tone="warn">{t(`progress.status.${j.status}`)}</Badge></li>)}</ul> : <p className="text-sm text-[var(--muted)]">{t('dashboard.noJobs')}</p>}
      </Card>
    </div>
  )
}
