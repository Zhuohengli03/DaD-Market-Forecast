import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSeries } from '../api/hooks'
import { SelectorBar, useSelection } from '../components/SelectorBar'
import { PriceChart } from '../components/charts/PriceChart'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ListingsTable } from './market/ListingsTable'

export default function Market() {
  const { t } = useTranslation()
  const sel = useSelection()
  const [highlight, setHighlight] = useState<string | null>(null)
  const series = useSeries(sel.item, { season_id: sel.season || undefined, start: sel.start || undefined, end: sel.end || undefined, resolution: sel.res })
  const hourOf = (ts: string) => { const d = new Date(ts); d.setUTCMinutes(0, 0, 0); return d.toISOString().replace('.000Z', '+00:00') }
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">{t('market.title')}</h1>
      <SelectorBar />
      <Card>
        {series.isLoading ? <p className="text-sm">{t('common.loading')}</p> : series.isError ? <EmptyState title={t('common.error')} hint={(series.error as Error).message} /> : !series.data?.points.length ? <EmptyState title={t('common.none')} /> :
          <PriceChart points={series.data.points} highlightTs={highlight} />}
      </Card>
      <Card title={t('market.listings')}>
        <ListingsTable item={sel.item} season_id={sel.season} start={sel.start} end={sel.end} onPick={ts => setHighlight(hourOf(ts))} />
      </Card>
    </div>
  )
}
