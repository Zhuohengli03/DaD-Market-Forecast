import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Activity } from 'lucide-react'
import { useItems, useSeasons, useSeries, useStartForecast } from '../api/hooks'
import { useToast } from '../components/ui/Toast'
import { QUICK_FORECAST } from './forecast/ForecastForm'
import { seasonLabel } from '../lib/labels'
import { SelectorBar, useSelection } from '../components/SelectorBar'
import { PriceChart } from '../components/charts/PriceChart'
import { Notice } from '../components/Notice'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { fmtNum } from '../lib/format'
import { floorToBucket, summarizeRange } from '../lib/market'
import { ListingsTable } from './market/ListingsTable'
import { RefreshData } from './market/RefreshData'

export default function Market() {
  const { t } = useTranslation()
  const sel = useSelection()
  const [highlight, setHighlight] = useState<string | null>(null)
  const [range, setRange] = useState<[number, number] | null>(null)
  const series = useSeries(sel.item, { season_id: sel.season || undefined, start: sel.start || undefined, end: sel.end || undefined, resolution: sel.res })
  useEffect(() => { setRange(null); setHighlight(null) }, [sel.item, sel.season, sel.start, sel.end, sel.res])
  const points = series.data?.points ?? []
  const items = useItems(); const seasons = useSeasons()
  const forecast = useStartForecast(); const navigate = useNavigate(); const toast = useToast()
  const quickForecast = () => forecast.mutate({ item: sel.item, season_id: sel.season, ...QUICK_FORECAST },
    { onSuccess: id => navigate(`/forecast?item=${encodeURIComponent(sel.item)}&season=${encodeURIComponent(sel.season)}&job=${id}`), onError: e => toast.push((e as Error).message, 'error') })
  const stored = items.data?.find(i => i.name === sel.item)?.rows ?? 0
  const seasonName = seasonLabel(seasons.data?.find(s => s.id === sel.season)?.name, t)
  // "no data" differs when the item has listings in other seasons: say so instead of asking to collect
  const empty = stored > 0 ? t('market.emptySeason', { season: seasonName, n: stored }) : t('dashboard.noData')
  const summary = range ? summarizeRange(points, range[0], range[1]) : null
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('market.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('market.intro')}</p></div>
      {!sel.items.length && <Notice to="/catalog" action={t('nav.catalog')}>{t('onboarding.noItems')}</Notice>}
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-[320px]"><SelectorBar /></div>
        <RefreshData item={sel.item} seasonId={sel.season || undefined} />
        <div className="flex flex-col items-end gap-1">
          <Button loading={forecast.isPending} disabled={!points.length || !sel.season} title={!sel.season ? t('market.forecastNeedsSeason') : undefined} onClick={quickForecast}><Activity size={14} />{t('market.forecastNow')}</Button>
          <span className="text-xs text-[var(--muted)]">{t('market.forecastHint')}</span>
        </div>
      </div>
      <Card>
        {series.isLoading ? <p className="text-sm">{t('common.loading')}</p> : series.isError ? <EmptyState title={t('common.error')} hint={(series.error as Error).message} /> : !points.length ? <EmptyState title={empty} /> :
          <>
            <PriceChart points={points} highlightTs={highlight} onRange={setRange} />
            {summary && <div className="mt-2 flex items-center justify-between text-sm bg-blue-50 rounded-md px-3 py-2">
              <span>{t('market.range', { hours: summary.hours, low: fmtNum(summary.low), median: fmtNum(summary.median), high: fmtNum(summary.high), listings: summary.listings })}</span>
              <Button size="sm" variant="ghost" onClick={() => setRange(null)}>{t('market.clearRange')}</Button></div>}
          </>}
      </Card>
      <Card title={t('market.listings')}>
        <ListingsTable item={sel.item} season_id={sel.season} start={sel.start} end={sel.end} onPick={ts => setHighlight(floorToBucket(ts, sel.res))} />
      </Card>
    </div>
  )
}
