import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSeries } from '../../api/hooks'
import type { Resolution, Season } from '../../api/types'
import { PriceChart, snapTs, type PriceMark } from '../../components/charts/PriceChart'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import { inputCls } from '../../components/ui/Field'
import { fmtNum } from '../../lib/format'
import { seasonLabel } from '../../lib/labels'
import { summarizeRange } from '../../lib/market'

/** The item's price over one season, with game updates marked; drag on the chart for a period's numbers. */
/** The season is chosen in the page header; this section follows it. */
export function PriceSection({ item, seasons, season, marks = [], focusTs = null }: {
  item: string; seasons: Season[]; season: string; marks?: PriceMark[]; focusTs?: string | null }) {
  const { t } = useTranslation()
  const [res, setRes] = useState<Resolution>('1h')
  const [range, setRange] = useState<[number, number] | null>(null)
  const series = useSeries(item, { season_id: season || undefined, resolution: res })
  useEffect(() => setRange(null), [item, season, res])
  const points = series.data?.points ?? []
  const summary = range ? summarizeRange(points, range[0], range[1]) : null
  const name = seasonLabel(seasons.find(s => s.id === season)?.name, t)
  return (
    <Card title={`${t('item.price')} · ${name}`} actions={<>
      <select aria-label={t('common.resolution')} className={inputCls} value={res} onChange={e => setRes(e.target.value as Resolution)}>
        {(['1h', '4h', '1d'] as const).map(r => <option key={r} value={r}>{t(`res.${r}`)}</option>)}</select></>}>
      {series.isLoading ? <p className="text-sm">{t('common.loading')}</p>
        : !points.length ? <EmptyState title={t('item.noSeasonData', { season: name })} />
        : <>
          <PriceChart points={points} highlightTs={focusTs ? snapTs(points, focusTs) : null} onRange={setRange} marks={marks} />
          {summary && <div className="mt-2 flex items-center justify-between text-sm bg-blue-50 rounded-md px-3 py-2">
            <span>{t('market.range', { hours: summary.hours, low: fmtNum(summary.low), median: fmtNum(summary.median), high: fmtNum(summary.high), listings: summary.listings })}</span>
            <Button size="sm" variant="ghost" onClick={() => setRange(null)}>{t('market.clearRange')}</Button></div>}
        </>}
    </Card>
  )
}
