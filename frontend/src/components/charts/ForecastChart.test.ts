import { referenceSeries } from './ForecastChart'
import type { ForecastRun } from '../../api/types'

it('plots reference seasons at their raw prices', () => {
  const run = {
    series_tail: [{ ts: 'a', median_price: 20 }, { ts: 'b', median_price: 21 }],
    forecast: [{ ts: 'c', h: 1 }, { ts: 'd', h: 2 }],
    references: [{ season_id: 's1', name: 'Season 1', points: [{ h: 1, median_price: 90 }, { h: 2, median_price: 95 }] }],
  } as unknown as ForecastRun
  const [s] = referenceSeries(run)
  expect(s.data).toEqual([null, null, 90, 95])
})
