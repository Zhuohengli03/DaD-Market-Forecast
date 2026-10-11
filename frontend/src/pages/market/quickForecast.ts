import { useNavigate } from 'react-router-dom'
import { useStartForecast } from '../../api/hooks'
import { useToast } from '../../components/ui/Toast'

/** What the one-click forecast uses unless its settings say otherwise. */
export const QUICK_FORECAST = { horizon_hours: 24, models: ['ridge', 'hgb', 'rf', 'ets', 'arima'], coverage: 0.9 }

/** The item page of a tracked quality (its type: id.item.gold_ores_5001 -> id.item.gold_ores). */
export const itemPageOf = (itemId: string) => `/item/${encodeURIComponent(itemId.replace(/_\d+$/, ''))}`

/** One-click forecast with the default settings, then open the item's page, where progress and results show. */
export function useQuickForecast() {
  const forecast = useStartForecast(); const navigate = useNavigate(); const toast = useToast()
  const run = (item: string, seasonId: string, itemId: string) => forecast.mutate({ item, season_id: seasonId, ...QUICK_FORECAST },
    { onSuccess: id => navigate(`${itemPageOf(itemId)}?${new URLSearchParams({ q: item, season: seasonId, job: id })}`),
      onError: e => toast.push((e as Error).message, 'error') })
  return { run, pending: forecast.isPending }
}
