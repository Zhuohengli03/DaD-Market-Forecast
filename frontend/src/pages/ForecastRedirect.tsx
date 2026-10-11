import { Navigate, useSearchParams } from 'react-router-dom'
import { useItems, useRuns } from '../api/hooks'

/** The forecast page is gone: a link to one forecast opens it on its item's page; anything else goes home. */
export default function ForecastRedirect() {
  const [params] = useSearchParams()
  const runs = useRuns(); const items = useItems()
  const runId = params.get('run')
  if (runId && (!runs.isSuccess || !items.isSuccess)) return null
  const run = runId ? runs.data?.find(r => r.run_id === runId) : undefined
  const itemId = run ? items.data?.find(i => i.name === run.item)?.item_id : undefined
  if (!run || !itemId) return <Navigate replace to="/" />
  const q = new URLSearchParams({ q: run.item, season: run.season_id, run: run.run_id })
  return <Navigate replace to={`/item/${encodeURIComponent(itemId.replace(/_\d+$/, ''))}?${q}`} />
}
