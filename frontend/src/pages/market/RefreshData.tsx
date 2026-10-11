import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { RefreshCw } from 'lucide-react'
import { useCollectEstimate, useItems, useJobs, useStartCollect, useStatus } from '../../api/hooks'
import { useJobStream } from '../../api/jobStream'
import { describe } from '../../components/ProgressPanel'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { fmtCount } from '../../lib/quota'
import { fmtDate, fmtRelative, fmtTsUtc } from '../../lib/format'
import { collectEta, fmtWait } from '../../lib/collectEta'

/** Collect the newest listings of the item on screen, then redraw the chart. */
/** Collects the selected season's window for one item; `latest` = the newest season (then it is "get latest data"). */
export function RefreshData({ item, seasonId, latest = true, seasonName = '' }: { item: string; seasonId?: string; latest?: boolean; seasonName?: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient(); const toast = useToast()
  const start = useStartCollect(); const estimate = useCollectEstimate(item, seasonId); const status = useStatus(); const items = useItems()
  const lastTs = items.data?.find(i => i.name === item)?.last_ts ?? null
  const [jobId, setJobId] = useState<string | null>(null)
  // a collection of this item started on another page (Collect, the overview…) is followed here too
  const jobs = useJobs().data ?? []
  const active = jobs.find(j => j.kind === 'collect' && j.meta?.item === item && (j.status === 'running' || j.status === 'queued'))
  const followed = jobId ?? active?.id ?? null
  const job = jobs.find(j => j.id === followed)
  const stream = useJobStream(followed)
  useEffect(() => { setJobId(null) }, [item, seasonId])
  useEffect(() => {
    if (stream.status === 'succeeded' || stream.status === 'failed') {
      // new listings change the chart, the table, the overview and what the next collection will cost
      for (const k of ['series', 'listings', 'items', 'overview', 'estimate', 'quota', 'status']) qc.invalidateQueries({ queryKey: [k] })
    }
  }, [stream.status, qc])
  const running = stream.status === 'running' || stream.status === 'queued'
  const elsewhere = !!status.data?.collecting?.[item] && !running
  const pages = Number(stream.progress?.done ?? 0)
  const eta = running && job ? collectEta(pages, job.cost?.estimate, job.created_at) : null
  const note = running && eta?.total ? t('market.collecting', { done: pages, total: eta.total, new: Number(stream.progress?.detail?.new ?? 0), left: fmtWait(eta.secondsLeft ?? 0) })
    : running ? describe('collect', stream.progress, t)
    // nothing new usually means DarkerDB itself has nothing newer yet: say how old its newest listing is
    // an earlier season: nothing new means this machine already has that season's window in full
    : stream.status === 'succeeded' && !Number(stream.progress?.detail?.new ?? 0) && !latest ? t('market.seasonComplete', { season: seasonName, from: fmtDate(estimate.data?.since), to: fmtDate(estimate.data?.until) })
    : stream.status === 'succeeded' && !Number(stream.progress?.detail?.new ?? 0) && lastTs ? t('market.noNewer', { when: fmtRelative(lastTs), at: fmtTsUtc(lastTs) })
    : stream.status === 'succeeded' ? describe('collect', stream.progress, t)
    : stream.status === 'failed' ? t('market.refreshFailed')
    : elsewhere ? t(`running.state.${status.data!.collecting![item]}`)
    : estimate.data?.opens ? t('collect.marketClosed', { date: estimate.data.opens })
    : estimate.data ? t('market.refreshCost', { n: fmtCount(estimate.data.requests), from: fmtDate(estimate.data.since), to: estimate.data.until ? fmtDate(estimate.data.until) : t('market.now') }) : ''
  return (
    <div className="flex flex-col items-end gap-1">
      <Button variant="ghost" title={t('market.refreshHint')} disabled={!item || elsewhere || !!estimate.data?.opens} loading={start.isPending || running}
        onClick={() => start.mutate({ item, seasonId }, { onSuccess: setJobId, onError: e => toast.push((e as Error).message, 'error') })}>
        <RefreshCw size={14} />{seasonName && seasonName !== '–' ? t('market.fetchSeason', { season: seasonName }) : t('market.refresh')}</Button>
      {eta?.share != null && <div role="progressbar" aria-label={t('market.collectProgress')} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(eta.share * 100)}
        className="w-48 h-1.5 rounded bg-[var(--line)] overflow-hidden"><div className="h-full bg-[var(--accent)] transition-all" style={{ width: `${Math.round(eta.share * 100)}%` }} /></div>}
      <span className={`text-xs ${stream.status === 'failed' ? 'text-[var(--bad)]' : 'text-[var(--muted)]'}`} aria-live="polite">{note}</span>
    </div>
  )
}
