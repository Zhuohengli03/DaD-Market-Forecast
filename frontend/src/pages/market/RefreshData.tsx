import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { RefreshCw } from 'lucide-react'
import { useCollectEstimate, useItems, useStartCollect, useStatus } from '../../api/hooks'
import { useJobStream } from '../../api/jobStream'
import { describe } from '../../components/ProgressPanel'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { fmtCount } from '../../lib/quota'
import { fmtDate, fmtRelative, fmtTsUtc } from '../../lib/format'

/** Collect the newest listings of the item on screen, then redraw the chart. */
export function RefreshData({ item, seasonId }: { item: string; seasonId?: string }) {
  const { t } = useTranslation()
  const qc = useQueryClient(); const toast = useToast()
  const start = useStartCollect(); const estimate = useCollectEstimate(item, seasonId); const status = useStatus(); const items = useItems()
  const lastTs = items.data?.find(i => i.name === item)?.last_ts ?? null
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  useEffect(() => { setJobId(null) }, [item, seasonId])
  useEffect(() => {
    if (stream.status === 'succeeded' || stream.status === 'failed') {
      // new listings change the chart, the table, the overview and what the next collection will cost
      for (const k of ['series', 'listings', 'items', 'overview', 'estimate', 'quota', 'status']) qc.invalidateQueries({ queryKey: [k] })
    }
  }, [stream.status, qc])
  const running = stream.status === 'running' || stream.status === 'queued'
  const elsewhere = !!status.data?.collecting?.[item] && !running
  const note = running ? describe('collect', stream.progress, t)
    // nothing new usually means DarkerDB itself has nothing newer yet: say how old its newest listing is
    : stream.status === 'succeeded' && !Number(stream.progress?.detail?.new ?? 0) && lastTs ? t('market.noNewer', { when: fmtRelative(lastTs), at: fmtTsUtc(lastTs) })
    : stream.status === 'succeeded' ? describe('collect', stream.progress, t)
    : stream.status === 'failed' ? t('market.refreshFailed')
    : elsewhere ? t(`running.state.${status.data!.collecting![item]}`)
    : estimate.data ? t('market.refreshCost', { n: fmtCount(estimate.data.requests), from: fmtDate(estimate.data.since), to: estimate.data.until ? fmtDate(estimate.data.until) : t('market.now') }) : ''
  return (
    <div className="flex flex-col items-end gap-1">
      <Button variant="ghost" title={t('market.refreshHint')} disabled={!item || elsewhere} loading={start.isPending || running}
        onClick={() => start.mutate({ item, seasonId }, { onSuccess: setJobId, onError: e => toast.push((e as Error).message, 'error') })}>
        <RefreshCw size={14} />{t('market.refresh')}</Button>
      <span className={`text-xs ${stream.status === 'failed' ? 'text-[var(--bad)]' : 'text-[var(--muted)]'}`} aria-live="polite">{note}</span>
    </div>
  )
}
