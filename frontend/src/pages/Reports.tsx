import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Sparkles } from 'lucide-react'
import { useCreateReport, useLLM, useReportEstimate, useReports } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { Notice } from '../components/Notice'
import { ProgressPanel } from '../components/ProgressPanel'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { inputCls } from '../components/ui/Field'
import { useToast } from '../components/ui/Toast'
import { fmtDate, fmtTs } from '../lib/format'
import { fmtCount } from '../lib/quota'
import { useUrlState } from '../lib/urlState'
import { ReportView } from './reports/ReportView'

const DAYS = [7, 14, 30]

export default function Reports() {
  const { t } = useTranslation()
  const qc = useQueryClient(); const toast = useToast()
  const llm = useLLM(); const reports = useReports(); const create = useCreateReport()
  const [id, setId] = useUrlState('report', '')
  const [days, setDays] = useState(7)
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  const configured = !!llm.data?.configured
  const estimate = useReportEstimate(days, configured)
  useEffect(() => {
    if (stream.status === 'succeeded' && stream.resultRef) { qc.invalidateQueries({ queryKey: ['reports'] }); setId(stream.resultRef); setJobId(null) }
  }, [stream.status, stream.resultRef, qc, setId])
  const est = estimate.data
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[280px_1fr] gap-4">
      <Card title={t('reports.history')} className="xl:h-[calc(100vh-6rem)] overflow-auto">
        {!reports.data?.length ? <EmptyState title={t('reports.none')} /> : (
          <ul className="divide-y divide-[var(--line)]">{reports.data.map(r => (
            <li key={r.id}><button type="button" onClick={() => setId(r.id)} className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-50 ${id === r.id ? 'bg-blue-50' : ''}`}>
              <div className="font-medium">{fmtDate(r.period.start)} ~ {fmtDate(r.period.end)}</div>
              <div className="text-xs text-[var(--muted)]">{t('reports.listMeta', { n: r.items, at: fmtTs(r.created_at) })}{r.unmatched ? ` · ⚠ ${r.unmatched}` : ''}</div>
            </button></li>))}</ul>)}
      </Card>
      <div className="space-y-4">
        <div><h1 className="text-xl font-semibold">{t('reports.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('reports.intro')}</p></div>
        {llm.isSuccess && !configured && <Notice to="/settings" action={t('common.goSettings')}>{t('reports.needKey')}</Notice>}
        <Card>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <label className="flex items-center gap-2">{t('reports.period')}
              <select aria-label={t('reports.period')} className={inputCls} value={days} onChange={e => setDays(Number(e.target.value))}>{DAYS.map(d => <option key={d} value={d}>{t('reports.days', { n: d })}</option>)}</select></label>
            <Button disabled={!configured || !est} loading={create.isPending || stream.status === 'running' || stream.status === 'queued'}
              onClick={() => create.mutate(days, { onSuccess: setJobId, onError: e => toast.push((e as Error).message, 'error') })}><Sparkles size={14} />{t('reports.generate')}</Button>
          </div>
          <p className="mt-2 text-xs text-[var(--muted)]">
            {estimate.isError ? (estimate.error as Error).message
              : est ? t('reports.estimate', { n: est.items, from: fmtDate(est.period.start), to: fmtDate(est.period.end), tin: fmtCount(est.prompt_tokens), tout: fmtCount(est.completion_tokens) }) + (est.cost != null ? ` ${t('reports.estimateCost', { n: est.cost.toFixed(4) })}` : '')
              : configured ? t('cost.estimating') : ''}</p>
          <p className="mt-1 text-xs text-[var(--muted)]">{est?.scope === 'all' ? t('reports.scopeAll') : t('reports.scopeHint')}</p>
          {jobId && <div className="mt-3"><ProgressPanel kind="report" status={stream.status} progress={stream.progress} error={stream.error} disconnected={stream.disconnected} /></div>}
        </Card>
        {id && <ReportView id={id} onDeleted={() => setId('')} />}
      </div>
    </div>
  )
}
