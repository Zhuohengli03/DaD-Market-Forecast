import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useActuals, useDeleteRun, useRun } from '../../api/hooks'
import { ForecastChart } from '../../components/charts/ForecastChart'
import { HorizonErrorChart } from '../../components/charts/HorizonErrorChart'
import { ImportanceChart } from '../../components/charts/ImportanceChart'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import { useToast } from '../../components/ui/Toast'
import { download, toCsv } from '../../lib/format'
import { BacktestTable } from './BacktestTable'
import { RunStatusBanner } from './RunStatusBanner'

export function RunResult({ runId, onDeleted }: { runId: string; onDeleted: () => void }) {
  const { t } = useTranslation()
  const run = useRun(runId)
  const [showRefs, setShowRefs] = useState(true)
  const [showActuals, setShowActuals] = useState(true)
  const actuals = useActuals(runId, showActuals)
  const del = useDeleteRun()
  const toast = useToast()
  useEffect(() => { if (run.isError) onDeleted() }, [run.isError, onDeleted])
  if (run.isError) return <EmptyState title={t('forecast.runMissing')} hint={(run.error as Error | null)?.message} />
  if (run.isLoading || !run.data) return <p className="text-sm">{t('common.loading')}</p>
  const r = run.data
  const exportCsv = () => download(`${r.run_id}.csv`, toCsv(r.forecast.map(f => ({ ts: f.ts, h: f.h, pred: f.pred, lower: f.lower, upper: f.upper }))))
  return (
    <div className="space-y-4">
      <RunStatusBanner run={r} />
      <Card actions={<>
        <label className="text-xs flex items-center gap-1"><input type="checkbox" checked={showRefs} onChange={e => setShowRefs(e.target.checked)} />{t('forecast.references')}</label>
        <label className="text-xs flex items-center gap-1"><input type="checkbox" checked={showActuals} onChange={e => setShowActuals(e.target.checked)} />{t('forecast.actuals')}</label>
        <Button size="sm" variant="ghost" onClick={exportCsv}>{t('common.download')}</Button>
        <Button size="sm" variant="danger" onClick={() => del.mutate(r.run_id, { onSuccess: () => { toast.push(t('forecast.deleteRun')); onDeleted() }, onError: e => toast.push((e as Error).message, 'error') })}>{t('common.delete')}</Button>
      </>}>
        <ForecastChart run={r} showRefs={showRefs} actuals={showActuals ? actuals.data : undefined} />
      </Card>
      {r.warnings.length > 0 && <Card title={t('forecast.warnings')}><ul className="text-sm list-disc pl-5 text-[var(--warn)]">{r.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Card>}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title={t('forecast.backtest')}><BacktestTable run={r} /></Card>
        <Card title={t('forecast.perHorizon')}><HorizonErrorChart run={r} /></Card>
        {r.feature_importance.length > 0 && <Card title={t('forecast.importance')}><ImportanceChart items={r.feature_importance} /></Card>}
      </div>
    </div>
  )
}
