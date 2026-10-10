import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import * as echarts from 'echarts'
import { AlertTriangle, Copy, Download } from 'lucide-react'
import { useDeleteReport, useEditReport, useReport } from '../../api/hooks'
import type { Report } from '../../api/types'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { useToast } from '../../components/ui/Toast'
import { fmtDate, fmtNum, fmtPct } from '../../lib/format'
import { fmtCount } from '../../lib/quota'
import { seasonLabel } from '../../lib/labels'
import { Markdown } from './Markdown'

const PARTS = ['public', 'body'] as const

/** A PNG of this period's price changes, to paste next to the report. */
export function moversChartPng(r: Report, title: string): string {
  const rows = r.facts.items.filter(i => i.change_pct != null).sort((a, b) => (b.change_pct ?? 0) - (a.change_pct ?? 0))
  const el = document.createElement('div')
  el.style.cssText = `width:800px;height:${Math.max(240, 40 + rows.length * 28)}px;position:fixed;left:-10000px`
  document.body.appendChild(el)
  const chart = echarts.init(el, undefined, { renderer: 'canvas' })
  chart.setOption({ animation: false, backgroundColor: '#ffffff', title: { text: title, left: 'center', textStyle: { fontSize: 14 } },
    grid: { left: 160, right: 60, top: 40, bottom: 20 }, xAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
    yAxis: { type: 'category', inverse: true, data: rows.map(i => i.name_zh || i.name) },
    series: [{ type: 'bar', data: rows.map(i => ({ value: i.change_pct, itemStyle: { color: (i.change_pct ?? 0) >= 0 ? '#16a34a' : '#dc2626' } })),
      label: { show: true, position: 'right', formatter: (p: { value: number }) => `${p.value > 0 ? '+' : ''}${p.value}%` } }] })
  const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#ffffff' })
  chart.dispose(); el.remove()
  return url
}

export function ReportView({ id, onDeleted }: { id: string; onDeleted: () => void }) {
  const { t } = useTranslation()
  const report = useReport(id); const edit = useEditReport(); const del = useDeleteReport(); const toast = useToast()
  const [tab, setTab] = useState<typeof PARTS[number]>('public')
  const [draft, setDraft] = useState({ public: '', body: '' })
  const [editing, setEditing] = useState(false)
  useEffect(() => { if (report.data) { setDraft({ public: report.data.public, body: report.data.body }); setEditing(false) } }, [report.data])
  if (!report.data) return <p className="text-sm">{t('common.loading')}</p>
  const r = report.data
  const dirty = draft.public !== r.public || draft.body !== r.body
  const copy = async (text: string) => { try { await navigator.clipboard.writeText(text); toast.push(t('reports.copied')) } catch { toast.push(t('reports.copyFailed'), 'error') } }
  const downloadChart = () => {
    const a = document.createElement('a')
    a.href = moversChartPng(r, t('reports.chartTitle', { from: fmtDate(r.period.start), to: fmtDate(r.period.end) }))
    a.download = `darker-market-${fmtDate(r.period.end)}.png`; a.click()
  }
  return (
    <div className="space-y-4">
      <Card title={t('reports.titleOf', { season: seasonLabel(r.season.name, t), from: fmtDate(r.period.start), to: fmtDate(r.period.end) })} actions={<>
        <Button size="sm" variant="ghost" onClick={downloadChart}><Download size={14} />{t('reports.chart')}</Button>
        <Button size="sm" variant="danger" loading={del.isPending} onClick={() => window.confirm(t('reports.confirmDelete')) && del.mutate(r.id, { onSuccess: onDeleted })}>{t('common.delete')}</Button></>}>
        <p className="text-xs text-[var(--muted)]">{t('reports.meta', { model: r.model, items: r.facts.items.length, tokens: fmtCount(r.usage.prompt_tokens + r.usage.completion_tokens) })}{r.cost != null ? ` · ${t('reports.costYuan', { n: r.cost.toFixed(4) })}` : ''}{r.edited ? ` · ${t('reports.edited')}` : ''}</p>
        {r.unmatched.length > 0 && <div role="alert" className="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm flex gap-2">
          <AlertTriangle size={16} className="text-[var(--warn)] shrink-0 mt-0.5" />
          <span>{t('reports.unmatched', { list: r.unmatched.join('、') })}</span></div>}
        <div role="tablist" className="mt-3 flex gap-1.5 text-sm">
          {PARTS.map(p => <button key={p} type="button" role="tab" aria-selected={tab === p} onClick={() => setTab(p)}
            className={`px-3 py-1 rounded-full border ${tab === p ? 'bg-[var(--accent)] text-white border-[var(--accent)]' : 'border-[var(--line)] text-[var(--muted)]'}`}>{t(`reports.parts.${p}`)}</button>)}
          <span className="ml-auto flex gap-1.5">
            <Button size="sm" variant="ghost" onClick={() => setEditing(e => !e)}>{editing ? t('reports.preview') : t('reports.edit')}</Button>
            <Button size="sm" variant="ghost" onClick={() => copy(draft[tab])}><Copy size={14} />{t('reports.copy')}</Button>
          </span>
        </div>
        <p className="mt-2 text-xs text-[var(--muted)]">{t(`reports.partHint.${tab}`)}</p>
        <div className="mt-2">
          {editing ? <textarea aria-label={t(`reports.parts.${tab}`)} className="w-full min-h-[320px] rounded-md border border-[var(--line)] p-3 font-mono text-sm" value={draft[tab]} onChange={e => setDraft(d => ({ ...d, [tab]: e.target.value }))} />
            : <div className="rounded-md border border-[var(--line)] p-4 bg-white"><Markdown text={draft[tab] || t('reports.emptyPart')} /></div>}
        </div>
        {dirty && <div className="mt-2 flex gap-2"><Button size="sm" loading={edit.isPending} onClick={() => edit.mutate({ id: r.id, ...draft }, { onSuccess: () => toast.push(t('settings.saved')), onError: e => toast.push((e as Error).message, 'error') })}>{t('common.save')}</Button>
          <Button size="sm" variant="ghost" onClick={() => setDraft({ public: r.public, body: r.body })}>{t('common.cancel')}</Button></div>}
      </Card>
      <Card title={t('reports.facts')}>
        <p className="text-xs text-[var(--muted)] mb-2">{t('reports.factsHint')}</p>
        <div className="overflow-x-auto"><table className="w-full text-sm whitespace-nowrap">
          <thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('common.item')}</th><th className="text-right pl-3">{t('reports.median')}</th><th className="text-right pl-3">{t('reports.change')}</th><th className="text-right pl-3">{t('reports.range')}</th><th className="text-right pl-3">{t('reports.listings')}</th><th className="text-right pl-3">{t('reports.volatility')}</th><th className="pl-3">{t('reports.forecast')}</th></tr></thead>
          <tbody>{r.facts.items.map(i => (
            <tr key={i.name} className="border-t border-[var(--line)]">
              <td className="py-1">{i.name_zh || i.name}</td>
              <td className="text-right pl-3 tabular-nums">{fmtNum(i.median)}</td>
              <td className={`text-right pl-3 tabular-nums ${i.change_pct == null ? 'text-[var(--muted)]' : i.change_pct >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}`}>{i.change_pct == null ? '–' : fmtPct(i.change_pct / 100)}</td>
              <td className="text-right pl-3 tabular-nums text-[var(--muted)]">{fmtNum(i.low)} ~ {fmtNum(i.high)}</td>
              <td className="text-right pl-3 tabular-nums">{i.listings}</td>
              <td className="text-right pl-3 tabular-nums text-[var(--muted)]">{i.volatility_pct}%</td>
              <td className="pl-3 text-xs">{!i.forecast ? <span className="text-[var(--muted)]">{t('reports.noForecast')}</span> : i.forecast.beats_baseline ? t('reports.forecastDir', { h: i.forecast.horizon_hours, p: fmtPct((i.forecast.change_pct ?? 0) / 100) }) : <span className="text-[var(--muted)]">{t('reports.forecastNoEdge')}</span>}</td>
            </tr>))}</tbody></table></div>
        {r.facts.skipped.length > 0 && <p className="mt-2 text-xs text-[var(--muted)]">{t('reports.skipped', { n: r.facts.skipped.length })}</p>}
      </Card>
    </div>
  )
}
