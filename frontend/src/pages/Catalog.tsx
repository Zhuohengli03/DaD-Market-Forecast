import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQueryClient } from '@tanstack/react-query'
import { useCatalog, useCatalogJob, useCatalogMeta, useItems, useSetup } from '../api/hooks'
import { useJobStream } from '../api/jobStream'
import { LogPanel } from '../components/LogPanel'
import { Notice } from '../components/Notice'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { inputCls } from '../components/ui/Field'
import { useToast } from '../components/ui/Toast'
import { fmtTs } from '../lib/format'
import { useDebounced } from '../lib/useDebounced'
import { CatalogTable } from './catalog/CatalogTable'

const PAGE = 50

export default function Catalog() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const toast = useToast()
  const setup = useSetup(); const meta = useCatalogMeta(); const items = useItems(); const job = useCatalogJob()
  const [q, setQ] = useState(''); const [type, setType] = useState(''); const [stats, setStats] = useState<'' | 'true' | 'false'>('')
  const [page, setPage] = useState(1)
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  const query = useDebounced(q, 300)
  useEffect(() => { setPage(1) }, [query, type, stats])
  useEffect(() => {
    if (stream.status === 'succeeded' || stream.status === 'failed') { qc.invalidateQueries({ queryKey: ['catalog'] }); qc.invalidateQueries({ queryKey: ['catalogMeta'] }) }
  }, [stream.status, qc])
  const cat = useCatalog({ q: query || undefined, item_type: type || undefined, has_stats: stats === '' ? undefined : stats === 'true', page, page_size: PAGE })
  const tracked = new Set((items.data ?? []).map(i => i.item_id ?? i.name))
  const run = (kind: 'sync' | 'check') => job.mutate(kind, { onSuccess: setJobId, onError: e => toast.push((e as Error).message, 'error') })
  const pages = Math.max(1, Math.ceil((cat.data?.total ?? 0) / PAGE))
  const noKey = setup.data && !setup.data.api_key_configured
  const m = meta.data
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('catalog.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('catalog.intro')}</p></div>
      {noKey && <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noKey')}</Notice>}
      <Card title={m?.items ? t('catalog.summary', { types: m.types, withStats: m.with_stats }) : t('catalog.title')} actions={<>
        <Button size="sm" variant="ghost" disabled={!!noKey} loading={job.isPending} onClick={() => run('sync')}>{t('catalog.sync')}</Button>
        <Button size="sm" disabled={!!noKey || !m?.items} loading={job.isPending} onClick={() => run('check')}>{m?.trade_check_complete ? t('catalog.recheck') : t('catalog.check')}</Button></>}>
        <p className="text-xs text-[var(--muted)] mb-2">{t('catalog.onlyTradable')}</p>
        {m?.items && !m.trade_check_complete ? <p className="text-xs text-[var(--warn)] mb-2">{t('catalog.notChecked')} {m.trade_checked > 0 && t('catalog.checking', { done: m.trade_checked, total: m.types_total })}</p> : null}
        <dl className="text-xs grid grid-cols-1 md:grid-cols-3 gap-2 text-[var(--muted)]">
          <div>{t('catalog.patch')}: <b className="text-[var(--ink)]">{m?.patch ?? '–'}</b></div>
          <div>{t('catalog.syncedAt')}: <b className="text-[var(--ink)]">{fmtTs(m?.synced_at)}</b></div>
          <div>{t('catalog.checkedAt')}: <b className="text-[var(--ink)]">{fmtTs(m?.trade_checked_at)}</b></div>
        </dl>
        {jobId && <div className="mt-3"><LogPanel lines={stream.lines} status={stream.status} disconnected={stream.disconnected} /></div>}
      </Card>
      <div className="flex flex-wrap gap-2 items-center text-sm">
        <input className={`${inputCls} w-64`} placeholder={t('catalog.search')} value={q} onChange={e => setQ(e.target.value)} aria-label={t('catalog.search')} />
        <select className={inputCls} value={type} onChange={e => setType(e.target.value)} aria-label={t('catalog.type')}><option value="">{t('catalog.type')}: {t('common.all')}</option>{cat.data?.facets.item_type.map(r => <option key={r}>{r}</option>)}</select>
        <select className={inputCls} value={stats} onChange={e => setStats(e.target.value as typeof stats)} aria-label={t('catalog.stats')}><option value="">{t('catalog.stats')}: {t('common.all')}</option><option value="false">{t('catalog.noStats')}</option><option value="true">{t('catalog.withStats')}</option></select>
        <span className="ml-auto text-[var(--muted)]">{t('market.page', { page, pages, total: cat.data?.total ?? 0 })}</span>
        <Button variant="ghost" size="sm" aria-label={t('common.prev')} disabled={page <= 1} onClick={() => setPage(p => p - 1)}>‹</Button>
        <Button variant="ghost" size="sm" aria-label={t('common.next')} disabled={page >= pages} onClick={() => setPage(p => p + 1)}>›</Button>
      </div>
      {!m?.items ? <EmptyState title={t('catalog.empty')} /> : !cat.data?.rows.length ? <EmptyState title={t('catalog.noMatch')} /> :
        <CatalogTable rows={cat.data.rows} tracked={tracked} />}
    </div>
  )
}
