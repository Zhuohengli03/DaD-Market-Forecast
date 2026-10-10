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
import { useDebounced } from '../lib/useDebounced'
import { fmtTs } from '../lib/format'
import { CatalogTable } from './catalog/CatalogTable'

const PAGE = 50

export default function Catalog() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const toast = useToast()
  const setup = useSetup(); const meta = useCatalogMeta(); const items = useItems(); const job = useCatalogJob()
  const [q, setQ] = useState(''); const [rarity, setRarity] = useState(''); const [type, setType] = useState('')
  const [tradable, setTradable] = useState<'' | 'true' | 'false'>(''); const [page, setPage] = useState(1)
  const [jobId, setJobId] = useState<string | null>(null)
  const stream = useJobStream(jobId)
  const query = useDebounced(q, 300)
  const [pagesBudget, setPagesBudget] = useState(40)
  useEffect(() => { setPage(1) }, [query, rarity, type, tradable])
  useEffect(() => {
    if (stream.status === 'succeeded' || stream.status === 'failed') { qc.invalidateQueries({ queryKey: ['catalog'] }); qc.invalidateQueries({ queryKey: ['catalogMeta'] }) }
  }, [stream.status, qc])
  const cat = useCatalog({ q: query || undefined, rarity: rarity || undefined, item_type: type || undefined, tradable: tradable === '' ? undefined : tradable === 'true', page, page_size: PAGE })
  const tracked = new Set((items.data ?? []).map(i => i.item_id ?? i.name))
  const run = (kind: 'sync' | 'scan') => job.mutate({ kind, maxPages: pagesBudget }, { onSuccess: setJobId, onError: e => toast.push((e as Error).message, 'error') })
  const pages = Math.max(1, Math.ceil((cat.data?.total ?? 0) / PAGE))
  const noKey = setup.data && !setup.data.api_key_configured
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">{t('catalog.title')}</h1><p className="text-sm text-[var(--muted)] mt-1 max-w-3xl">{t('catalog.intro')}</p></div>
      {noKey && <Notice to="/settings" action={t('common.goSettings')}>{t('onboarding.noKey')}</Notice>}
      <Card actions={<>
        <Button size="sm" variant="ghost" disabled={!!noKey} loading={job.isPending} onClick={() => run('sync')}>{t('catalog.sync')}</Button>
        <select className={inputCls} value={pagesBudget} onChange={e => setPagesBudget(Number(e.target.value))} aria-label={t('catalog.budget')} title={t('catalog.budgetHint')}>
          {[40, 120, 300].map(n => <option key={n} value={n}>{t('catalog.budgetOption', { n: n * 50, min: Math.ceil(n * 1.1 / 60) })}</option>)}</select>
        <Button size="sm" disabled={!!noKey || !meta.data?.items} loading={job.isPending} onClick={() => run('scan')}>{t('catalog.scan')}</Button></>}
        title={t('catalog.summary', { items: meta.data?.items ?? 0, tradable: meta.data?.tradable ?? 0 })}>
        <dl className="text-xs grid grid-cols-2 md:grid-cols-4 gap-2 text-[var(--muted)]">
          <div>{t('catalog.patch')}: <b className="text-[var(--ink)]">{meta.data?.patch ?? '–'}</b></div>
          <div>{t('catalog.syncedAt')}: <b className="text-[var(--ink)]">{fmtTs(meta.data?.synced_at)}</b></div>
          <div>{t('catalog.scannedAt')}: <b className="text-[var(--ink)]">{fmtTs(meta.data?.market_scanned_at)}</b></div>
          <div>{t('catalog.coverage')}: <b className="text-[var(--ink)]">{meta.data?.market_covered_from ? t('catalog.coverageValue', { from: fmtTs(meta.data.market_covered_from), to: fmtTs(meta.data.market_covered_to), n: meta.data.market_listings_read }) : '–'}</b></div>
        </dl>
        {jobId && <div className="mt-3"><LogPanel lines={stream.lines} status={stream.status} disconnected={stream.disconnected} /></div>}
      </Card>
      <div className="flex flex-wrap gap-2 items-center text-sm">
        <input className={`${inputCls} w-64`} placeholder={t('catalog.search')} value={q} onChange={e => setQ(e.target.value)} aria-label={t('catalog.search')} />
        <select className={inputCls} value={rarity} onChange={e => setRarity(e.target.value)} aria-label={t('catalog.rarity')}><option value="">{t('catalog.rarity')}: {t('common.all')}</option>{cat.data?.facets.rarity.map(r => <option key={r}>{r}</option>)}</select>
        <select className={inputCls} value={type} onChange={e => setType(e.target.value)} aria-label={t('catalog.type')}><option value="">{t('catalog.type')}: {t('common.all')}</option>{cat.data?.facets.item_type.map(r => <option key={r}>{r}</option>)}</select>
        <select className={inputCls} value={tradable} onChange={e => setTradable(e.target.value as typeof tradable)} aria-label={t('catalog.tradable')}><option value="">{t('catalog.tradable')}: {t('common.all')}</option><option value="true">{t('catalog.tradableYes')}</option><option value="false">{t('catalog.tradableNo')}</option></select>
        <span className="ml-auto text-[var(--muted)]">{t('market.page', { page, pages, total: cat.data?.total ?? 0 })}</span>
        <Button variant="ghost" size="sm" aria-label={t('common.prev')} disabled={page <= 1} onClick={() => setPage(p => p - 1)}>‹</Button>
        <Button variant="ghost" size="sm" aria-label={t('common.next')} disabled={page >= pages} onClick={() => setPage(p => p + 1)}>›</Button>
      </div>
      {!meta.data?.items ? <EmptyState title={t('catalog.empty')} /> : !cat.data?.rows.length ? <EmptyState title={t('catalog.noMatch')} /> :
        <CatalogTable rows={cat.data.rows} tracked={tracked} />}
    </div>
  )
}
