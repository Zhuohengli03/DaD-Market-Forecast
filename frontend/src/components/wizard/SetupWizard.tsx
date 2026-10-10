import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CheckCircle2, AlertCircle } from 'lucide-react'
import { useCatalog, useCatalogJob, useDatabaseSetup, useInitialize, useItems, useSetup } from '../../api/hooks'
import { useJobStream } from '../../api/jobStream'
import type { DatabaseInput } from '../../api/types'
import { itemName, rarityLabel } from '../../lib/labels'
import { useDebounced } from '../../lib/useDebounced'
import { ApiKeyForm } from '../ApiKeyForm'
import { ProgressPanel } from '../ProgressPanel'
import { Button } from '../ui/Button'
import { Field, inputCls } from '../ui/Field'
import { useToast } from '../ui/Toast'
import { InitProgress } from './InitProgress'
import { ScheduleForm } from './ScheduleForm'

const STEP_KEYS = ['env', 'key', 'init', 'schedule'] as const
const MAX_PAGES = 200
export const OPEN_WIZARD_EVENT = 'open-setup-wizard'

function EnvStep() {
  const { t } = useTranslation()
  const setup = useSetup().data!
  const env = setup.environment
  const db = useDatabaseSetup(); const toast = useToast()
  const [mode, setMode] = useState<'file' | 'pg'>(env.storage === 'postgres' ? 'pg' : 'file')
  const [form, setForm] = useState<DatabaseInput>({ host: 'localhost', port: 5432, database: 'darkerdb', user: '', password: '' })
  const [error, setError] = useState<string | null>(null)
  const set = (k: keyof DatabaseInput, v: string) => setForm(f => ({ ...f, [k]: k === 'port' ? Number(v) : v }))
  const toFile = () => { setMode('file'); if (env.storage === 'postgres') db.clear.mutate(undefined, { onSuccess: () => toast.push(t('wizard.env.switched')) }) }
  return (
    <div className="space-y-4 text-sm">
      <div className="flex items-center gap-2">{env.python_ok ? <CheckCircle2 size={16} className="text-[var(--good)]" /> : <AlertCircle size={16} className="text-[var(--bad)]" />}
        {env.python_ok ? t('wizard.env.python', { v: env.python_version }) : t('wizard.env.pythonOld', { v: env.python_version })}</div>
      <fieldset className="space-y-2">
        <legend className="font-medium mb-1">{t('wizard.env.storage')}</legend>
        <label className="flex gap-2 items-start"><input type="radio" name="storage" aria-label={t('wizard.env.file')} checked={mode === 'file'} onChange={toFile} />
          <span><span>{t('wizard.env.file')}</span><span className="block text-xs text-[var(--muted)]">{t('wizard.env.fileHint', { dir: env.data_dir })}</span></span></label>
        <label className="flex gap-2 items-start"><input type="radio" name="storage" aria-label={t('wizard.env.pg')} checked={mode === 'pg'} onChange={() => setMode('pg')} />
          <span><span>{t('wizard.env.pg')}</span><span className="block text-xs text-[var(--muted)]">{env.storage === 'postgres' ? t('wizard.env.pgOk', { host: env.db_host, db: env.db_database }) : t('wizard.env.pgHint')}</span></span></label>
      </fieldset>
      {mode === 'pg' && env.storage !== 'postgres' && (
        <div className="grid grid-cols-2 gap-2">
          {(['host', 'port', 'database', 'user', 'password'] as const).map(k => (
            <Field key={k} label={t(`wizard.env.${k}`)}><input className={inputCls} type={k === 'password' ? 'password' : k === 'port' ? 'number' : 'text'} value={String(form[k])} onChange={e => set(k, e.target.value)} /></Field>))}
          <div className="col-span-2 flex items-center gap-3">
            <Button size="sm" loading={db.save.isPending} onClick={() => { setError(null); db.save.mutate(form, { onError: e => setError((e as Error).message) }) }}>{t('wizard.env.test')}</Button>
            {error && <span role="alert" className="text-xs text-[var(--bad)]">{error}</span>}
          </div>
        </div>)}
    </div>
  )
}

function KeyStep() {
  const { t } = useTranslation()
  const setup = useSetup().data!
  const [change, setChange] = useState(false)
  if (setup.api_key_configured && !change) {
    return <div className="flex items-center gap-2 text-sm"><CheckCircle2 size={16} className="text-[var(--good)]" />{t('wizard.key.done', { hint: setup.api_key_hint })}
      <Button size="sm" variant="ghost" onClick={() => setChange(true)}>{t('wizard.key.change')}</Button></div>
  }
  return <div className="space-y-2"><p className="text-sm">{t('apiKey.why')}</p><ApiKeyForm onSaved={() => setChange(false)} /></div>
}

function InitStep({ jobId, onStart }: { jobId: string | null; onStart: (id: string) => void }) {
  const { t, i18n } = useTranslation()
  const setup = useSetup().data!
  const items = useItems(); const catalogJob = useCatalogJob(); const init = useInitialize(); const toast = useToast()
  const [syncJob, setSyncJob] = useState<string | null>(null)
  const sync = useJobStream(syncJob)
  const stream = useJobStream(jobId)
  const [q, setQ] = useState(''); const query = useDebounced(q, 300)
  const cat = useCatalog({ q: query || undefined, has_stats: false, page: 1, page_size: 40 })
  const [picked, setPicked] = useState<string[] | null>(null)
  useEffect(() => {
    if (!setup.catalog_items && !syncJob && setup.api_key_configured) catalogJob.mutate('sync', { onSuccess: setSyncJob })
  }, [setup.catalog_items, setup.api_key_configured]) // eslint-disable-line react-hooks/exhaustive-deps
  const options = useMemo(() => (cat.data?.rows ?? []).flatMap(r => r.variants.map(v => ({
    id: v.id, type: r, label: r.variants.length > 1 ? `${itemName(r, i18n.language)}（${rarityLabel(v.rarity, t)}）` : itemName(r, i18n.language),
  }))), [cat.data, i18n.language, t])
  const tracked = new Set((items.data ?? []).map(i => i.item_id ?? i.name))
  const chosen = picked ?? options.filter(o => tracked.has(o.id) || tracked.has(o.type.name)).map(o => o.id)
  const toggle = (id: string) => setPicked(chosen.includes(id) ? chosen.filter(x => x !== id) : [...chosen, id])
  if (jobId) return <InitProgress status={stream.status} progress={stream.progress} error={stream.error} />
  if (!setup.catalog_items) return <div className="space-y-2"><p className="text-sm">{t('wizard.init.needCatalog')}</p><ProgressPanel kind="catalog" status={sync.status} progress={sync.progress} error={sync.error} disconnected={sync.disconnected} /></div>
  return (
    <div className="space-y-3 text-sm">
      <p>{t('wizard.init.intro')}</p>
      <p className="text-xs text-[var(--muted)]">{t('wizard.init.choose')}</p>
      <input className={`${inputCls} w-full`} placeholder={t('wizard.init.search')} value={q} onChange={e => setQ(e.target.value)} />
      <div className="max-h-56 overflow-auto border border-[var(--line)] rounded-md p-2 grid grid-cols-2 gap-1">
        {options.map(o => <label key={o.id} className="flex items-center gap-2"><input type="checkbox" aria-label={o.label} checked={chosen.includes(o.id)} onChange={() => toggle(o.id)} />{o.label}</label>)}
      </div>
      <div className="flex items-center justify-between">
        <span className="text-xs text-[var(--muted)]">{t('wizard.init.estimate', { n: chosen.length, min: Math.ceil(chosen.length * MAX_PAGES * 1.1 / 60) + 1 })}</span>
        <Button disabled={!chosen.length} loading={init.isPending} onClick={() => init.mutate({ item_ids: chosen, max_pages: MAX_PAGES }, { onSuccess: onStart, onError: e => toast.push((e as Error).message, 'error') })}>{t('wizard.init.start')}</Button>
      </div>
    </div>
  )
}

/** First-run setup: environment → API key → initialization → auto-collect. Re-openable from Settings. */
export function SetupWizard() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const setup = useSetup()
  const [step, setStep] = useState(0)
  const [initJob, setInitJob] = useState<string | null>(null)
  const initStream = useJobStream(initJob)
  const [dismissed, setDismissed] = useState(() => { try { return sessionStorage.getItem('wizardDismissed') === '1' } catch { return false } })
  const [forced, setForced] = useState(false)
  useEffect(() => {
    const open = () => { setForced(true); setDismissed(false); setStep(0) }
    window.addEventListener(OPEN_WIZARD_EVENT, open)
    return () => window.removeEventListener(OPEN_WIZARD_EVENT, open)
  }, [])
  if (!setup.isSuccess) return null
  const s = setup.data
  if (!(forced || (!s.initialized && !dismissed))) return null
  const dismiss = () => { setDismissed(true); setForced(false); try { sessionStorage.setItem('wizardDismissed', '1') } catch { /* private mode */ } }
  const close = () => { setForced(false); setDismissed(true); navigate('/') }
  const canNext = step === 0 ? s.environment.python_ok : step === 1 ? s.api_key_configured : step === 2 ? initStream.status === 'succeeded' : true
  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center p-4" onKeyDown={e => { if (e.key === 'Escape') dismiss() }}>
      <div role="dialog" aria-modal="true" aria-labelledby="wizard-title" className="bg-white rounded-xl shadow-xl max-w-2xl w-full p-6 space-y-5 max-h-[90vh] overflow-auto">
        <div><h2 id="wizard-title" className="text-lg font-semibold">{t('wizard.title')}</h2><p className="text-sm text-[var(--muted)]">{t('wizard.intro')}</p></div>
        <ol className="flex gap-2 text-sm">
          {STEP_KEYS.map((k, i) => (
            <li key={k} className={`flex-1 text-center px-2 py-1.5 rounded-md border ${i === step ? 'border-[var(--accent)] text-[var(--accent)] font-medium' : i < step ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-[var(--line)] text-[var(--muted)]'}`}>{i + 1} {t(`wizard.steps.${k}`)}</li>))}
        </ol>
        <div>
          {step === 0 && <EnvStep />}
          {step === 1 && <KeyStep />}
          {step === 2 && <InitStep jobId={initJob} onStart={setInitJob} />}
          {step === 3 && <ScheduleForm initialHours={s.collect_interval_hours} initialAuto={true} submitLabel={t('wizard.finish')} onSaved={close} />}
        </div>
        <div className="flex justify-between items-center border-t border-[var(--line)] pt-4">
          <div className="flex gap-2">
            {step > 0 && <Button variant="ghost" onClick={() => setStep(step - 1)} disabled={step === 2 && initStream.status === 'running'}>{t('wizard.back')}</Button>}
            {step === 0 && <Button variant="ghost" onClick={dismiss}>{t('wizard.later')}</Button>}
          </div>
          {step < 3 && (step === 2 && initStream.status !== 'succeeded'
            ? <Button variant="ghost" disabled={!!initJob && initStream.status !== 'failed'} onClick={() => setStep(3)}>{t('wizard.skipInit')}</Button>
            : <Button disabled={!canNext} onClick={() => setStep(step + 1)}>{t('wizard.next')}</Button>)}
        </div>
      </div>
    </div>
  )
}
