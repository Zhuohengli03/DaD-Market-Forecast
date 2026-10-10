import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CheckCircle2, AlertCircle } from 'lucide-react'
import { useCatalogJob, useDatabaseSetup, useInitialize, useItems, useSetup } from '../../api/hooks'
import { useJobStream } from '../../api/jobStream'
import type { DatabaseInput } from '../../api/types'
import { ApiKeyForm } from '../ApiKeyForm'
import { ProgressPanel } from '../ProgressPanel'
import { Button } from '../ui/Button'
import { Field, inputCls } from '../ui/Field'
import { ItemPicker, usePickOptions } from '../ItemPicker'
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
  const [mode, setMode] = useState<'db' | 'pg'>(env.storage === 'postgres' ? 'pg' : 'db')
  const saved = env.db_saved
  const [form, setForm] = useState<DatabaseInput>({ host: saved?.host ?? 'localhost', port: saved?.port ?? 5432, database: saved?.database ?? 'darkerdb', user: saved?.user ?? '', password: '' })
  const [error, setError] = useState<string | null>(null)
  const set = (k: keyof DatabaseInput, v: string) => setForm(f => ({ ...f, [k]: k === 'port' ? Number(v) : v }))
  // choosing an option only changes what is shown; switching storage always needs an explicit confirmation
  const confirmBuiltIn = () => db.clear.mutate(undefined, { onSuccess: () => toast.push(t('wizard.env.switched')), onError: e => setError((e as Error).message) })
  return (
    <div className="space-y-4 text-sm">
      <div className="flex items-center gap-2">{env.python_ok ? <CheckCircle2 size={16} className="text-[var(--good)]" /> : <AlertCircle size={16} className="text-[var(--bad)]" />}
        {env.python_ok ? t('wizard.env.python', { v: env.python_version }) : t('wizard.env.pythonOld', { v: env.python_version })}</div>
      <fieldset className="space-y-2">
        <legend className="font-medium mb-1">{t('wizard.env.storage')}</legend>
        <label className="flex gap-2 items-start"><input type="radio" name="storage" aria-label={t('wizard.env.builtin')} checked={mode === 'db'} onChange={() => setMode('db')} />
          <span><span>{t('wizard.env.builtin')}</span><span className="block text-xs text-[var(--muted)]">{t('wizard.env.builtinHint', { path: `${env.data_dir}/market.db` })}</span></span></label>
        <label className="flex gap-2 items-start"><input type="radio" name="storage" aria-label={t('wizard.env.pg')} checked={mode === 'pg'} onChange={() => setMode('pg')} />
          <span><span>{t('wizard.env.pg')}</span><span className="block text-xs text-[var(--muted)]">{env.storage === 'postgres' ? t('wizard.env.pgOk', { host: env.db_host, db: env.db_database }) : t('wizard.env.pgHint')}</span></span></label>
      </fieldset>
      {mode === 'db' && env.storage === 'postgres' && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 space-y-2">
          <p>{t('wizard.env.toBuiltinWarn')}</p>
          <Button size="sm" loading={db.clear.isPending} onClick={confirmBuiltIn}>{t('wizard.env.toBuiltin')}</Button>
        </div>)}
      {mode === 'pg' && env.storage !== 'postgres' && (
        <div className="grid grid-cols-2 gap-2">
          {(['host', 'port', 'database', 'user', 'password'] as const).map(k => (
            <Field key={k} label={t(`wizard.env.${k}`)}><input aria-label={t(`wizard.env.${k}`)} className={inputCls} type={k === 'password' ? 'password' : k === 'port' ? 'number' : 'text'} value={String(form[k])} onChange={e => set(k, e.target.value)} /></Field>))}
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
  const { t } = useTranslation()
  const setup = useSetup().data!
  const items = useItems(); const catalogJob = useCatalogJob(); const init = useInitialize(); const toast = useToast()
  const [syncJob, setSyncJob] = useState<string | null>(null)
  const sync = useJobStream(syncJob)
  const stream = useJobStream(jobId)
  const { options } = usePickOptions()
  const [picked, setPicked] = useState<string[] | null>(null)
  const qc = useQueryClient()
  const catalogReady = setup.catalog_items > 0 || sync.status === 'succeeded'
  useEffect(() => {
    // the catalog was empty when these queries first ran; refresh them once the download finishes
    if (sync.status === 'succeeded') ['setup', 'catalog', 'catalogMeta'].forEach(k => qc.invalidateQueries({ queryKey: [k] }))
  }, [sync.status, qc])
  useEffect(() => {
    if (!setup.catalog_items && !syncJob && setup.api_key_configured) catalogJob.mutate('sync', { onSuccess: setSyncJob })
  }, [setup.catalog_items, setup.api_key_configured]) // eslint-disable-line react-hooks/exhaustive-deps
  const tracked = new Set((items.data ?? []).map(i => i.item_id ?? i.name))
  const chosen = picked ?? options.filter(o => tracked.has(o.id) || tracked.has(o.type.name)).map(o => o.id)
  const upperMin = Math.ceil((chosen.length * MAX_PAGES) / 60)
  if (jobId) return <InitProgress status={stream.status} progress={stream.progress} error={stream.error} />
  if (!catalogReady) return <div className="space-y-2"><p className="text-sm">{t('wizard.init.needCatalog')}</p><ProgressPanel kind="catalog" status={sync.status} progress={sync.progress} error={sync.error} disconnected={sync.disconnected} /></div>
  return (
    <div className="space-y-3 text-sm">
      <p>{t('wizard.init.intro')}</p>
      <p className="text-xs text-[var(--muted)]">{t('wizard.init.choose')}</p>
      <ItemPicker options={options} chosen={chosen} onChange={setPicked} />
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-[var(--muted)]">{t('wizard.init.cost', { n: chosen.length })} {t('wizard.init.estimate', { n: chosen.length, time: upperMin >= 120 ? t('wizard.init.hours', { h: Math.ceil(upperMin / 60) }) : t('wizard.init.minutes', { m: upperMin }) })}</span>
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
  const qcRoot = useQueryClient()
  useEffect(() => {
    // initialization changed seasons, tracked items, collected data and forecasts: refresh what the app shows
    if (initStream.status === 'succeeded') ['setup', 'items', 'seasons', 'runs', 'status', 'catalog', 'catalogMeta'].forEach(k => qcRoot.invalidateQueries({ queryKey: [k] }))
  }, [initStream.status, qcRoot])
  const [dismissed, setDismissed] = useState(() => { try { return sessionStorage.getItem('wizardDismissed') === '1' } catch { return false } })
  const [forced, setForced] = useState(false)
  // once shown, the wizard stays until the user finishes or postpones it, even though initialization
  // marks the app initialized half-way through
  const [open, setOpen] = useState(false)
  const shouldOpen = setup.isSuccess && !setup.data?.initialized && !dismissed
  useEffect(() => { if (shouldOpen) setOpen(true) }, [shouldOpen])
  useEffect(() => {
    const reopen = () => { setForced(true); setDismissed(false); setStep(0) }
    window.addEventListener(OPEN_WIZARD_EVENT, reopen)
    return () => window.removeEventListener(OPEN_WIZARD_EVENT, reopen)
  }, [])
  if (!setup.isSuccess) return null
  const s = setup.data
  if (!(forced || open || shouldOpen)) return null
  const dismiss = () => { setDismissed(true); setForced(false); setOpen(false); try { sessionStorage.setItem('wizardDismissed', '1') } catch { /* private mode */ } }
  const close = () => { setForced(false); setOpen(false); setDismissed(true); navigate('/') }
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
          {step === 3 && <ScheduleForm initialSeconds={s.collect_interval_seconds} initialAuto={true} trackedItems={s.tracked_items} rateLimit={s.rate_limit_per_minute} monthlyQuota={s.monthly_quota} submitLabel={t('wizard.finish')} onSaved={close} />}
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
