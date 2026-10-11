import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useLLM, useSaveLLM, useTestLLM } from '../../api/hooks'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'

const MODELS = ['qwen-plus', 'qwen-max', 'qwen-turbo']
const REGIONS = ['cn', 'intl', 'us'] as const

/** Alibaba Cloud Bailian settings for AI market reports: key, region, optional workspace, model and prices. */
export function LLMCard() {
  const { t } = useTranslation()
  const llm = useLLM(); const save = useSaveLLM(); const test = useTestLLM(); const toast = useToast()
  const [key, setKey] = useState('')
  const [model, setModel] = useState('qwen-plus')
  const [region, setRegion] = useState<typeof REGIONS[number]>('cn')
  const [workspace, setWorkspace] = useState('')
  const [priceIn, setPriceIn] = useState('')
  const [priceOut, setPriceOut] = useState('')
  useEffect(() => {
    if (!llm.data) return
    setModel(llm.data.model); setRegion(llm.data.region); setWorkspace(llm.data.workspace)
    setPriceIn(llm.data.price_in == null ? '' : String(llm.data.price_in)); setPriceOut(llm.data.price_out == null ? '' : String(llm.data.price_out))
  }, [llm.data])
  if (!llm.data) return <p className="text-sm">{t('common.loading')}</p>
  const num = (v: string) => (v.trim() === '' ? null : Number(v))
  const badPrice = [priceIn, priceOut].some(v => v.trim() !== '' && !(Number(v) >= 0))
  const onSave = () => save.mutate({ ...(key.trim() ? { api_key: key.trim() } : {}), model: model.trim(), region, workspace: workspace.trim(), price_in: num(priceIn), price_out: num(priceOut) },
    { onSuccess: () => { setKey(''); toast.push(t('settings.saved')) }, onError: e => toast.push((e as Error).message, 'error') })
  return (
    <div className="space-y-3 text-sm">
      <p className="text-[var(--muted)]">{t('llm.intro')}</p>
      <label className="flex flex-wrap items-center gap-2">{t('llm.key')}
        <input type="password" autoComplete="off" aria-label={t('llm.key')} className={`${inputCls} w-72`} value={key} onChange={e => setKey(e.target.value)}
          placeholder={llm.data.configured ? t('llm.keySaved', { hint: llm.data.key_hint }) : 'sk-…'} />
        <a className="text-xs text-[var(--accent)] hover:underline" href="https://bailian.console.aliyun.com/?tab=model#/api-key" target="_blank" rel="noreferrer">{t('llm.getKey')}</a>
      </label>
      <div className="flex flex-wrap gap-3">
        <label className="flex items-center gap-2">{t('llm.region')}
          <select aria-label={t('llm.region')} className={inputCls} value={region} onChange={e => setRegion(e.target.value as typeof region)}>
            {REGIONS.map(r => <option key={r} value={r}>{t(`llm.regions.${r}`)}</option>)}</select></label>
        <label className="flex items-center gap-2">{t('llm.workspace')}
          <input aria-label={t('llm.workspace')} className={`${inputCls} w-40`} value={workspace} onChange={e => setWorkspace(e.target.value)} placeholder={t('llm.optional')} /></label>
        <label className="flex items-center gap-2">{t('llm.model')}
          <input aria-label={t('llm.model')} list="llm-models" className={`${inputCls} w-40`} value={model} onChange={e => setModel(e.target.value)} />
          <datalist id="llm-models">{MODELS.map(m => <option key={m} value={m} />)}</datalist></label>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <span>{t('llm.prices')}</span>
        <label className="flex items-center gap-1">{t('llm.priceIn')}<input type="number" min={0} step="any" aria-label={t('llm.priceIn')} className={`${inputCls} w-24`} value={priceIn} onChange={e => setPriceIn(e.target.value)} placeholder={t('llm.optional')} /></label>
        <label className="flex items-center gap-1">{t('llm.priceOut')}<input type="number" min={0} step="any" aria-label={t('llm.priceOut')} className={`${inputCls} w-24`} value={priceOut} onChange={e => setPriceOut(e.target.value)} placeholder={t('llm.optional')} /></label>
        <span className="text-xs text-[var(--muted)]">{t('llm.pricesHint')}</span>
      </div>
      <p className="text-xs text-[var(--muted)] break-all">{t('llm.address', { url: llm.data.base_url })}</p>
      <div className="flex gap-2">
        <Button size="sm" disabled={badPrice || !model.trim()} loading={save.isPending} onClick={onSave}>{t('common.save')}</Button>
        <Button size="sm" variant="ghost" disabled={!llm.data.configured} loading={test.isPending}
          onClick={() => test.mutate(undefined, { onSuccess: d => toast.push(t('llm.testOk', { n: d.tokens })), onError: e => toast.push((e as Error).message, 'error') })}>{t('llm.test')}</Button>
      </div>
    </div>
  )
}
