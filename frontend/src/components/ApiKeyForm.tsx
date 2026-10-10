import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSaveApiKey } from '../api/hooks'
import { Button } from './ui/Button'
import { inputCls } from './ui/Field'

export const KEYS_URL = 'https://darkerdb.com/dashboard/api-keys'

export function ApiKeyForm({ onSaved, extra }: { onSaved?: () => void; extra?: React.ReactNode }) {
  const { t } = useTranslation()
  const save = useSaveApiKey()
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    save.mutate(key.trim(), { onSuccess: () => { setKey(''); onSaved?.() }, onError: err => setError((err as Error).message) })
  }
  return (
    <form onSubmit={submit} className="space-y-3">
      <p className="text-sm text-[var(--muted)]">
        {t('apiKey.steps', { site: t('apiKey.site') })}{' '}
        <a href={KEYS_URL} target="_blank" rel="noreferrer" className="text-[var(--accent)] underline">darkerdb.com/dashboard/api-keys</a>
      </p>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-[var(--muted)]">{t('apiKey.label')}</span>
        <div className="flex gap-2">
          <input aria-label={t('apiKey.label')} className={`${inputCls} flex-1 font-mono`} type={show ? 'text' : 'password'} autoComplete="off" spellCheck={false}
            placeholder={t('apiKey.placeholder')} value={key} onChange={e => setKey(e.target.value)} />
          <Button variant="ghost" size="sm" onClick={() => setShow(s => !s)}>{show ? t('apiKey.hide') : t('apiKey.show')}</Button>
        </div>
        {error && <span role="alert" className="text-xs text-[var(--bad)]">{error}</span>}
      </label>
      <div className="flex gap-2 justify-end">
        {extra}
        <button type="submit" disabled={!key.trim() || save.isPending}
          className="inline-flex items-center gap-2 rounded-md font-medium px-3.5 py-2 text-sm bg-[var(--accent)] text-white disabled:opacity-50">{t('apiKey.save')}</button>
      </div>
    </form>
  )
}
