import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useRules, useSaveRulesSource, useSyncRules } from '../../api/hooks'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { fmtRelative } from '../../lib/format'

/** Which rules are in force (season dates, patch filters), and an optional rules service that keeps them current. */
export function RulesCard() {
  const { t } = useTranslation()
  const toast = useToast()
  const q = useRules(); const save = useSaveRulesSource(); const sync = useSyncRules()
  const [url, setUrl] = useState<string | null>(null)
  const [key, setKey] = useState('')
  const d = q.data
  if (!d) return <p className="text-sm">{t('common.loading')}</p>
  const shownUrl = url ?? d.url
  const fail = (e: unknown) => toast.push((e as Error).message, 'error')
  return (
    <div className="space-y-3 text-sm">
      <p className="text-[var(--muted)]">{t('rules.intro')}</p>
      <p>{t('rules.version', { v: d.version, date: d.updated ?? '–' })} · {d.source === 'synced'
        ? t('rules.synced', { when: fmtRelative(d.synced_at) }) : t('rules.bundled')}</p>
      {d.error && <p className="text-[var(--warn)]">{d.error}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <input aria-label={t('rules.url')} placeholder={t('rules.urlPlaceholder')} className={`${inputCls} w-80`} value={shownUrl} onChange={e => setUrl(e.target.value)} />
        <input aria-label={t('rules.key')} type="password" placeholder={d.key_saved ? t('llm.keySaved', { hint: d.key_hint }) : t('rules.key')} className={`${inputCls} w-48`} value={key} onChange={e => setKey(e.target.value)} />
        <Button size="sm" loading={save.isPending} onClick={() => save.mutate({ url: shownUrl.trim(), ...(key.trim() ? { key: key.trim() } : {}) },
          { onSuccess: () => { setKey(''); setUrl(null); toast.push(t('rules.saved'), 'ok') }, onError: fail })}>{t('common.save')}</Button>
        <Button size="sm" variant="ghost" disabled={!d.url} loading={sync.isPending} onClick={() => sync.mutate(undefined, { onError: fail })}>{t('rules.syncNow')}</Button>
      </div>
    </div>
  )
}
