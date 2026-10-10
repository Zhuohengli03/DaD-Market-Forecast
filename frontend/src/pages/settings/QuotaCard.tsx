import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuota, useSaveQuota } from '../../api/hooks'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { fmtRelative } from '../../lib/format'
import { fmtCount } from '../../lib/quota'

/** The plan's monthly credits (used for estimates and warnings) and what DarkerDB last reported as left. */
export function QuotaCard({ initial }: { initial: number }) {
  const { t } = useTranslation()
  const quota = useQuota().data; const save = useSaveQuota(); const toast = useToast()
  const [value, setValue] = useState(String(initial))
  const n = Number(value)
  const bad = !Number.isInteger(n) || n < 100
  return (
    <div className="space-y-3 text-sm">
      <p className="text-[var(--muted)]">{t('quota.intro')}</p>
      <p>{quota?.credits_remaining == null ? t('quota.unknown') : t('quota.leftLong', { n: fmtCount(quota.credits_remaining), when: fmtRelative(quota.updated_at) })}</p>
      <label className="flex flex-wrap items-center gap-2">{t('quota.monthly')}
        <input type="number" min={100} aria-label={t('quota.monthly')} className={`${inputCls} w-32`} value={value} onChange={e => setValue(e.target.value)} />{t('quota.unit')}
        <Button size="sm" disabled={bad} loading={save.isPending} onClick={() => save.mutate(n, { onSuccess: () => toast.push(t('settings.saved')), onError: e => toast.push((e as Error).message, 'error') })}>{t('common.save')}</Button>
      </label>
      {bad && <p role="alert" className="text-xs text-[var(--bad)]">{t('quota.tooSmall')}</p>}
    </div>
  )
}
