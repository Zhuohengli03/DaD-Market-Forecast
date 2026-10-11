import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSaveSchedule, useScheduleEstimate } from '../../api/hooks'
import { MIN_SECONDS, splitSeconds } from '../../components/wizard/ScheduleForm'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { fmtCount } from '../../lib/quota'
import { useDebounced } from '../../lib/useDebounced'

const UNITS = [['d', 86400], ['h', 3600], ['m', 60], ['s', 1]] as const

/** The interval used by every item without its own schedule. Saving keeps scheduled collection on or off as it is. */
export function DefaultInterval({ seconds, running, monthlyQuota }: { seconds: number; running: boolean; monthlyQuota: number }) {
  const { t } = useTranslation()
  const save = useSaveSchedule(); const toast = useToast()
  const [parts, setParts] = useState(splitSeconds(seconds))
  const total = UNITS.reduce((n, [k, mul]) => n + (parts[k] || 0) * mul, 0)
  const tooShort = total < MIN_SECONDS
  const changed = total !== seconds
  const preview = useScheduleEstimate(useDebounced(changed && !tooShort ? total : undefined, 400))
  const perMonth = changed ? preview.data?.per_month : undefined
  return (
    <div className="space-y-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{t('scheduler.defaultInterval')}</span>
        <span className="text-[var(--muted)]">{t('scheduler.every')}</span>
        {UNITS.map(([k]) => (
          <label key={k} className="flex items-center gap-1">
            <input aria-label={t(`wizard.schedule.unit.${k}`)} type="number" min={0} className={`${inputCls} w-16`} value={parts[k]}
              onChange={e => setParts(p => ({ ...p, [k]: Math.max(0, Math.floor(Number(e.target.value) || 0)) }))} />
            {t(`wizard.schedule.unit.${k}`)}
          </label>))}
        <Button size="sm" disabled={!changed || tooShort} loading={save.isPending}
          onClick={() => save.mutate({ interval_seconds: total, auto_collect: running }, { onSuccess: () => toast.push(t('scheduler.saved')), onError: e => toast.push((e as Error).message, 'error') })}>{t('common.save')}</Button>
      </div>
      {tooShort && <p role="alert" className="text-xs text-[var(--bad)]">{t('wizard.schedule.min')}</p>}
      {perMonth != null && <p className={`text-xs ${perMonth > monthlyQuota ? 'text-[var(--bad)] font-medium' : 'text-[var(--muted)]'}`}>{t('scheduler.previewMonth', { n: fmtCount(perMonth), quota: fmtCount(monthlyQuota) })}</p>}
      <p className="text-xs text-[var(--muted)]">{t('wizard.schedule.quota')}</p>
    </div>
  )
}
