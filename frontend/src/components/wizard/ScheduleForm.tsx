import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSaveSchedule } from '../../api/hooks'
import { Button } from '../ui/Button'
import { Field, inputCls } from '../ui/Field'
import { useToast } from '../ui/Toast'

export const INTERVALS = [1, 2, 3, 6, 12, 24]

/** Collection interval + auto-collect switch; used by the wizard's last step and the Settings page. */
export function ScheduleForm({ initialHours, initialAuto, submitLabel, onSaved }: { initialHours: number; initialAuto: boolean; submitLabel: string; onSaved?: () => void }) {
  const { t } = useTranslation()
  const save = useSaveSchedule(); const toast = useToast()
  const [hours, setHours] = useState(initialHours)
  const [auto, setAuto] = useState(initialAuto)
  return (
    <div className="space-y-3">
      <p className="text-sm text-[var(--muted)]">{t('wizard.schedule.intro')}</p>
      <div className="flex flex-wrap items-end gap-4">
        <Field label={t('wizard.schedule.interval')}>
          <select aria-label={t('wizard.schedule.interval')} className={inputCls} value={hours} onChange={e => setHours(Number(e.target.value))}>
            {INTERVALS.map(h => <option key={h} value={h}>{t('wizard.schedule.every', { h })}</option>)}
          </select>
        </Field>
        <label className="flex items-center gap-2 text-sm pb-2"><input type="checkbox" checked={auto} onChange={e => setAuto(e.target.checked)} />{t('wizard.schedule.auto')}</label>
      </div>
      <p className="text-xs text-[var(--muted)]">{t('wizard.schedule.hint')}</p>
      <Button loading={save.isPending} onClick={() => save.mutate({ interval_hours: hours, auto_collect: auto }, { onSuccess: () => { toast.push(t('settings.saved')); onSaved?.() }, onError: e => toast.push((e as Error).message, 'error') })}>{submitLabel}</Button>
    </div>
  )
}
