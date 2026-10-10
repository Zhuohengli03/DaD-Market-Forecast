import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { AlertTriangle } from 'lucide-react'
import { useSaveSchedule } from '../../api/hooks'
import { Button } from '../ui/Button'
import { inputCls } from '../ui/Field'
import { useToast } from '../ui/Toast'
import { fmtCount, monthlyRequests } from '../../lib/quota'

export const MIN_SECONDS = 30
const UNITS = [['d', 86400], ['h', 3600], ['m', 60], ['s', 1]] as const
type Parts = Record<'d' | 'h' | 'm' | 's', number>

export function splitSeconds(total: number): Parts {
  const d = Math.floor(total / 86400), h = Math.floor((total % 86400) / 3600), m = Math.floor((total % 3600) / 60)
  return { d, h, m, s: total % 60 }
}

/** Each scheduled collection costs at least one request per tracked item. */
export function requestsPerMinute(intervalSeconds: number, items: number): number {
  return Math.ceil((items * 60) / Math.max(intervalSeconds, 1))
}

interface Props { initialSeconds: number; initialAuto: boolean; trackedItems: number; rateLimit: number; monthlyQuota?: number; submitLabel: string; onSaved?: () => void }

export function ScheduleForm({ initialSeconds, initialAuto, trackedItems, rateLimit, monthlyQuota = 10000, submitLabel, onSaved }: Props) {
  const { t } = useTranslation()
  const save = useSaveSchedule(); const toast = useToast()
  const [parts, setParts] = useState<Parts>(splitSeconds(initialSeconds))
  const [auto, setAuto] = useState(initialAuto)
  const total = UNITS.reduce((sum, [k, mul]) => sum + (parts[k] || 0) * mul, 0)
  const tooShort = total < MIN_SECONDS
  const perMin = requestsPerMinute(total, Math.max(trackedItems, 1))
  const over = auto && !tooShort && perMin > rateLimit
  const perMonth = monthlyRequests(total, Math.max(trackedItems, 1))
  const overMonth = auto && !tooShort && perMonth > monthlyQuota
  const [ack, setAck] = useState(false)
  return (
    <div className="space-y-3 text-sm">
      <p className="text-[var(--muted)]">{t('wizard.schedule.intro')}</p>
      <div>
        <div className="mb-1 text-[var(--muted)]">{t('wizard.schedule.interval')}</div>
        <div className="flex flex-wrap items-center gap-2">
          {UNITS.map(([k]) => (
            <label key={k} className="flex items-center gap-1">
              <input aria-label={t(`wizard.schedule.unit.${k}`)} type="number" min={0} className={`${inputCls} w-20`} value={parts[k]}
                onChange={e => setParts(p => ({ ...p, [k]: Math.max(0, Math.floor(Number(e.target.value) || 0)) }))} />
              <span>{t(`wizard.schedule.unit.${k}`)}</span>
            </label>))}
        </div>
        {tooShort && <p role="alert" className="mt-1 text-xs text-[var(--bad)]">{t('wizard.schedule.min')}</p>}
      </div>
      <label className="flex items-center gap-2"><input type="checkbox" checked={auto} onChange={e => setAuto(e.target.checked)} />{t('wizard.schedule.auto')}</label>
      <div className={`rounded-md border px-3 py-2 ${over ? 'border-red-200 bg-red-50' : 'border-amber-200 bg-amber-50'}`}>
        <p className="flex gap-2"><AlertTriangle size={16} className={over ? 'text-[var(--bad)]' : 'text-[var(--warn)]'} />{t('wizard.schedule.quota')}</p>
        {!tooShort && <p className="mt-1 text-xs">{t('wizard.schedule.estimate', { items: trackedItems, perMin, perDay: Math.ceil((trackedItems * 86400) / total) })}</p>}
        {over && <p className="mt-1 text-xs font-medium text-[var(--bad)]">{t('wizard.schedule.over', { limit: rateLimit })}</p>}
        {!tooShort && <p className="mt-1 text-xs">{t('wizard.schedule.month', { n: fmtCount(perMonth), quota: fmtCount(monthlyQuota) })}</p>}
        {overMonth && <>
          <p className="mt-1 text-xs font-medium text-[var(--bad)]">{t('wizard.schedule.overMonth', { days: Math.max(1, Math.floor((monthlyQuota / perMonth) * 30)) })}</p>
          <label className="mt-1 flex items-center gap-2 text-xs"><input type="checkbox" checked={ack} onChange={e => setAck(e.target.checked)} />{t('wizard.schedule.ack')}</label>
        </>}
      </div>
      <p className="text-xs text-[var(--muted)]">{t('wizard.schedule.hint')}</p>
      <Button disabled={tooShort || (overMonth && !ack)} loading={save.isPending} onClick={() => save.mutate({ interval_seconds: total, auto_collect: auto }, { onSuccess: () => { toast.push(t('settings.saved')); onSaved?.() }, onError: e => toast.push((e as Error).message, 'error') })}>{submitLabel}</Button>
    </div>
  )
}
