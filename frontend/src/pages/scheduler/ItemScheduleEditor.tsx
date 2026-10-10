import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { Plus, X } from 'lucide-react'
import { useSetItemSchedule, useSetScheduleBulk } from '../../api/hooks'
import type { ItemConfigEntry, ScheduleInput } from '../../api/types'
import { MIN_SECONDS, splitSeconds } from '../../components/wizard/ScheduleForm'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'

const UNITS = [['d', 86400], ['h', 3600], ['m', 60], ['s', 1]] as const
const MAX_SECONDS = 30 * 86400
const TIME_RE = /^([01]\d|2[0-3]):[0-5]\d$/

/** "1天2时" style text for a number of seconds, skipping zero parts. */
export function fmtInterval(seconds: number, t: TFunction): string {
  const p = splitSeconds(seconds)
  const text = UNITS.filter(([k]) => p[k]).map(([k]) => `${p[k]}${t(`wizard.schedule.unit.${k}`)}`).join('')
  return text || `0${t('wizard.schedule.unit.s')}`
}

/** What the backend stores for an item, as the editor's starting point. */
export function toInput(schedule: ItemConfigEntry['schedule']): ScheduleInput {
  if (typeof schedule === 'string' && schedule) return { mode: 'daily', times: [schedule] }
  if (schedule && typeof schedule === 'object') return { mode: schedule.mode, seconds: schedule.seconds, times: schedule.times }
  return { mode: 'default' }
}

/** "每 6时" / "每天 08:00、20:00" for an item's own schedule, or the default interval. */
export function scheduleText(schedule: ItemConfigEntry['schedule'], defaultSeconds: number, t: TFunction): string {
  const s = toInput(schedule)
  if (s.mode === 'interval' && s.seconds) return t('scheduler.everyText', { v: fmtInterval(s.seconds, t) })
  if (s.mode === 'daily' && s.times?.length) return t('scheduler.dailyText', { v: s.times.join('、') })
  return t('scheduler.everyText', { v: fmtInterval(defaultSeconds, t) })
}

/** Returns an error key, or null when the schedule can be saved. */
export function validate(s: ScheduleInput): string | null {
  if (s.mode === 'interval') return !s.seconds || s.seconds < MIN_SECONDS ? 'tooShort' : s.seconds > MAX_SECONDS ? 'tooLong' : null
  if (s.mode === 'daily') return !s.times?.length ? 'noTimes' : s.times.some(x => !TIME_RE.test(x)) ? 'badTime' : null
  return null
}

/** `names`: apply the same schedule to several items at once (e.g. every quality of one type). */
interface Props { name: string; names?: string[]; schedule: ItemConfigEntry['schedule']; defaultSeconds: number; onDone: () => void }

export function ItemScheduleEditor({ name, names, schedule, defaultSeconds, onDone }: Props) {
  const { t } = useTranslation()
  const save = useSetItemSchedule(); const bulk = useSetScheduleBulk(); const toast = useToast()
  const start = toInput(schedule)
  const [mode, setMode] = useState<ScheduleInput['mode']>(start.mode)
  const [parts, setParts] = useState(splitSeconds(start.seconds ?? defaultSeconds))
  const [times, setTimes] = useState<string[]>(start.times?.length ? start.times : ['08:00', '20:00'])
  const seconds = UNITS.reduce((sum, [k, mul]) => sum + (parts[k] || 0) * mul, 0)
  const input: ScheduleInput = mode === 'interval' ? { mode, seconds } : mode === 'daily' ? { mode, times: [...new Set(times)].sort() } : { mode }
  const error = validate(input)
  const done = { onSuccess: () => { toast.push(t('scheduler.saved')); onDone() }, onError: (e: unknown) => toast.push((e as Error).message, 'error') }
  const submit = () => (names && names.length > 1 ? bulk.mutate({ names, ...input }, done) : save.mutate({ name, ...input }, done))
  return (
    <div className="space-y-3 text-sm">
      <div className="flex flex-wrap gap-4">
        {(['default', 'interval', 'daily'] as const).map(m => (
          <label key={m} className="flex items-center gap-1.5"><input type="radio" name={`mode-${name}`} checked={mode === m} onChange={() => setMode(m)} />
            {m === 'default' ? t('scheduler.modeDefault', { v: fmtInterval(defaultSeconds, t) }) : t(`scheduler.mode_${m}`)}</label>))}
      </div>
      {mode === 'interval' && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[var(--muted)]">{t('scheduler.every')}</span>
          {UNITS.map(([k]) => (
            <label key={k} className="flex items-center gap-1">
              <input type="number" min={0} aria-label={t(`wizard.schedule.unit.${k}`)} className={`${inputCls} w-16`} value={parts[k]} onChange={e => setParts(p => ({ ...p, [k]: Math.max(0, Number(e.target.value) || 0) }))} />
              {t(`wizard.schedule.unit.${k}`)}
            </label>))}
        </div>)}
      {mode === 'daily' && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[var(--muted)]">{t('scheduler.dailyAt')}</span>
          {times.map((x, i) => (
            <span key={i} className="flex items-center gap-1">
              <input type="time" aria-label={t('scheduler.timeN', { n: i + 1 })} className={inputCls} value={x} onChange={e => setTimes(ts => ts.map((v, k) => (k === i ? e.target.value : v)))} />
              {times.length > 1 && <button type="button" aria-label={t('scheduler.removeTime')} className="text-[var(--muted)] hover:text-[var(--bad)]" onClick={() => setTimes(ts => ts.filter((_, k) => k !== i))}><X size={14} /></button>}
            </span>))}
          {times.length < 24 && <Button size="sm" variant="ghost" onClick={() => setTimes(ts => [...ts, '12:00'])}><Plus size={14} />{t('scheduler.addTime')}</Button>}
        </div>)}
      {error && <p role="alert" className="text-xs text-[var(--bad)]">{t(`scheduler.err.${error}`)}</p>}
      <div className="flex gap-2">
        <Button size="sm" disabled={!!error} loading={save.isPending || bulk.isPending} onClick={submit}>{t('common.save')}</Button>
        <Button size="sm" variant="ghost" onClick={onDone}>{t('common.cancel')}</Button>
      </div>
    </div>
  )
}
