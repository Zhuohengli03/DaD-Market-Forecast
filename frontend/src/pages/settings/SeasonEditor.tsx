import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSeasonMutations, useSeasons } from '../../api/hooks'
import type { Season, SeasonInput } from '../../api/types'
import { Button } from '../../components/ui/Button'
import { Field, inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'
import { fmtTsUtc } from '../../lib/format'

const toInput = (iso: string | null) => (iso ? new Date(iso).toISOString().slice(0, 16) : '')
const fromInput = (v: string) => (v ? new Date(v + 'Z').toISOString() : null)
const empty: SeasonInput = { name: '', start: '', end: null, note: '' }

function SeasonForm({ initial, onSubmit, pending, error, submitLabel, onCancel }: { initial: SeasonInput; onSubmit: (s: SeasonInput) => void; pending: boolean; error: string | null; submitLabel: string; onCancel?: () => void }) {
  const { t } = useTranslation()
  const [s, setS] = useState(initial)
  return (
    <div className="flex flex-wrap items-end gap-3">
      <Field label={t('settings.seasonName')}><input className={inputCls} value={s.name} onChange={e => setS({ ...s, name: e.target.value })} /></Field>
      <Field label={t('settings.seasonStart')}><input type="datetime-local" className={inputCls} value={toInput(s.start || null)} onChange={e => setS({ ...s, start: fromInput(e.target.value) ?? '' })} /></Field>
      <Field label={t('settings.seasonEnd')}><input type="datetime-local" className={inputCls} value={toInput(s.end)} onChange={e => setS({ ...s, end: fromInput(e.target.value) })} /></Field>
      <Field label={t('settings.note')} error={error}><input className={inputCls} value={s.note} onChange={e => setS({ ...s, note: e.target.value })} /></Field>
      <Button loading={pending} disabled={!s.name || !s.start} onClick={() => onSubmit(s)}>{submitLabel}</Button>
      {onCancel && <Button variant="ghost" onClick={onCancel}>{t('common.cancel')}</Button>}
    </div>
  )
}

export function SeasonEditor() {
  const { t } = useTranslation()
  const seasons = useSeasons(); const m = useSeasonMutations(); const toast = useToast()
  const [editing, setEditing] = useState<Season | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fail = (e: unknown) => { setError((e as Error).message); toast.push((e as Error).message, 'error') }
  const okay = () => { setError(null); setEditing(null); toast.push(t('settings.saved')) }
  return (
    <div className="space-y-4">
      <table className="w-full text-sm"><thead className="text-left text-[var(--muted)]"><tr><th className="py-1">{t('settings.seasonName')}</th><th>{t('common.start')}</th><th>{t('common.end')}</th><th>{t('settings.note')}</th><th></th></tr></thead>
        <tbody>{(seasons.data ?? []).map(s => (
          <tr key={s.id} className="border-t border-[var(--line)]"><td className="py-1.5 font-medium">{s.name}</td><td>{fmtTsUtc(s.start)}</td><td>{s.end ? fmtTsUtc(s.end) : '—'}</td><td className="text-[var(--muted)]">{s.note}</td>
            <td className="text-right space-x-2"><Button size="sm" variant="ghost" onClick={() => { setEditing(s); setError(null) }}>✎</Button><Button size="sm" variant="danger" onClick={() => m.remove.mutate(s.id, { onSuccess: okay, onError: fail })}>{t('common.delete')}</Button></td></tr>))}</tbody></table>
      {editing
        ? <SeasonForm key={editing.id} initial={editing} pending={m.update.isPending} error={error} submitLabel={t('common.save')} onCancel={() => setEditing(null)} onSubmit={s => m.update.mutate({ id: editing.id, ...s }, { onSuccess: okay, onError: fail })} />
        : <SeasonForm key="new" initial={empty} pending={m.add.isPending} error={error} submitLabel={t('common.add')} onSubmit={s => m.add.mutate(s, { onSuccess: okay, onError: fail })} />}
    </div>
  )
}
