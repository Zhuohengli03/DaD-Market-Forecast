import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { useAnnouncements, useItemAnnouncements } from '../../api/hooks'
import { Card } from '../../components/ui/Card'
import { fmtDate } from '../../lib/format'
import { patchLabel } from '../../lib/labels'

/** The item's name in a patch line, highlighted (any case, plural allowed). */
function Highlight({ text, name }: { text: string; name: string }) {
  const words = name.split(/\s+/).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+')
  const parts = text.split(new RegExp(`(${words}(?:e?s)?)`, 'i'))
  return <>{parts.map((p, i) => (i % 2 ? <mark key={i} className="bg-yellow-100 rounded px-0.5">{p}</mark> : <Fragment key={i}>{p}</Fragment>))}</>
}

const change = (x: number | null | undefined) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${Math.round(x * 100)}%`)

/** Steam patches that name this item, newest first; click one to show it on the price chart. */
export function RelatedUpdates({ archetype, item, onPick }: { archetype: string; item: string; onPick: (ts: string) => void }) {
  const { t } = useTranslation()
  const q = useItemAnnouncements(archetype, item)
  const all = useAnnouncements()
  const found = q.data?.mentions ?? []
  return (
    <Card title={t('item.updates')}>
      {all.data?.error && <p className="text-xs text-[var(--warn)] mb-2">{all.data.error}</p>}
      {q.isLoading ? <p className="text-sm">{t('common.loading')}</p> : !found.length ? <p className="text-sm text-[var(--muted)]">{t('item.noUpdates')}</p> :
        <ul className="divide-y divide-[var(--line)]">{found.map(m => (
          <li key={m.id} className="py-2 text-sm flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="tabular-nums text-[var(--muted)]">{fmtDate(m.date)}</span>
            <button type="button" className="font-medium hover:underline text-left" onClick={() => onPick(m.date)}>{patchLabel(m.title, t)}</button>
            <span className={`tabular-nums text-xs ${(m.change3d ?? 0) > 0 ? 'text-[var(--good)]' : (m.change3d ?? 0) < 0 ? 'text-[var(--bad)]' : 'text-[var(--muted)]'}`}
              title={t('item.change3dHint')}>{change(m.change3d)}</span>
            <a href={m.url} target="_blank" rel="noreferrer" className="text-xs text-[var(--accent)] hover:underline">Steam</a>
            <p className="basis-full text-[var(--muted)]"><span className="mr-1 text-xs">{t('common.original')}：</span><span lang="en"><Highlight text={m.snippet} name={q.data?.name ?? ''} /></span></p>
          </li>))}</ul>}
    </Card>
  )
}
