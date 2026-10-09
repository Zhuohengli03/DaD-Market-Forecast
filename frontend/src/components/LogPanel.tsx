import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import type { JobStatus } from '../api/types'
import { Badge } from './ui/Badge'

export function LogPanel({ lines, status, disconnected }: { lines: string[]; status: JobStatus | null; disconnected: boolean }) {
  const { t } = useTranslation()
  const ref = useRef<HTMLPreElement>(null)
  useEffect(() => { ref.current?.scrollTo?.({ top: ref.current.scrollHeight }) }, [lines.length])
  const tone = status === 'succeeded' ? 'ok' : status === 'failed' ? 'bad' : 'warn'
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-sm"><span className="text-[var(--muted)]">{t('common.log')}</span>{status && <Badge tone={tone}>{status}</Badge>}{disconnected && <Badge tone="warn">{t('forecast.disconnected')}</Badge>}</div>
      <pre ref={ref} className="bg-gray-900 text-gray-100 text-xs rounded-lg p-3 h-56 overflow-auto font-mono whitespace-pre-wrap">{lines.join('\n') || '…'}</pre>
    </div>
  )
}
