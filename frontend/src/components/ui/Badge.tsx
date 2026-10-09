import type { ReactNode } from 'react'
export function Badge({ tone = 'muted', children }: { tone?: 'ok' | 'bad' | 'warn' | 'muted'; children: ReactNode }) {
  const t = { ok: 'bg-emerald-50 text-emerald-700 border-emerald-200', bad: 'bg-red-50 text-red-700 border-red-200', warn: 'bg-amber-50 text-amber-700 border-amber-200', muted: 'bg-gray-50 text-gray-600 border-gray-200' }
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-medium ${t[tone]}`}>{children}</span>
}
