import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'

export function Notice({ children, to, action }: { children: ReactNode; to?: string; action?: string }) {
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm flex items-center justify-between gap-4">
      <span>{children}</span>
      {to && action && <Link to={to} className="shrink-0 font-medium text-[var(--accent)] underline">{action}</Link>}
    </div>
  )
}
