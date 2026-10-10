import type { ReactNode } from 'react'
export function Field({ label, hint, error, children }: { label: string; hint?: string; error?: string | null; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-[var(--muted)]">{label}</span>
      {children}
      {hint && !error && <span className="text-xs text-[var(--muted)]">{hint}</span>}
      {error && <span className="text-xs text-[var(--bad)]" role="alert">{error}</span>}
    </label>
  )
}
export const inputCls = 'border border-[var(--line)] rounded-md px-2.5 py-1.5 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30'
