import type { ReactNode } from 'react'
export function Card({ title, actions, children, className = '' }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`bg-[var(--panel)] border border-[var(--line)] rounded-xl ${className}`}>
      {(title || actions) && <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--line)]"><h3 className="font-semibold text-sm">{title}</h3><div className="flex gap-2 items-center">{actions}</div></header>}
      <div className="p-4">{children}</div>
    </section>
  )
}
