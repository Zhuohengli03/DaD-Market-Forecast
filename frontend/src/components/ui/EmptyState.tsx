export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return <div className="text-center py-10 text-[var(--muted)]"><p className="font-medium">{title}</p>{hint && <p className="text-xs mt-1">{hint}</p>}</div>
}
