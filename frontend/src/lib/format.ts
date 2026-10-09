export function fmtNum(n: number | null | undefined, digits = 2): string {
  return n === null || n === undefined || Number.isNaN(n) ? '–' : n.toFixed(digits)
}
export function fmtPct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '–'
  const v = (x * 100).toFixed(digits)
  return (x >= 0 ? '+' : '') + v + '%'
}
export function fmtTs(iso: string | null | undefined, opts: { withDate?: boolean } = { withDate: true }): string {
  if (!iso) return '–'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return opts.withDate === false ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}
export function fmtRelative(iso: string | null | undefined): string {
  if (!iso) return '–'
  const ms = Date.now() - new Date(iso).getTime()
  const h = Math.round(ms / 3.6e6)
  if (Math.abs(h) < 1) return `${Math.round(ms / 6e4)} min`
  if (Math.abs(h) < 48) return `${h} h`
  return `${Math.round(h / 24)} d`
}
export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return ''
  const keys = Object.keys(rows[0])
  return [keys.join(','), ...rows.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))].join('\n')
}
export function download(filename: string, text: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }))
  a.download = filename; a.click(); URL.revokeObjectURL(a.href)
}
