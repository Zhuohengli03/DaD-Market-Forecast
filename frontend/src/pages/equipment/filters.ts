export interface FilterRow { key: string; min: string; max: string }
export interface StatLabel { label: string; percent: boolean }

const num = (s: string) => (s.trim() === '' || Number.isNaN(Number(s)) ? null : Number(s))

/** UI filter rows -> /api/equipment/query body. A stat key looks like `secondary_armor_penetration`. */
export function buildQuery(archetype: string, rarity: string, rows: FilterRow[], soldOnly: boolean, maxPages: number) {
  const filters = rows
    .filter(r => r.key && (num(r.min) !== null || num(r.max) !== null))
    .map(r => {
      const i = r.key.indexOf('_')
      return { kind: r.key.slice(0, i) as 'primary' | 'secondary', name: r.key.slice(i + 1), min: num(r.min), max: num(r.max) }
    })
  return { archetype, rarity: rarity || null, sold_only: soldOnly, max_pages: maxPages, filters }
}

export function formatStat(key: string, value: number, labels: Record<string, StatLabel>): string {
  const l = labels[key]
  if (!l) return `${key.slice(key.indexOf('_') + 1).replace(/_/g, ' ')} ${value}`
  return `${l.label} ${value}${l.percent ? '%' : ''}`
}
