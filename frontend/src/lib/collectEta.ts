import i18n from '../i18n'

const SECONDS_PER_PAGE = 1   // DarkerDB allows 60 requests a minute; one page is one request

export interface CollectEta { total: number | null; share: number | null; secondsLeft: number | null }

/** Progress of a collection from pages done, the expected pages (its credit estimate) and the pace so far. */
export function collectEta(pages: number, expected: number | null | undefined, startedAt: string, now = Date.now()): CollectEta {
  if (!expected || pages > expected) return { total: null, share: null, secondsLeft: null }
  const elapsed = Math.max(0, (now - Date.parse(startedAt)) / 1000)
  const perPage = pages >= 2 && elapsed > 0 ? elapsed / pages : SECONDS_PER_PAGE
  return { total: expected, share: pages / expected, secondsLeft: Math.round((expected - pages) * perPage) }
}

/** "不到 1 分钟" / "约 3 分钟" / "约 1 小时 7 分钟". */
export function fmtWait(seconds: number): string {
  if (seconds < 60) return i18n.t('wait.underMinute')
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return i18n.t('wait.minutes', { n: minutes })
  return i18n.t('wait.hours', { h: Math.floor(minutes / 60), m: minutes % 60 })
}
