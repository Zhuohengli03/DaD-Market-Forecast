/** DarkerDB charges one credit per request; the free plan has about 10,000 a month. */
export const MONTH_SECONDS = 30 * 86400

/** Each scheduled collection costs at least one request per item. */
export function monthlyRequests(intervalSeconds: number, items: number): number {
  return items * Math.ceil(MONTH_SECONDS / Math.max(intervalSeconds, 1))
}

/** Runs a month for one scheduler task as the backend describes it. */
export function taskRunsPerMonth(scheduleType: string, scheduleValue: string): number {
  if (scheduleType === 'seconds') return Math.ceil(MONTH_SECONDS / Math.max(Number(scheduleValue) || 1, 1))
  return scheduleValue.split('、').filter(Boolean).length * 30
}

export function fmtCount(n: number | null | undefined): string {
  return n === null || n === undefined ? '–' : n.toLocaleString('en-US')
}
