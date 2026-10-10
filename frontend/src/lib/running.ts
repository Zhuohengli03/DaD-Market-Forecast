import type { StatusResponse } from '../api/types'

export interface RunningSummary { collecting: number; waiting: number; forecast: number; catalog: number; init: number; equipment: number; report: number }

/** Work in progress grouped by area; data collection counts every item on its own. */
export function summarizeRunning(status: StatusResponse | undefined, equipmentQueries: number): RunningSummary {
  const states = Object.values(status?.collecting ?? {})
  const jobs = status?.jobs_running ?? []
  const count = (kind: string) => jobs.filter(j => j.kind === kind).length
  return {
    collecting: states.filter(s => s === 'collecting').length, waiting: states.filter(s => s === 'waiting').length,
    forecast: count('forecast'), catalog: count('catalog'), init: count('init'), equipment: equipmentQueries, report: count('report'),
  }
}

export const isIdle = (s: RunningSummary) => Object.values(s).every(n => n === 0)
