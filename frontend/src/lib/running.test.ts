import { isIdle, summarizeRunning } from './running'
import { monthlyRequests, taskRunsPerMonth } from './quota'

it('counts collected items one by one and other work by area', () => {
  const s = summarizeRunning({ collecting: { A: 'collecting', B: 'waiting', C: 'waiting' }, jobs_running: [{ kind: 'forecast' }, { kind: 'catalog' }, { kind: 'collect' }] } as never, 1)
  expect(s).toEqual({ collecting: 1, waiting: 2, forecast: 1, catalog: 1, init: 0, equipment: 1 })
  expect(isIdle(summarizeRunning(undefined, 0))).toBe(true)
})

it('estimates monthly requests for intervals and fixed times', () => {
  expect(monthlyRequests(6 * 3600, 7)).toBe(840)
  expect(taskRunsPerMonth('seconds', '3600')).toBe(720)
  expect(taskRunsPerMonth('daily_times', '08:00、20:00')).toBe(60)
})
