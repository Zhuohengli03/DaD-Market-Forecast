import { horizonLabels } from './HorizonErrorChart'

it('labels each point by how many hours ahead it ends, also for 4-hour and daily blocks', () => {
  const run = (resolution: string | undefined, n: number) => ({ backtest: { resolution, per_horizon: { naive: Array(n).fill(1) } } }) as never
  expect(horizonLabels(run(undefined, 3))).toEqual([1, 2, 3])
  expect(horizonLabels(run('4h', 6))).toEqual([4, 8, 12, 16, 20, 24])
  expect(horizonLabels(run('1d', 2))).toEqual([24, 48])
})
