import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import type { HoldoutRun } from '../../api/types'
import { HoldoutResult, holdoutVerdict } from './HoldoutResult'

vi.mock('echarts-for-react', () => ({ default: () => null }))   // no canvas in the test DOM

const RUN: HoldoutRun = {
  run_id: 'h1', created_at: '2026-10-10T00:00:00Z', item: 'Gold Ore', listings: 5000,
  request: { item: 'Gold Ore', season_id: 's10', train_frac: 0.7, horizon_hours: 24, models: [], coverage: 0.9 },
  season: { id: 's10', name: 'Season 10', start: '2026-07-17T00:00:00Z', end: '2026-10-09T03:00:00Z' },
  split_ts: '2026-09-20T14:00:00Z', train_hours: 1200, test_hours: 48, horizon: 24, nominal_coverage: 0.9,
  selected: { name: 'ridge', members: [{ name: 'ridge', weight: 1 }] }, train_skill: 0.12,
  metrics: { mae: 8, naive_mae: 10, skill: 0.2, mape: 0.05, mdape: 0.031, direction_acc: 0.7, coverage: 0.85, scored_hours: 48 },
  windows: [
    { origin: '2026-09-20T14:00:00Z', mae: 7, naive_mae: 10, skill: 0.3, mape: 0.04, direction_acc: 0.7, coverage: 0.9, scored_hours: 24 },
    { origin: '2026-09-21T14:00:00Z', mae: 9, naive_mae: 10, skill: 0.1, mape: 0.06, direction_acc: 0.7, coverage: 0.8, scored_hours: 24 },
  ],
  points: [{ ts: '2026-09-20T14:00:00Z', actual: 100, pred: 101, naive: 99, lower: 90, upper: 110 }],
  train_tail: [{ ts: '2026-09-20T13:00:00Z', actual: 99 }],
  warnings: [],
}

it('says in one line whether the forecaster beat the baseline on the unseen part', () => {
  const t = (k: string, o?: Record<string, unknown>) => `${k}:${JSON.stringify(o ?? {})}`
  expect(holdoutVerdict(RUN, t as never)).toContain('holdout.verdictBetter')
  expect(holdoutVerdict({ ...RUN, metrics: { ...RUN.metrics, skill: -0.05 } }, t as never)).toContain('holdout.verdictWorse')
})

it('shows the scores and lists each day in New York time, or UTC on request', () => {
  render(<HoldoutResult run={RUN} />)
  expect(screen.getByText('+20.0%')).toBeInTheDocument()            // better than the baseline
  expect(screen.getByText('85%')).toBeInTheDocument()               // interval hit rate
  expect(screen.getByText('3.1%')).toBeInTheDocument()              // typical error
  expect(screen.getByText('09-20 10:00')).toBeInTheDocument()       // 14:00 UTC is 10:00 in New York
  fireEvent.click(screen.getByRole('button', { name: 'UTC' }))
  expect(screen.getByText('09-20 14:00')).toBeInTheDocument()
})

it('translates warnings stored by older versions in the holdout result', () => {
  render(<HoldoutResult run={{ ...RUN, warnings: ['4 forward-filled hour-steps excluded from backtest scoring'] }} />)
  expect(screen.getByText('4 个前向填充的时间点没有计入回测评分')).toBeInTheDocument()
})
