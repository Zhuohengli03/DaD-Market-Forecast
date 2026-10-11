import { render, screen } from '@testing-library/react'
import '../../i18n'
import { BacktestTable } from './BacktestTable'
import type { ForecastRun } from '../../api/types'

const run = {
  selected: { name: 'ridge', members: [{ name: 'ridge', weight: 1 }] },
  backtest: { origins: 12, horizon: 24, coverage: 0.88, nominal_coverage: 0.9, failures: { arima: 'LinAlgError: x' }, per_horizon: {}, models: [
    { name: 'ridge', mae_h1: 0.5, mae_h24: 1.2, mae_mean: 0.9, rmse_mean: 1.1, mase_mean: 0.7, direction_acc: 0.6, mdape_mean: 0.083 },
    { name: 'naive', mae_h1: 0.6, mae_h24: 1.5, mae_mean: 1.3, rmse_mean: 1.6, mase_mean: 1, direction_acc: 0.5 },
    { name: 'seasonal_naive_24', mae_h1: 0.8, mae_h24: 1.4, mae_mean: 1.2, rmse_mean: 1.5, mase_mean: 0.92, direction_acc: 0.55 },
  ] },
} as unknown as ForecastRun

it('lists baselines first and marks the selected model', () => {
  render(<BacktestTable run={run} />)
  const rows = screen.getAllByRole('row').slice(1)
  expect(rows[0]).toHaveTextContent('沿用上一小时'); expect(rows[1]).toHaveTextContent('沿用昨天同一时刻'); expect(rows[2]).toHaveTextContent('线性回归')
  expect(rows[2].className).toMatch(/bg-blue-50/)
  expect(screen.getByText(/ARIMA/)).toHaveTextContent('LinAlgError')
  expect(screen.getByText('平均误差')).toHaveAttribute('title', '预测价和真实价平均差多少')
})

it('says how many backtests are really independent, and that the band comes from the same backtests', () => {
  render(<BacktestTable run={{ ...run, backtest: { ...run.backtest, independent_windows: 3, in_sample: true } } as ForecastRun} />)
  expect(screen.getByText(/其中互不重叠的 3 段/)).toBeInTheDocument()
  expect(screen.getByText(/同一批回测/)).toBeInTheDocument()
})

it('shows the typical error, which a rare spike cannot dominate', () => {
  render(<BacktestTable run={run} />)
  expect(screen.getByText('典型误差')).toBeInTheDocument()
  expect(screen.getByText('8.3%')).toBeInTheDocument()
})

it('says the method was picked on earlier backtests and judged on later ones', () => {
  render(<BacktestTable run={{ ...run, backtest: { ...run.backtest, in_sample: false, selection_origins: 20, evaluation_origins: 24 } } as ForecastRun} />)
  expect(screen.getByText(/前 20 次回测挑方法，后 24 次检验/)).toBeInTheDocument()
  expect(screen.queryByText(/同一批回测/)).not.toBeInTheDocument()
})
