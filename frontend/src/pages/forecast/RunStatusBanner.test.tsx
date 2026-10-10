import { render, screen } from '@testing-library/react'
import '../../i18n'
import { RunStatusBanner } from './RunStatusBanner'
import type { ForecastRun } from '../../api/types'

const base = { selected: { name: 'naive', members: [] }, backtest: { origins: 12 }, data_age_hours: 2 }

it('RunStatusBanner shows not-better state with the reason', () => {
  render(<RunStatusBanner run={{ ...base, beats_baseline: false, skill_vs_naive: 0.01, baseline_reason: 'best model (ridge) was only 1.0% better, within noise (needs > 2%)' } as unknown as ForecastRun} />)
  expect(screen.getByText('没有找到比基准明显更准的方法，下面显示的是基准预测')).toBeInTheDocument()
  expect(screen.getByText(/最好的方法（线性回归）只比基准准 1.0%/)).toBeInTheDocument()
  expect(screen.getByText(/基于 12 次回测/)).toBeInTheDocument()
  expect(screen.getByText('沿用上一小时')).toBeInTheDocument()
})

it('warns when the data is stale', () => {
  render(<RunStatusBanner run={{ ...base, beats_baseline: true, skill_vs_naive: 0.2, data_age_hours: 72 } as unknown as ForecastRun} />)
  expect(screen.getByText(/3 天前/)).toBeInTheDocument()
})

it('says how much lower the error is, without a confusing plus sign', () => {
  render(<RunStatusBanner run={{ ...base, beats_baseline: true, skill_vs_naive: 0.212 } as unknown as ForecastRun} />)
  expect(screen.getByText('误差比“沿用上一小时”低 21.2%')).toBeInTheDocument()
})

it('does not ask to collect fresh data for a season that has ended', () => {
  render(<RunStatusBanner run={{ ...base, beats_baseline: true, skill_vs_naive: 0.2, data_age_hours: 6000, season: { id: 's7', name: 'Season 7', start: '2025-01-01', end: '2025-03-01T00:00:00Z' } } as unknown as ForecastRun} />)
  expect(screen.queryByText(/请先采集新数据/)).not.toBeInTheDocument()
  expect(screen.getByText(/第 7 赛季（已结束）/)).toBeInTheDocument()
})
