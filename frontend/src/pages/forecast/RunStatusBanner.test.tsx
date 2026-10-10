import { render, screen } from '@testing-library/react'
import '../../i18n'
import { RunStatusBanner } from './RunStatusBanner'
import type { ForecastRun } from '../../api/types'

const base = { selected: { name: 'naive', members: [] }, backtest: { origins: 12 }, data_age_hours: 2 }

it('RunStatusBanner shows not-better state with the reason', () => {
  render(<RunStatusBanner run={{ ...base, beats_baseline: false, skill_vs_naive: 0.01, baseline_reason: 'best model (ridge) was only 1.0% better' } as unknown as ForecastRun} />)
  expect(screen.getByText('没有找到比基准明显更准的方法，下面显示的是基准预测')).toBeInTheDocument()
  expect(screen.getByText(/only 1.0% better/)).toBeInTheDocument()
  expect(screen.getByText(/基于 12 次回测/)).toBeInTheDocument()
  expect(screen.getByText('沿用上一小时')).toBeInTheDocument()
})

it('warns when the data is stale', () => {
  render(<RunStatusBanner run={{ ...base, beats_baseline: true, skill_vs_naive: 0.2, data_age_hours: 72 } as unknown as ForecastRun} />)
  expect(screen.getByText(/3 天前/)).toBeInTheDocument()
})
