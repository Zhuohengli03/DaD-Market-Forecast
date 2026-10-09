import { render, screen } from '@testing-library/react'
import '../../i18n'
import { RunStatusBanner } from './RunStatusBanner'
import type { ForecastRun } from '../../api/types'

it('RunStatusBanner shows not-better state', () => {
  render(<RunStatusBanner run={{ beats_baseline: false, skill_vs_naive: -0.05, selected: { name: 'naive', members: [] } } as unknown as ForecastRun} />)
  expect(screen.getByText('未优于基线，显示基线预测')).toBeInTheDocument()
  expect(screen.getByText('-5.0%')).toBeInTheDocument()
})
