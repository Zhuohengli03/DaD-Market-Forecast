import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { RunList } from './RunList'

vi.mock('../../api/hooks')

it('labels the chosen method in plain language', () => {
  vi.mocked(hooks.useRuns).mockReturnValue({ data: [{ run_id: 'r', created_at: '2026-10-09T00:00:00Z', source: 'csv', beats_baseline: true, skill_vs_naive: 0.2, item: 'Iron Ore', season_id: 's', season_name: 'S', horizon_hours: 24, selected: 'ensemble', warnings: 0 }] } as never)
  render(<RunList selected={null} onSelect={() => {}} />)
  expect(screen.getByText('多模型加权')).toBeInTheDocument()
})
