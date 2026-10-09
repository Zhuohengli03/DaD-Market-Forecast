import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import { JobList } from './JobList'

vi.mock('../api/hooks')

it('filters jobs by kind and shows status', () => {
  vi.mocked(hooks.useJobs).mockReturnValue({ data: [
    { id: 'a', kind: 'collect', label: 'collect Gold Ore', status: 'succeeded', created_at: '2026-01-01T00:00:00Z', finished_at: null, error: null, result_ref: null, meta: {}, log_tail: [] },
    { id: 'b', kind: 'forecast', label: 'fc', status: 'running', created_at: '2026-01-01T00:00:00Z', finished_at: null, error: null, result_ref: null, meta: {}, log_tail: [] },
  ] } as never)
  render(<JobList kinds={['collect']} selected={null} onSelect={() => {}} />)
  expect(screen.getByText('collect Gold Ore')).toBeInTheDocument()
  expect(screen.queryByText('fc')).not.toBeInTheDocument()
  expect(screen.getByText('succeeded')).toBeInTheDocument()
})
