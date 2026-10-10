import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import { JobList } from './JobList'

vi.mock('../api/hooks')

it('filters jobs by kind and shows status', () => {
  vi.mocked(hooks.useJobs).mockReturnValue({ data: [
    { id: 'a', kind: 'collect', label: 'collect Gold Ore', status: 'succeeded', created_at: '2026-01-01T00:00:00Z', finished_at: null, error: null, result_ref: null, meta: {}, progress: { stage: 'done', done: null, total: null, detail: { new: 7 } } },
    { id: 'b', kind: 'forecast', label: 'fc', status: 'running', created_at: '2026-01-01T00:00:00Z', finished_at: null, error: null, result_ref: null, meta: {}, progress: null },
  ] } as never)
  render(<JobList kinds={['collect']} selected={null} onSelect={() => {}} />)
  expect(screen.getByText('collect Gold Ore')).toBeInTheDocument()
  expect(screen.queryByText('fc')).not.toBeInTheDocument()
  expect(screen.getByText('已完成')).toBeInTheDocument()
  expect(screen.getByText(/新增 7 条挂单/)).toBeInTheDocument()
})
