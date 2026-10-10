import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { RunResult } from './RunResult'

vi.mock('../../api/hooks')

it('shows a not-found state and clears the selection when the run is gone', () => {
  vi.mocked(hooks.useRun).mockReturnValue({ isLoading: false, isError: true, data: undefined, error: new Error('run x not found') } as never)
  vi.mocked(hooks.useActuals).mockReturnValue({ data: undefined } as never)
  vi.mocked(hooks.useDeleteRun).mockReturnValue({ mutate: vi.fn() } as never)
  const onDeleted = vi.fn()
  render(<QueryClientProvider client={new QueryClient()}><RunResult runId="x" onDeleted={onDeleted} /></QueryClientProvider>)
  expect(screen.getByText('这条预测记录不存在或已被删除')).toBeInTheDocument()
  expect(onDeleted).toHaveBeenCalled()
})
