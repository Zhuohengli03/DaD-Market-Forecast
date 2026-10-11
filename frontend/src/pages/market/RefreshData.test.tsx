import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import * as stream from '../../api/jobStream'
import { RefreshData } from './RefreshData'

vi.mock('../../api/hooks')
vi.mock('../../api/jobStream')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

it('follows a collection started elsewhere and shows pages, a bar and the time left', () => {
  const started = new Date(Date.now() - 60_000).toISOString()   // 60 pages in a minute so far
  vi.mocked(hooks.useStartCollect).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useCollectEstimate).mockReturnValue({ data: { requests: 180, known: true, since: null, until: null, opens: null } } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { collecting: { 'Gold Ore': 'collecting' } } } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [] } as never)
  vi.mocked(hooks.useJobs).mockReturnValue({ data: [{ id: 'j9', kind: 'collect', status: 'running', created_at: started, meta: { item: 'Gold Ore' }, cost: { estimate: 180, used: 60 } }] } as never)
  const useStream = vi.mocked(stream.useJobStream).mockReturnValue({ status: 'running', progress: { stage: 'collect', done: 60, total: 200, detail: { new: 3000 } }, disconnected: false, error: null, resultRef: null } as never)
  render(<QueryClientProvider client={new QueryClient()}><RefreshData item="Gold Ore" /></QueryClientProvider>)
  expect(useStream).toHaveBeenLastCalledWith('j9')
  expect(screen.getByText(/60 \/ 约 180 页/)).toBeInTheDocument()
  expect(screen.getByText(/约 2 分钟/)).toBeInTheDocument()
  expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '33')
})
