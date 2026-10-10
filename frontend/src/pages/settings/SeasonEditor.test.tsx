import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { ApiError } from '../../api/client'
import { SeasonEditor } from './SeasonEditor'

vi.mock('../../api/hooks')

it('SeasonEditor shows server validation error', async () => {
  const add = { mutate: vi.fn((_: unknown, opts: { onError: (e: Error) => void }) => opts.onError(new ApiError(422, 'range overlaps season Season 2'))), isPending: false }
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [{ id: 's2', name: 'Season 2', start: '2026-01-20T00:00:00+00:00', end: null, note: '' }] } as never)
  vi.mocked(hooks.useSeasonMutations).mockReturnValue({ add, update: { mutate: vi.fn() }, remove: { mutate: vi.fn() }, importSeasons: { mutate: vi.fn(), isPending: false } } as never)
  render(<QueryClientProvider client={new QueryClient()}><SeasonEditor /></QueryClientProvider>)
  fireEvent.change(screen.getByLabelText('名称'), { target: { value: 'X' } })
  fireEvent.change(screen.getByLabelText('开始 (UTC)'), { target: { value: '2026-02-01T00:00' } })
  fireEvent.click(screen.getByText('添加'))
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('range overlaps season Season 2'))
  expect(screen.getAllByRole('row')).toHaveLength(2) // header + 1 season
})

it('imports seasons from DarkerDB after confirmation', () => {
  const importSeasons = { mutate: vi.fn(), isPending: false }
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [] } as never)
  vi.mocked(hooks.useSeasonMutations).mockReturnValue({ add: { mutate: vi.fn() }, update: { mutate: vi.fn() }, remove: { mutate: vi.fn() }, importSeasons } as never)
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  render(<QueryClientProvider client={new QueryClient()}><SeasonEditor /></QueryClientProvider>)
  fireEvent.click(screen.getByRole('button', { name: '从 DarkerDB 导入赛季' }))
  expect(importSeasons.mutate.mock.calls[0][0]).toEqual({ replace: true })
})
