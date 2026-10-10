import { renderHook } from '@testing-library/react'
import { vi } from 'vitest'
import * as sse from './sse'
import * as hooks from './hooks'
import { useJobStream } from './jobStream'

vi.mock('./sse')
vi.mock('./hooks')

it('falls back to polling the job when the event stream drops', () => {
  vi.mocked(sse.useJobEvents).mockReturnValue({ progress: { stage: 'collect', done: 1, total: 200, detail: {} }, status: 'running', disconnected: true })
  vi.mocked(hooks.useJob).mockReturnValue({ data: { status: 'succeeded', progress: { stage: 'done', done: null, total: null, detail: { new: 5 } }, error: null, result_ref: 'r1' } } as never)
  const { result } = renderHook(() => useJobStream('j1'))
  expect(vi.mocked(hooks.useJob).mock.calls.at(-1)).toEqual(['j1', { poll: true }])
  expect(result.current).toMatchObject({ status: 'succeeded', progress: { stage: 'done', detail: { new: 5 } }, resultRef: 'r1', disconnected: false })
})
