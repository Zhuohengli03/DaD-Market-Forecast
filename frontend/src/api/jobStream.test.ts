import { renderHook } from '@testing-library/react'
import { vi } from 'vitest'
import * as sse from './sse'
import * as hooks from './hooks'
import { useJobStream } from './jobStream'

vi.mock('./sse')
vi.mock('./hooks')

it('falls back to polling the job when the event stream drops', () => {
  vi.mocked(sse.useJobEvents).mockReturnValue({ lines: ['a'], status: 'running', disconnected: true })
  vi.mocked(hooks.useJob).mockReturnValue({ data: { status: 'succeeded', log_tail: ['a', 'b'], error: null, result_ref: 'r1' } } as never)
  const { result } = renderHook(() => useJobStream('j1'))
  expect(vi.mocked(hooks.useJob).mock.calls.at(-1)).toEqual(['j1', { poll: true }])
  expect(result.current).toMatchObject({ status: 'succeeded', lines: ['a', 'b'], resultRef: 'r1', disconnected: false })
})
