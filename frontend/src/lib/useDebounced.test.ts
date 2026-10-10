import { act, renderHook } from '@testing-library/react'
import { vi } from 'vitest'
import { useDebounced } from './useDebounced'

it('only publishes the value after the user stops typing', () => {
  vi.useFakeTimers()
  const { result, rerender } = renderHook(({ v }) => useDebounced(v, 300), { initialProps: { v: 'a' } })
  rerender({ v: 'ab' }); rerender({ v: 'abc' })
  expect(result.current).toBe('a')
  act(() => { vi.advanceTimersByTime(300) })
  expect(result.current).toBe('abc')
  vi.useRealTimers()
})
