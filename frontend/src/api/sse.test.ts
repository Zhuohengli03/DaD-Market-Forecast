import { renderHook, act } from '@testing-library/react'
import { useJobEvents } from './sse'

class FakeES {
  static last: FakeES
  listeners: Record<string, ((e: { data: string }) => void)[]> = {}
  onerror: (() => void) | null = null
  closed = false
  constructor(public url: string) { FakeES.last = this }
  addEventListener(type: string, cb: (e: { data: string }) => void) { (this.listeners[type] ||= []).push(cb) }
  emit(type: string, data: string) { (this.listeners[type] || []).forEach(cb => cb({ data })) }
  close() { this.closed = true }
}

beforeEach(() => { (globalThis as unknown as { EventSource: unknown }).EventSource = FakeES })

it('collects log lines and closes on terminal status', () => {
  const { result } = renderHook(() => useJobEvents('abc'))
  expect(FakeES.last.url).toBe('/api/jobs/abc/events')
  act(() => { FakeES.last.emit('log', 'hello'); FakeES.last.emit('status', 'running') })
  expect(result.current.lines).toEqual(['hello']); expect(result.current.status).toBe('running')
  act(() => FakeES.last.emit('status', 'succeeded'))
  expect(result.current.status).toBe('succeeded'); expect(FakeES.last.closed).toBe(true)
})

it('useJobEvents reports disconnect', () => {
  const { result } = renderHook(() => useJobEvents('abc'))
  act(() => { FakeES.last.emit('status', 'running'); FakeES.last.onerror?.() })
  expect(result.current.disconnected).toBe(true)
})
