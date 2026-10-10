import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { vi } from 'vitest'
import { useSetScheduled } from './hooks'

// ticking an item must show at once; the save happens in the background and is undone if it fails
function setup(response: () => Promise<Response>) {
  const qc = new QueryClient()
  qc.setQueryData(['items'], [{ name: 'Gold Ore', scheduled: false }, { name: 'Iron Ore', scheduled: false }])
  vi.stubGlobal('fetch', vi.fn(response))
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  const spy = vi.spyOn(qc, 'invalidateQueries')
  return { qc, spy, hook: renderHook(() => useSetScheduled(), { wrapper }) }
}
afterEach(() => vi.unstubAllGlobals())

it('updates the list immediately and does not reload every item', async () => {
  let release!: () => void
  const pending = new Promise<void>(r => { release = r })
  const { qc, spy, hook } = setup(async () => { await pending; return new Response(JSON.stringify({ ok: true, data: { changed: 1 } }), { headers: { 'Content-Type': 'application/json' } }) })
  act(() => hook.result.current.mutate({ names: ['Gold Ore'], scheduled: true }))
  await waitFor(() => expect((qc.getQueryData(['items']) as { scheduled: boolean }[])[0].scheduled).toBe(true))   // before the server answers
  release()
  await waitFor(() => expect(hook.result.current.isSuccess).toBe(true))
  expect(spy.mock.calls.map(c => (c[0] as { queryKey: string[] }).queryKey[0])).not.toContain('items')
})

it('puts the tick back when saving fails', async () => {
  const { qc, hook } = setup(async () => new Response(JSON.stringify({ ok: false, data: null, error: 'boom' }), { status: 500, headers: { 'Content-Type': 'application/json' } }))
  act(() => hook.result.current.mutate({ names: ['Gold Ore'], scheduled: true }))
  await waitFor(() => expect(hook.result.current.isError).toBe(true))
  expect((qc.getQueryData(['items']) as { scheduled: boolean }[])[0].scheduled).toBe(false)
})
