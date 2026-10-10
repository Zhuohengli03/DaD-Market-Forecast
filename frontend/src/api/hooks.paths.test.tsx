import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { vi } from 'vitest'
import { useCollectEstimate, useSetItemSchedule } from './hooks'

// the API client encodes each path segment itself; hooks must pass raw names ("Gold Ore", not "Gold%20Ore")
function wrapper({ children }: { children: ReactNode }) { return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider> }

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ ok: true, data: { requests: 1, known: true }, error: null, meta: null }), { headers: { 'Content-Type': 'application/json' } })))
})
afterEach(() => vi.unstubAllGlobals())

it('encodes an item name with spaces exactly once', async () => {
  renderHook(() => useCollectEstimate('Ruby (Flawed) (common)'), { wrapper })
  await waitFor(() => expect(fetch).toHaveBeenCalled())
  expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/estimate/collect/Ruby%20(Flawed)%20(common)')
})

it('saves a per-item schedule for a name with spaces', async () => {
  const { result } = renderHook(() => useSetItemSchedule(), { wrapper })
  await result.current.mutateAsync({ name: 'Gold Ore', mode: 'interval', seconds: 3600 })
  expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/items/Gold%20Ore/schedule')
})
