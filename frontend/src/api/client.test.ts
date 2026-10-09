import { describe, it, expect, vi, beforeEach } from 'vitest'
import { api, ApiError } from './client'

const mockFetch = vi.fn()
beforeEach(() => { mockFetch.mockReset(); globalThis.fetch = mockFetch as unknown as typeof fetch })

function jsonResponse(body: unknown, status = 200) {
  return { ok: status < 400, status, headers: { get: () => 'application/json' }, json: async () => body, text: async () => JSON.stringify(body) }
}

describe('api client', () => {
  it('unwraps the envelope and builds query strings', async () => {
    mockFetch.mockResolvedValue(jsonResponse({ ok: true, data: [1, 2], error: null, meta: { total: 2 } }))
    const r = await api.get<number[]>('/series/Iron Ore', { season_id: 's2', resolution: '1h', skip: undefined })
    expect(r.data).toEqual([1, 2]); expect(r.meta).toEqual({ total: 2 })
    expect(mockFetch.mock.calls[0][0]).toBe('/api/series/Iron%20Ore?season_id=s2&resolution=1h')
  })
  it('throws ApiError with the server message on ok=false', async () => {
    mockFetch.mockResolvedValue(jsonResponse({ ok: false, data: null, error: 'range overlaps season', meta: null }, 422))
    await expect(api.post('/seasons', { name: 'x' })).rejects.toMatchObject({ status: 422, message: 'range overlaps season' })
  })
  it('throws readable error on non-JSON response', async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 502, headers: { get: () => 'text/html' }, json: async () => { throw new Error('bad') }, text: async () => '<html>Bad Gateway</html>' })
    await expect(api.get('/status')).rejects.toBeInstanceOf(ApiError)
    await expect(api.get('/status')).rejects.toMatchObject({ status: 502 })
  })
  it('throws on network failure', async () => {
    mockFetch.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(api.get('/status')).rejects.toMatchObject({ status: 0 })
  })
})
