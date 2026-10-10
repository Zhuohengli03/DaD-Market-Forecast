export type Meta = Record<string, unknown>
export interface Envelope<T> { ok: boolean; data: T; error: string | null; meta: Meta | null }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status; this.name = 'ApiError' }
}

type Params = Record<string, string | number | boolean | undefined | null>

function qs(params?: Params): string {
  if (!params) return ''
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  if (!entries.length) return ''
  return '?' + entries.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&')
}

function encodePath(path: string): string {
  return path.split('/').map((seg, i) => (i === 0 ? seg : encodeURIComponent(seg))).join('/')
}

async function request<T>(method: string, path: string, body?: unknown, params?: Params): Promise<{ data: T; meta: Meta | null }> {
  let res: Response
  try {
    res = await fetch('/api' + encodePath(path) + qs(params), {
      method,
      // the backend refuses writes without this header, which other websites cannot send (CSRF guard)
      headers: { 'X-Darker-Market': '1', ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (e) {
    throw new ApiError(0, `network error: ${(e as Error).message}`)
  }
  let env: Envelope<T> | null = null
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) {
    try { env = (await res.json()) as Envelope<T> } catch { env = null }
  }
  if (!env) {
    const text = (await res.text().catch(() => '')).slice(0, 200)
    throw new ApiError(res.status, `HTTP ${res.status}: ${text || 'non-JSON response'}`)
  }
  if (!res.ok || !env.ok) throw new ApiError(res.status, env.error || `HTTP ${res.status}`)
  return { data: env.data, meta: env.meta }
}

export const api = {
  get: <T>(path: string, params?: Params) => request<T>('GET', path, undefined, params),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  put: <T>(path: string, body: unknown) => request<T>('PUT', path, body),
  del: <T>(path: string) => request<T>('DELETE', path),
}
