import { useSearchParams } from 'react-router-dom'
import { useCallback } from 'react'

export function useUrlState(key: string, fallback = ''): [string, (v: string) => void] {
  const [params, setParams] = useSearchParams()
  const value = params.get(key) ?? fallback
  const set = useCallback((v: string) => {
    setParams(prev => { const next = new URLSearchParams(prev); if (v) next.set(key, v); else next.delete(key); return next }, { replace: true })
  }, [key, setParams])
  return [value, set]
}

/** Change several URL parameters in one navigation (separate setters would each start from the old URL). */
export function useUrlPatch(): (patch: Record<string, string>) => void {
  const [, setParams] = useSearchParams()
  return useCallback((patch: Record<string, string>) => {
    setParams(prev => {
      const next = new URLSearchParams(prev)
      for (const [k, v] of Object.entries(patch)) { if (v) next.set(k, v); else next.delete(k) }
      return next
    }, { replace: true })
  }, [setParams])
}
