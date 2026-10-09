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
