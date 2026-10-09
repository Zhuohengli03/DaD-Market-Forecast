import { useEffect, useRef, useState } from 'react'
import type { JobStatus } from './types'

const TERMINAL: JobStatus[] = ['succeeded', 'failed']

export function useJobEvents(jobId: string | null) {
  const [lines, setLines] = useState<string[]>([])
  const [status, setStatus] = useState<JobStatus | null>(null)
  const [disconnected, setDisconnected] = useState(false)
  const statusRef = useRef<JobStatus | null>(null)
  useEffect(() => {
    setLines([]); setStatus(null); setDisconnected(false); statusRef.current = null
    if (!jobId) return
    const es = new EventSource(`/api/jobs/${jobId}/events`)
    es.addEventListener('log', (e: MessageEvent | { data: string }) => setLines(l => [...l, (e as { data: string }).data]))
    es.addEventListener('status', (e: MessageEvent | { data: string }) => {
      const s = (e as { data: string }).data as JobStatus
      statusRef.current = s; setStatus(s)
      if (TERMINAL.includes(s)) es.close()
    })
    es.onerror = () => { if (!statusRef.current || !TERMINAL.includes(statusRef.current)) setDisconnected(true); es.close() }
    return () => es.close()
  }, [jobId])
  return { lines, status, disconnected }
}
