import { useEffect, useRef, useState } from 'react'
import type { JobProgress, JobStatus } from './types'

const TERMINAL: JobStatus[] = ['succeeded', 'failed']

/** Live job progress over Server-Sent Events (`progress` and `status` events). */
export function useJobEvents(jobId: string | null) {
  const [progress, setProgress] = useState<JobProgress | null>(null)
  const [status, setStatus] = useState<JobStatus | null>(null)
  const [disconnected, setDisconnected] = useState(false)
  const statusRef = useRef<JobStatus | null>(null)
  useEffect(() => {
    setProgress(null); setStatus(null); setDisconnected(false); statusRef.current = null
    if (!jobId) return
    const es = new EventSource(`/api/jobs/${jobId}/events`)
    es.addEventListener('progress', (e: MessageEvent | { data: string }) => {
      try { setProgress(JSON.parse((e as { data: string }).data) as JobProgress) } catch { /* ignore malformed */ }
    })
    es.addEventListener('status', (e: MessageEvent | { data: string }) => {
      const s = (e as { data: string }).data as JobStatus
      statusRef.current = s; setStatus(s)
      if (TERMINAL.includes(s)) es.close()
    })
    es.onerror = () => { if (!statusRef.current || !TERMINAL.includes(statusRef.current)) setDisconnected(true); es.close() }
    return () => es.close()
  }, [jobId])
  return { progress, status, disconnected }
}
