import type { JobProgress, JobStatus } from './types'
import { useJob } from './hooks'
import { useJobEvents } from './sse'

const TERMINAL: JobStatus[] = ['succeeded', 'failed']

/** Live job progress over SSE; if the stream drops or the job ends, the job endpoint fills in the truth. */
export function useJobStream(jobId: string | null) {
  const ev = useJobEvents(jobId)
  const evDone = ev.status !== null && TERMINAL.includes(ev.status)
  const job = useJob(jobId, { poll: ev.disconnected || evDone }).data
  const jobDone = !!job && TERMINAL.includes(job.status)
  return {
    progress: ((jobDone || ev.disconnected) && job ? job.progress : ev.progress) as JobProgress | null,
    status: (jobDone ? job.status : ev.status) as JobStatus | null,
    disconnected: ev.disconnected && !jobDone,
    error: job?.error ?? null,
    resultRef: jobDone ? job.result_ref : null,
  }
}
