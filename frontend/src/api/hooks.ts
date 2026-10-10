import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { Actual, DatabaseInput, CatalogFacets, CatalogMeta, CatalogType, EquipmentQueryBody, EquipmentResult, EquipmentStat, ForecastInput, JobStatus, SetupState, ForecastRun, ItemInfo, ItemsConfig, Job, Listing, Resolution, RunSummary, SchedulerStatus, Season, SeasonInput, SeriesResponse, StatusResponse } from './types'

export const useStatus = () => useQuery({ queryKey: ['status'], queryFn: () => api.get<StatusResponse>('/status').then(r => r.data), refetchInterval: 5000 })
export const useItems = () => useQuery({ queryKey: ['items'], queryFn: () => api.get<ItemInfo[]>('/items').then(r => r.data) })
export const useSeasons = () => useQuery({ queryKey: ['seasons'], queryFn: () => api.get<Season[]>('/seasons').then(r => r.data) })
export const useModels = () => useQuery({ queryKey: ['models'], queryFn: () => api.get<string[]>('/forecast/models').then(r => r.data) })

export interface SeriesParams { season_id?: string; start?: string; end?: string; resolution: Resolution }
export const useSeries = (item: string, p: SeriesParams, enabled = true) =>
  useQuery({ queryKey: ['series', item, p], enabled: enabled && !!item, queryFn: () => api.get<SeriesResponse>(`/series/${item}`, { ...p }).then(r => r.data) })

export interface ListingParams { season_id?: string; start?: string; end?: string; sold?: boolean; min_qty?: number; max_qty?: number; page: number; page_size: number }
export const useListings = (item: string, p: ListingParams) =>
  useQuery({ queryKey: ['listings', item, p], enabled: !!item, placeholderData: prev => prev, queryFn: () => api.get<Listing[]>(`/listings/${item}`, { ...p }).then(r => ({ rows: r.data, total: Number(r.meta?.total ?? 0) })) })

export const useRuns = () => useQuery({ queryKey: ['runs'], queryFn: () => api.get<RunSummary[]>('/forecast/runs').then(r => r.data) })
export const useRun = (id: string | null) => useQuery({ queryKey: ['run', id], enabled: !!id, queryFn: () => api.get<ForecastRun>(`/forecast/runs/${id}`).then(r => r.data) })
export const useActuals = (id: string | null, enabled: boolean) => useQuery({ queryKey: ['actuals', id], enabled: !!id && enabled, queryFn: () => api.get<Actual[]>(`/forecast/runs/${id}/actuals`).then(r => r.data) })
export const useJobs = () => useQuery({ queryKey: ['jobs'], queryFn: () => api.get<Job[]>('/jobs').then(r => r.data), refetchInterval: 3000 })
const TERMINAL: JobStatus[] = ['succeeded', 'failed']
export const useJob = (id: string | null, opts: { poll?: boolean } = {}) => useQuery({
  queryKey: ['job', id], enabled: !!id, queryFn: () => api.get<Job>(`/jobs/${id}`).then(r => r.data),
  refetchInterval: q => (opts.poll && !TERMINAL.includes(q.state.data?.status as JobStatus) ? 1500 : false),
})
export const useScheduler = () => useQuery({ queryKey: ['scheduler'], queryFn: () => api.get<SchedulerStatus>('/scheduler').then(r => r.data), refetchInterval: 5000 })
export const useItemsConfig = () => useQuery({ queryKey: ['itemsConfig'], queryFn: () => api.get<ItemsConfig>('/config/items').then(r => r.data) })

export function useStartForecast() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (body: ForecastInput) => api.post<{ job_id: string }>('/forecast', body).then(r => r.data.job_id), onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export function useStartCollect() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (item: string) => api.post<{ job_id: string }>(`/collect/${item}`).then(r => r.data.job_id), onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export function useSchedulerAction() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (action: 'start' | 'stop') => api.post<{ running: boolean }>(`/scheduler/${action}`).then(r => r.data.running), onSuccess: () => qc.invalidateQueries({ queryKey: ['scheduler'] }) })
}
export function useRunTask() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (script_path: string) => api.post<{ job_id: string }>('/scheduler/run', { script_path }).then(r => r.data.job_id), onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export function useSaveItemsConfig() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (cfg: ItemsConfig) => api.put<ItemsConfig>('/config/items', cfg).then(r => r.data), onSuccess: () => { qc.invalidateQueries({ queryKey: ['itemsConfig'] }); qc.invalidateQueries({ queryKey: ['items'] }) } })
}
export function useSeasonMutations() {
  const qc = useQueryClient()
  const done = () => qc.invalidateQueries({ queryKey: ['seasons'] })
  return {
    add: useMutation({ mutationFn: (s: SeasonInput) => api.post<Season>('/seasons', s).then(r => r.data), onSuccess: done }),
    update: useMutation({ mutationFn: ({ id, ...s }: SeasonInput & { id: string }) => api.put<Season>(`/seasons/${id}`, s).then(r => r.data), onSuccess: done }),
    remove: useMutation({ mutationFn: (id: string) => api.del(`/seasons/${id}`), onSuccess: done }),
    importSeasons: useMutation({ mutationFn: (body: { replace: boolean }) => api.post<{ added: number; skipped: number }>('/seasons/import', body).then(r => r.data), onSuccess: done }),
  }
}
export function useDeleteRun() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (id: string) => api.del(`/forecast/runs/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ['runs'] }) })
}

export const useSetup = () => useQuery({ queryKey: ['setup'], queryFn: () => api.get<SetupState>('/setup').then(r => r.data) })
export function useSaveApiKey() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (key: string) => api.put<SetupState>('/setup/api-key', { api_key: key }).then(r => r.data), onSuccess: d => { qc.setQueryData(['setup'], d) } })
}

export interface CatalogParams { q?: string; item_type?: string; has_stats?: boolean; page: number; page_size: number }
export const useCatalog = (p: CatalogParams) => useQuery({
  queryKey: ['catalog', p], placeholderData: prev => prev,
  queryFn: () => api.get<CatalogType[]>('/catalog', { ...p }).then(r => ({ rows: r.data, total: Number(r.meta?.total ?? 0), facets: (r.meta?.facets ?? { item_type: [] }) as CatalogFacets })),
})
export const useEquipmentStats = (archetype: string) => useQuery({
  queryKey: ['equipmentStats', archetype], enabled: !!archetype, staleTime: 5 * 60_000,
  queryFn: () => api.get<{ stats: EquipmentStat[]; rarities: string[] }>(`/equipment/${archetype}/stats`).then(r => r.data),
})
export const useEquipmentQuery = () => useMutation({ mutationFn: (body: EquipmentQueryBody) => api.post<EquipmentResult>('/equipment/query', body).then(r => r.data) })
export const useCatalogMeta = () => useQuery({ queryKey: ['catalogMeta'], queryFn: () => api.get<CatalogMeta>('/catalog/meta').then(r => r.data) })
export function useCatalogJob() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (kind: 'sync' | 'check') => api.post<{ job_id: string }>(kind === 'sync' ? '/catalog/sync' : '/catalog/check-trade', kind === 'check' ? { force: false } : undefined).then(r => r.data.job_id), onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export function useTrackItem() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (item_id: string) => api.post<{ name: string }>('/items/track', { item_id }).then(r => r.data), onSuccess: () => { ['items', 'itemsConfig', 'setup', 'scheduler'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) } })
}
export function useUntrackItem() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (name: string) => api.del(`/items/${name}`), onSuccess: () => { ['items', 'itemsConfig', 'setup', 'scheduler'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) } })
}

export function useSaveSchedule() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (b: { interval_hours: number; auto_collect: boolean }) => api.put<SetupState>('/setup/schedule', b).then(r => r.data), onSuccess: d => { qc.setQueryData(['setup'], d); qc.invalidateQueries({ queryKey: ['scheduler'] }) } })
}
export function useDatabaseSetup() {
  const qc = useQueryClient()
  const done = (d: SetupState) => { qc.setQueryData(['setup'], d); ['status', 'items'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) }
  return {
    save: useMutation({ mutationFn: (b: DatabaseInput) => api.put<SetupState>('/setup/database', b).then(r => r.data), onSuccess: done }),
    clear: useMutation({ mutationFn: () => api.del<SetupState>('/setup/database').then(r => r.data), onSuccess: done }),
  }
}
export function useInitialize() {
  return useMutation({ mutationFn: (b: { item_ids: string[]; max_pages: number }) => api.post<{ job_id: string }>('/setup/initialize', b).then(r => r.data.job_id) })
}
