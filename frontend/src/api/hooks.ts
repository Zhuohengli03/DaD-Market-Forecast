import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { Actual, ForecastInput, ForecastRun, ItemInfo, ItemsConfig, Job, Listing, Resolution, RunSummary, SchedulerStatus, Season, SeasonInput, SeriesResponse, StatusResponse } from './types'

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
export const useJob = (id: string | null) => useQuery({ queryKey: ['job', id], enabled: !!id, queryFn: () => api.get<Job>(`/jobs/${id}`).then(r => r.data) })
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
  }
}
export function useDeleteRun() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (id: string) => api.del(`/forecast/runs/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ['runs'] }) })
}
