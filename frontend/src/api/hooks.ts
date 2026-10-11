import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { Actual, RulesStatus, Announcement, ItemMention, SeasonCompareRow, NextHour, DbDelete, DbItem, DbRow, HoldoutRun, HoldoutSummary, DatabaseInput, CatalogFacets, CatalogMeta, CatalogType, EquipmentQueryBody, EquipmentResult, EquipmentStat, ForecastInput, JobStatus, SetupState, ForecastRun, ItemInfo, ItemsConfig, Job, Listing, Resolution, CollectAllEstimate, ItemsEstimate, LLMSettings, OverviewGroup, Report, ReportEstimate, ReportSummary, QuotaInfo, RunSummary, ScheduleEstimate, ScheduleInput, TypeHistory, SchedulerStatus, Season, SeasonInput, SeriesResponse, StatusResponse } from './types'

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
  // a season id collects only that season: from its first day (or the newest listing stored in it) to its end
  return useMutation({ mutationFn: (arg: string | { item: string; seasonId?: string }) => {
    const { item, seasonId } = typeof arg === 'string' ? { item: arg, seasonId: undefined } : arg
    return api.post<{ job_id: string }>(`/collect/${item}`, undefined, seasonId ? { season_id: seasonId } : undefined).then(r => r.data.job_id)
  }, onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
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
  const done = () => Promise.all(['seasons', 'series', 'listings', 'items', 'overview', 'estimate', 'catalog'].map(k => qc.invalidateQueries({ queryKey: [k] })))
  return {
    add: useMutation({ mutationFn: (s: SeasonInput) => api.post<Season>('/seasons', s).then(r => r.data), onSuccess: done }),
    update: useMutation({ mutationFn: ({ id, ...s }: SeasonInput & { id: string }) => api.put<Season>(`/seasons/${id}`, s).then(r => r.data), onSuccess: done }),
    remove: useMutation({ mutationFn: (id: string) => api.del(`/seasons/${id}`), onSuccess: done }),
    importSeasons: useMutation({ mutationFn: (body: { replace: boolean }) => api.post<{ added: number; skipped: number }>('/seasons/import', body).then(r => r.data), onSuccess: done }),
  }
}
// the local database page: look and delete, never edit
export const useDbItems = () => useQuery({ queryKey: ['db', 'items'], queryFn: () => api.get<DbItem[]>('/db/items').then(r => ({ items: r.data, source: String(r.meta?.source ?? '') })) })
export const useDbRows = (key: string, start: string, end: string, page: number) => useQuery({ queryKey: ['db', 'rows', key, start, end, page], enabled: !!key,
  queryFn: () => api.get<DbRow[]>('/db/rows', { key, start, end, page, page_size: 100 }).then(r => ({ rows: r.data, total: Number(r.meta?.total ?? 0) })) })
export function useDbDelete() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (body: DbDelete) => api.post<{ deleted: number }>('/db/delete', body).then(r => r.data.deleted),
    // fewer rows change every count, chart and estimate in the app
    onSuccess: () => { for (const k of ['db', 'items', 'overview', 'series', 'listings', 'estimate', 'status']) qc.invalidateQueries({ queryKey: [k] }) } })
}
/** Where next hour's sales will land, from local data (no DarkerDB credits). */
export const useNextHour = (item: string, seasonId?: string) => useQuery({ queryKey: ['series', 'nextHour', item, seasonId ?? null], enabled: !!item, retry: false,
  queryFn: () => api.get<NextHour>(`/next-hour/${item}`, seasonId ? { season_id: seasonId } : undefined).then(r => r.data) })
// game patch notes from Steam (no DarkerDB credits)
export const useAnnouncements = () => useQuery({ queryKey: ['announcements'], staleTime: 300_000,
  queryFn: () => api.get<Announcement[]>('/announcements').then(r => ({ items: r.data, updatedAt: (r.meta?.updated_at as string | null) ?? null, error: (r.meta?.error as string | null) ?? null })) })
export const useItemAnnouncements = (archetype: string, item?: string) => useQuery({ queryKey: ['announcements', 'item', archetype, item ?? null], enabled: !!archetype,
  queryFn: () => api.get<{ name: string; mentions: ItemMention[] }>(`/announcements/item/${archetype}`, item ? { item } : undefined).then(r => r.data) })
/** Each season's daily median, lined up by the day trading opened. */
export const useSeasonCompare = (item: string) => useQuery({ queryKey: ['series', 'seasons', item], enabled: !!item,
  queryFn: () => api.get<{ seasons: SeasonCompareRow[] }>(`/series/${item}/seasons`).then(r => r.data) })
export const useHoldouts = () => useQuery({ queryKey: ['holdouts'], queryFn: () => api.get<HoldoutSummary[]>('/forecast/holdouts').then(r => r.data) })
export const useHoldout = (id: string | null) => useQuery({ queryKey: ['holdouts', id], enabled: !!id, queryFn: () => api.get<HoldoutRun>(`/forecast/holdouts/${id}`).then(r => r.data) })
/** Season holdout: pick the method on the first part of an ended season, score it day by day on the rest. */
export function useStartHoldout() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (body: { item: string; season_id: string; train_frac?: number; resolution?: string }) => api.post<{ job_id: string }>('/forecast/holdout', body).then(r => r.data.job_id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export function useDeleteHoldout() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (id: string) => api.del(`/forecast/holdouts/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ['holdouts'] }) })
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
export const useEquipmentQuery = () => useMutation({ mutationKey: ['equipmentQuery'], mutationFn: (body: EquipmentQueryBody) => api.post<EquipmentResult>('/equipment/query', body).then(r => r.data) })
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
  return useMutation({ mutationFn: (b: { interval_seconds: number; auto_collect: boolean }) => api.put<SetupState>('/setup/schedule', b).then(r => r.data), onSuccess: d => { qc.setQueryData(['setup'], d); qc.invalidateQueries({ queryKey: ['scheduler'] }) } })
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

export function useSetItemSchedule() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: ({ name, ...b }: ScheduleInput & { name: string }) => api.put<unknown>(`/items/${name}/schedule`, b).then(r => r.data),
    onSuccess: () => { ['itemsConfig', 'items', 'scheduler'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) } })
}
export interface BulkTrackResult { added: number; skipped: { has_stats: number; already_tracked: number; not_in_catalog: number } }
export function useTrackBulk() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (item_ids: string[]) => api.post<BulkTrackResult>('/items/track-bulk', { item_ids }).then(r => r.data),
    onSuccess: () => { ['items', 'itemsConfig', 'setup', 'scheduler', 'status'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) } })
}
export function useCollectAll() {
  const qc = useQueryClient()
  // names: collect only these tracked items (e.g. the ones without data); omitted = all of them
  // seasonId: only that season's trading window; omitted = the current season
  return useMutation({ mutationFn: (arg?: string[] | { names?: string[]; seasonId?: string }) => {
    const { names, seasonId } = Array.isArray(arg) || arg === undefined ? { names: arg, seasonId: undefined } : arg
    const body = { ...(names ? { names } : {}), ...(seasonId ? { season_id: seasonId } : {}) }
    return api.post<{ job_id: string }>('/collect-all', Object.keys(body).length ? body : undefined).then(r => r.data.job_id)
  }, onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export const useQuota = () => useQuery({ queryKey: ['quota'], queryFn: () => api.get<QuotaInfo>('/quota').then(r => r.data), refetchInterval: 15000 })
export function useSaveQuota() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (monthly_quota: number) => api.put<SetupState>('/setup/quota', { monthly_quota }).then(r => r.data),
    onSuccess: d => { qc.setQueryData(['setup'], d); qc.invalidateQueries({ queryKey: ['quota'] }) } })
}
export function useTypeHistory() {
  const qc = useQueryClient()
  // refresh: ask DarkerDB again even when this season was already looked up (the count keeps growing)
  return useMutation({ mutationFn: (arg: string | { archetype: string; refresh?: boolean }) => {
    const { archetype, refresh } = typeof arg === 'string' ? { archetype: arg, refresh: false } : arg
    return api.post<TypeHistory>(`/catalog/types/${archetype}/history`, undefined, refresh ? { refresh: true } : undefined).then(r => r.data)
  },
    onSuccess: () => { ['catalog', 'quota'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) } })
}
// cost estimates (DarkerDB credits), always rounded up by the server
export const useCollectAllEstimate = (enabled = true, seasonId?: string) => useQuery({ queryKey: ['estimate', 'collectAll', seasonId ?? null], enabled, staleTime: 60_000,
  queryFn: () => api.get<CollectAllEstimate>('/estimate/collect-all', seasonId ? { season_id: seasonId } : undefined).then(r => r.data) })
export const useScheduleEstimate = (intervalSeconds?: number) => useQuery({ queryKey: ['estimate', 'schedule', intervalSeconds ?? null], staleTime: 60_000,
  queryFn: () => api.get<ScheduleEstimate>(`/estimate/schedule${intervalSeconds ? `?interval_seconds=${intervalSeconds}` : ''}`).then(r => r.data) })
export const useItemsEstimate = (itemIds: string[], maxPages: number) => useQuery({ queryKey: ['estimate', 'items', itemIds, maxPages], enabled: itemIds.length > 0, staleTime: 60_000,
  queryFn: () => api.post<ItemsEstimate>('/estimate/items', { item_ids: itemIds, max_pages: maxPages }).then(r => r.data) })
export const useCatalogType = (archetype: string) => useQuery({ queryKey: ['catalog', 'type', archetype], enabled: !!archetype,
  queryFn: () => api.get<CatalogType>(`/catalog/types/${archetype}`).then(r => r.data) })
export function useUntrackBulk() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (names: string[]) => api.post<{ removed: number; missing: number }>('/items/untrack-bulk', { names }).then(r => r.data),
    onSuccess: () => { ['items', 'itemsConfig', 'setup', 'scheduler', 'status', 'estimate'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) } })
}
/** Cards for one season: the chosen one, or the current one when none is given. The previous answer stays on screen while a new season loads. */
/** One season's cards (the current one by default); `all`: every item at its own newest trading season. */
export const useOverview = (seasonId?: string, all = false, enabled = true) => useQuery({ queryKey: ['overview', all ? 'all' : seasonId ?? null], refetchInterval: 60_000, placeholderData: prev => prev, enabled,
  queryFn: () => api.get<{ groups: OverviewGroup[] }>('/overview', all ? { all: true } : seasonId ? { season_id: seasonId } : undefined).then(r => r.data.groups) })
export const useCollectEstimate = (name: string, seasonId?: string) => useQuery({ queryKey: ['estimate', 'collect', name, seasonId ?? null], enabled: !!name, staleTime: 60_000,
  queryFn: () => api.get<{ requests: number; known: boolean; since: string | null; until: string | null; opens?: string | null; progress?: { through: string | null; complete: boolean; share: number | null } }>(`/estimate/collect/${name}`, seasonId ? { season_id: seasonId } : undefined).then(r => r.data) })
export interface NetworkState { lan: boolean; port: number; running_port: number; running_lan: boolean; restart_required: boolean; lan_urls: string[] }
export const useNetwork = () => useQuery({ queryKey: ['network'], queryFn: () => api.get<NetworkState>('/setup/network').then(r => r.data) })
export function useSaveNetwork() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (b: { lan: boolean; port: number }) => api.put<NetworkState>('/setup/network', b).then(r => r.data), onSuccess: d => qc.setQueryData(['network'], d) })
}
export const useRestart = () => useMutation({ mutationFn: () => api.post<{ restarting: boolean }>('/setup/restart').then(r => r.data) })
/** Change some tracked items in the cached list at once (no reload of every item); returns the previous list. */
function patchItems(qc: ReturnType<typeof useQueryClient>, names: string[], patch: Partial<ItemInfo>) {
  const prev = qc.getQueryData<ItemInfo[]>(['items'])
  const wanted = new Set(names)
  if (prev) qc.setQueryData<ItemInfo[]>(['items'], prev.map(i => (wanted.has(i.name) ? { ...i, ...patch } : i)))
  return prev
}

// ticks show at once; the save runs in the background and is undone if it fails
export function useSetScheduled() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (b: { names: string[]; scheduled: boolean }) => api.put<{ changed: number }>('/items/scheduled', b).then(r => r.data),
    onMutate: async b => { await qc.cancelQueries({ queryKey: ['items'] }); return { prev: patchItems(qc, b.names, { scheduled: b.scheduled }) } },
    onError: (_e, _b, ctx) => { if (ctx?.prev) qc.setQueryData(['items'], ctx.prev) },
    onSettled: () => { ['estimate', 'scheduler'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) },
  })
}
export function useSetScheduleBulk() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (b: ScheduleInput & { names: string[] }) => api.put<{ changed: number }>('/items/schedule-bulk', b).then(r => r.data),
    onMutate: async b => {
      await qc.cancelQueries({ queryKey: ['items'] })
      const schedule = b.mode === 'default' ? undefined : b.mode === 'interval' ? { mode: 'interval' as const, seconds: b.seconds } : { mode: 'daily' as const, times: b.times }
      return { prev: patchItems(qc, b.names, { schedule }) }
    },
    onError: (_e, _b, ctx) => { if (ctx?.prev) qc.setQueryData(['items'], ctx.prev) },
    onSettled: () => { ['estimate', 'scheduler', 'itemsConfig'].forEach(k => qc.invalidateQueries({ queryKey: [k] })) },
  })
}
// AI market reports (Alibaba Cloud Bailian)
// rules that need upkeep (season dates, patch filters): bundled, or synced from a rules service
export const useRules = () => useQuery({ queryKey: ['rules'], queryFn: () => api.get<RulesStatus>('/rules').then(r => r.data) })
export function useSaveRulesSource() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (b: { url: string; key?: string }) => api.put<RulesStatus>('/rules/source', b).then(r => r.data), onSuccess: d => qc.setQueryData(['rules'], d) })
}
export function useSyncRules() {
  const qc = useQueryClient()
  // new season dates change what every season-based page shows
  return useMutation({ mutationFn: () => api.post<RulesStatus>('/rules/sync').then(r => r.data),
    onSuccess: d => { qc.setQueryData(['rules'], d); for (const k of ['seasons', 'overview', 'series', 'announcements']) qc.invalidateQueries({ queryKey: [k] }) } })
}
export const useLLM = () => useQuery({ queryKey: ['llm'], queryFn: () => api.get<LLMSettings>('/setup/llm').then(r => r.data) })
export function useSaveLLM() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (b: { api_key?: string; model: string; region: string; workspace: string; price_in: number | null; price_out: number | null }) => api.put<LLMSettings>('/setup/llm', b).then(r => r.data),
    onSuccess: d => qc.setQueryData(['llm'], d) })
}
export const useTestLLM = () => useMutation({ mutationFn: () => api.post<{ reply: string; tokens: number }>('/setup/llm/test').then(r => r.data) })
export const useReports = () => useQuery({ queryKey: ['reports'], queryFn: () => api.get<ReportSummary[]>('/reports').then(r => r.data) })
export const useReport = (id: string | null) => useQuery({ queryKey: ['reports', id], enabled: !!id, queryFn: () => api.get<Report>(`/reports/${id}`).then(r => r.data) })
export const useReportEstimate = (days: number, enabled: boolean) => useQuery({ queryKey: ['reports', 'estimate', days], enabled, staleTime: 30_000, retry: false,
  queryFn: () => api.post<ReportEstimate>('/reports/estimate', { days }).then(r => r.data) })
export function useCreateReport() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (days: number) => api.post<{ job_id: string }>('/reports', { days }).then(r => r.data.job_id), onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }) })
}
export function useEditReport() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: ({ id, ...b }: { id: string; public: string; body: string }) => api.put<Report>(`/reports/${id}`, b).then(r => r.data),
    onSuccess: d => { qc.setQueryData(['reports', d.id], d); qc.invalidateQueries({ queryKey: ['reports'], exact: true }) } })
}
export function useDeleteReport() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (id: string) => api.del(`/reports/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ['reports'] }) })
}
