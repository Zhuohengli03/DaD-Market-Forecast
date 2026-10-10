export interface ItemInfo { name: string; name_zh?: string | null; item_id?: string; file?: string; csv?: string; category: string; description: string; enabled: boolean; rows: number; first_ts: string | null; last_ts: string | null }
export interface SeriesPoint { ts: string; median_price: number | null; min_price: number | null; max_price: number | null; p25: number | null; p75: number | null; listings: number; sold: number; volume: number }
export interface SeriesResponse { item: string; resolution: Resolution; start: string | null; end: string | null; points: SeriesPoint[] }
export type Resolution = '1h' | '4h' | '1d'
export interface Listing { item_id: string; item: string; quantity: number; price_per_unit: number; price: number; has_sold: boolean; created_at: string }
export interface Season { id: string; name: string; start: string; end: string | null; note: string }
export interface SeasonInput { name: string; start: string; end: string | null; note: string }
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'
export interface JobProgress { stage: string; done: number | null; total: number | null; detail: Record<string, unknown> }
export type JobKind = 'forecast' | 'collect' | 'task' | 'catalog' | 'init'
export interface Job { id: string; kind: JobKind; label: string; status: JobStatus; created_at: string; finished_at: string | null; error: string | null; result_ref: string | null; meta: Record<string, string>; progress: JobProgress | null }
export interface ForecastInput { item: string; season_id: string; horizon_hours: number; models: string[]; coverage: number; backtest_origin_step?: number }
export interface RunSummary { run_id: string; created_at: string; source: string; beats_baseline: boolean; skill_vs_naive: number; item: string; season_id: string; season_name: string; horizon_hours: number; selected: string; warnings: number }
export interface ModelMetric { name: string; mae_h1: number | null; mae_h24: number | null; mae_mean: number | null; rmse_mean: number | null; mase_mean: number | null; direction_acc: number | null }
export interface ForecastRun {
  run_id: string; created_at: string; source: string
  request: { item: string; season_id: string; horizon_hours: number; models: string[]; coverage: number }
  season: { id: string; name: string; start: string; end: string | null }
  series_tail: { ts: string; median_price: number | null; p25: number | null; p75: number | null; volume: number; listings: number }[]
  forecast: { ts: string; h: number; pred: number; lower: number; upper: number }[]
  selected: { name: string; members: { name: string; weight: number }[] }
  backtest: { origins: number; horizon: number; coverage: number | null; nominal_coverage: number; models: ModelMetric[]; per_horizon: Record<string, (number | null)[]>; failures: Record<string, string> }
  beats_baseline: boolean; skill_vs_naive: number; data_age_hours?: number; baseline_reason?: string | null
  feature_importance: { name: string; value: number }[]
  references: { season_id: string; name: string; points: { h: number; median_price: number }[] }[]
  warnings: string[]
}
export interface Actual { ts: string; median_price: number }
export interface SchedulerTask { name: string; item?: string; script_path: string; schedule_type: string; schedule_value: string; enabled: boolean; description: string }
export interface SchedulerStatus { running: boolean; tasks: SchedulerTask[] }
export interface ItemConfigEntry { name: string; item_id?: string; file?: string; csv?: string; category: string; description: string; enabled: boolean; schedule?: string }
export interface ItemsConfig { items: ItemConfigEntry[]; categories: Record<string, string>; auto_discovery?: Record<string, unknown> }
export interface StatusResponse { source: string; python: string; items: { name: string; rows: number; first_ts: string | null; last_ts: string | null }[]; jobs_running: Job[]; scheduler_running: boolean }
export interface Environment { python_version: string; python_ok: boolean; storage: string; data_dir: string; db_host: string | null; db_database: string | null }
export interface SetupState { initialized: boolean; collect_interval_hours: number; auto_collect: boolean; environment: Environment; api_key_configured: boolean; api_key_hint: string | null; data_source: string; seasons: number; tracked_items: number; catalog_items: number }
export interface InitItem { item_id: string; name: string; name_zh?: string | null; status: 'waiting' | 'collecting' | 'collected' | 'collect_failed' | 'forecasting' | 'done'; pages: number; collected: number; forecast: 'ok' | 'not_enough_data' | 'failed' | null; run_id: string | null; hours: number | null; error?: string }
export interface DatabaseInput { host: string; port: number; database: string; user: string; password: string }
export interface MarketStats { name: string; listings: number; sold: number; min_price_per_unit: number; median_price_per_unit: number; last_seen: string }
export interface CatalogVariant { id: string; rarity: string | null; gear_score?: number | null; vendor_price?: number | null; num_secondary_attributes?: number | null; icon_url?: string | null; raw: Record<string, unknown> }
export interface CatalogType { archetype: string; name: string; name_zh?: string | null; item_type: string | null; slot_type?: string | null; icon_url: string | null; has_stats: boolean; rarities: string[]; listings_total: number | null; variants: CatalogVariant[] }
export interface CatalogMeta { patch: string | null; synced_at: string | null; items: number; types: number; types_total: number; trade_checked: number; trade_check_complete: boolean; trade_checked_at: string | null; with_stats: number }
export interface CatalogFacets { item_type: string[] }
export interface EquipmentStat { key: string; kind: 'primary' | 'secondary'; name: string; label: string; label_zh?: string | null; percent: boolean; min: number; max: number; seen: number }
export interface EquipmentListing { id: string; item_id: string | null; name: string | null; rarity: string | null; quantity: number; price: number; price_per_unit: number; has_sold: boolean; listing_state: string | null; created_at: string; attributes: Record<string, number> }
export interface EquipmentDay { day: string; listings: number; sold: number; median: number; low: number; high: number; sold_median: number | null }
export interface EquipmentResult { count: number; listings: EquipmentListing[]; daily: EquipmentDay[]; covered_from: string | null; covered_to: string | null; truncated: boolean; since?: string | null }
export interface EquipmentQueryBody { archetype: string; rarity: string | null; sold_only: boolean; max_pages: number; filters: { kind: 'primary' | 'secondary'; name: string; min: number | null; max: number | null }[] }
