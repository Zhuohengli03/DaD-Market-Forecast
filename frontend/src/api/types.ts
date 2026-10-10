export interface ItemInfo { name: string; item_id?: string; file?: string; csv?: string; category: string; description: string; enabled: boolean; rows: number; first_ts: string | null; last_ts: string | null }
export interface SeriesPoint { ts: string; median_price: number | null; min_price: number | null; max_price: number | null; p25: number | null; p75: number | null; listings: number; sold: number; volume: number }
export interface SeriesResponse { item: string; resolution: Resolution; start: string | null; end: string | null; points: SeriesPoint[] }
export type Resolution = '1h' | '4h' | '1d'
export interface Listing { item_id: string; item: string; quantity: number; price_per_unit: number; price: number; has_sold: boolean; created_at: string }
export interface Season { id: string; name: string; start: string; end: string | null; note: string }
export interface SeasonInput { name: string; start: string; end: string | null; note: string }
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'
export interface Job { id: string; kind: 'forecast' | 'collect' | 'task'; label: string; status: JobStatus; created_at: string; finished_at: string | null; error: string | null; result_ref: string | null; meta: Record<string, string>; log_tail: string[] }
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
export interface SetupState { api_key_configured: boolean; api_key_hint: string | null; data_source: string; seasons: number; tracked_items: number; catalog_items: number }
export interface MarketStats { name: string; listings: number; sold: number; min_price_per_unit: number; median_price_per_unit: number; last_seen: string }
export interface CatalogRow { id: string; name: string; rarity?: string; item_type?: string; slot_type?: string; armor_type?: string; weapon_type?: string; hand_type?: string; gear_score?: number; vendor_price?: number; archetype?: string; market: MarketStats | null; raw: Record<string, unknown> }
export interface CatalogMeta { patch: string | null; synced_at: string | null; items: number; market_scanned_at: string | null; market_since: string | null; market_covered_from: string | null; market_covered_to: string | null; market_listings_read: number; tradable: number }
export interface CatalogFacets { rarity: string[]; item_type: string[] }
