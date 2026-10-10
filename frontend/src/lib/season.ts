import type { TFunction } from 'i18next'
import type { CatalogType } from '../api/types'
import { fmtTsUtc } from './format'

/** Whether this season's numbers were looked up: a count, plus its first listing unless there was none. */
export const seasonLookedUp = (r: Pick<CatalogType, 'season_listings' | 'season_first'>) =>
  r.season_listings != null && (r.season_listings === 0 || !!r.season_first)

/** First listing of the current season (UTC), "none" when the season has no listing yet, "–" when not looked up. */
export function seasonFirstLabel(r: Pick<CatalogType, 'season_listings' | 'season_first'>, t: TFunction): string {
  if (r.season_first) return fmtTsUtc(r.season_first)
  return seasonLookedUp(r) ? t('catalog.noListing') : '–'
}
