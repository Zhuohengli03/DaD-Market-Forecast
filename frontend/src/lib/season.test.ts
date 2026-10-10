import i18n from '../i18n'
import { seasonFirstLabel, seasonLookedUp } from './season'

it('shows the first listing of this season, or none, or a dash before looking it up', () => {
  expect(seasonFirstLabel({ season_listings: 12, season_first: '2026-10-09T00:03:10Z' }, i18n.t)).toBe('2026-10-09 00:03Z')
  expect(seasonFirstLabel({ season_listings: 0, season_first: null }, i18n.t)).toBe('无')
  expect(seasonFirstLabel({ season_listings: null, season_first: null }, i18n.t)).toBe('–')
  expect(seasonLookedUp({ season_listings: 5, season_first: null })).toBe(false)   // looked up before this field existed
})
