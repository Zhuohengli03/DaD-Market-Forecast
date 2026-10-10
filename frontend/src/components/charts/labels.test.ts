import i18n from '../../i18n'
import { featureLabel } from './ImportanceChart'
import { buildPriceOption, niceCeil } from './PriceChart'
import { itemNote, seasonLabel } from '../../lib/labels'

it('rounds axis tops to round numbers', () => {
  expect(niceCeil(341)).toBe(400)
  expect(niceCeil(230)).toBe(250)
  expect(niceCeil(0.7)).toBe(0.8)
})

it('names model inputs in plain Chinese', () => {
  expect(featureLabel('price_lag_24', i18n.t)).toBe('24 小时前的价格')
  expect(featureLabel('price_roll_mean_6', i18n.t)).toBe('过去 6 小时均价')
  expect(featureLabel('hours_since_start', i18n.t)).toBe('赛季开始后过了多久')
})

it('translates old season names and hides raw rarity notes', () => {
  expect(seasonLabel('Season 2', i18n.t)).toBe('第 2 赛季')
  expect(seasonLabel('第 7 赛季', i18n.t)).toBe('第 7 赛季')
  expect(itemNote('common')).toBe('')
  expect(itemNote('黄金矿石')).toBe('黄金矿石')
})

it('gives every price-chart series its own colour so the legend matches the lines', () => {
  const opt = buildPriceOption([{ ts: '2026-10-09T00:00:00Z', median_price: 10, min_price: 9, max_price: 11, p25: 9.5, p75: 10.5, listings: 3, sold: 1, volume: 5 }] as never)
  for (const s of opt.series as { color?: string }[]) expect(s.color).toMatch(/^#/)
})
