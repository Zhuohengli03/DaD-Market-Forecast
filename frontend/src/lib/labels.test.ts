import i18n from '../i18n'
import { modelLabel } from './labels'

it('gives models plain-language names', () => {
  expect(modelLabel('naive', i18n.t)).toBe('沿用上一小时')
  expect(modelLabel('seasonal_naive_24', i18n.t)).toBe('沿用昨天同一时刻')
  expect(modelLabel('mystery', i18n.t)).toBe('mystery')
})

import { itemName, localizeMessage, paramLabel, rarityLabel, typeLabel } from './labels'

it('translates game codes and picks Chinese names', () => {
  expect(rarityLabel('legendary', i18n.t)).toBe('传说')
  expect(rarityLabel('mystery', i18n.t)).toBe('mystery')
  expect(typeLabel('weapon', i18n.t)).toBe('武器')
  expect(paramLabel('vendor_price', i18n.t)).toBe('商人收购价')
  expect(itemName({ name: 'Iron Ore', name_zh: '铁矿石' }, 'zh')).toBe('铁矿石')
  expect(itemName({ name: 'Iron Ore', name_zh: '铁矿石' }, 'en')).toBe('Iron Ore')
  expect(itemName({ name: 'Iron Ore', name_zh: null }, 'zh')).toBe('Iron Ore')
})

it('shows only the matching half of bilingual server messages', () => {
  expect(localizeMessage('请先同步道具库 / sync the catalog first', 'zh')).toBe('请先同步道具库')
  expect(localizeMessage('请先同步道具库 / sync the catalog first', 'en')).toBe('sync the catalog first')
  expect(localizeMessage('only one language', 'zh')).toBe('only one language')
})
