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

import { paramValue, patchLabel } from './labels'
import { fmtTs } from './format'

it('translates parameter values, patch and season names', () => {
  expect(paramValue('rarity', 'unique', i18n.t)).toBe('独特')
  expect(paramValue('item_type', 'accessory', i18n.t)).toBe('饰品')
  expect(paramValue('season_label', 'Season 9', i18n.t)).toBe('第 9 赛季')
  expect(paramValue('has_model', true, i18n.t)).toBe('是')
  expect(paramValue('gear_score', 30, i18n.t)).toBe('30')
  expect(patchLabel('Early Access Season #11', i18n.t)).toBe('抢先体验第 11 赛季')
  expect(i18n.t('param.has_model')).toBe('有 3D 模型')
})

it('formats times in Chinese when the interface is Chinese', () => {
  expect(fmtTs('2026-10-09T06:42:00Z')).toMatch(/月/)
})

it('has Chinese names for every category DarkerDB uses', () => {
  for (const c of ['food', 'soul_heart', 'ore', 'drink', 'hunting_loot']) expect(i18n.t(`category.${c}`)).toMatch(/[\u4e00-\u9fff]/)
})

import { localizeWarning, seasonLabel } from './labels'

it('localizes quality suffixes and historical warnings while preserving English and unknown names', () => {
  const item = { name: 'Potion (rare)', name_zh: '药水 (rare)' }
  expect(itemName(item, 'zh')).toBe('药水（稀有）')
  expect(itemName(item, 'en')).toBe('Potion (rare)')
  expect(itemName({ name: 'Mystery (Custom)' }, 'zh')).toBe('Mystery (Custom)')
  expect(localizeWarning('4 forward-filled hour-steps excluded from backtest scoring', 'zh')).toContain('4 个前向填充')
  expect(localizeWarning('方法 hgb 运行失败：数据不足', 'zh')).toBe('方法 梯度提升树 运行失败：数据不足')
  expect(localizeMessage('logs / archive', 'zh')).toBe('logs / archive')
  expect(seasonLabel('Early Access Season #10', i18n.t)).toBe('第 10 赛季')
  expect(patchLabel('Early Access Hotfix #121-1', i18n.t)).toBe('抢先体验热修复 #121-1')
  expect(patchLabel('Season #10 rewards preview', i18n.t)).toBe('Season #10 rewards preview')
})
