import { buildQuery, formatStat } from './filters'

it('builds a query from filter rows, skipping empty ones', () => {
  const q = buildQuery('id.item.arming_sword', 'legendary', [
    { key: 'secondary_armor_penetration', min: '2', max: '5' },
    { key: 'secondary_luck', min: '', max: '' },
    { key: 'primary_weapon_damage', min: '', max: '34' },
  ], true, 10)
  expect(q).toEqual({ archetype: 'id.item.arming_sword', rarity: 'legendary', sold_only: true, max_pages: 10, sort: 'created_at:desc', filters: [
    { kind: 'secondary', name: 'armor_penetration', min: 2, max: 5 },
    { kind: 'primary', name: 'weapon_damage', min: null, max: 34 },
  ] })
  expect(buildQuery('id.item.x', '', [], false, 5).rarity).toBeNull()
})

it('formats stat values with labels and percent', () => {
  const labels = { secondary_armor_penetration: { label: 'Armor Penetration', percent: true }, primary_weapon_damage: { label: 'Weapon Damage', percent: false } }
  expect(formatStat('secondary_armor_penetration', 2.9, labels)).toBe('Armor Penetration 2.9%')
  expect(formatStat('primary_weapon_damage', 32, labels)).toBe('Weapon Damage 32')
  expect(formatStat('secondary_unknown_thing', 1, labels)).toBe('unknown thing 1')
})
