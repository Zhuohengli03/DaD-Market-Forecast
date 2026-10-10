import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { CatalogTable } from './CatalogTable'

vi.mock('../../api/hooks')

const rows = [
  { id: 'id.item.gold_ore_1001', name: 'Gold Ore', rarity: 'common', item_type: 'Misc', market: { name: 'Gold Ore', listings: 12, sold: 3, min_price_per_unit: 30, median_price_per_unit: 41, last_seen: '2026-10-01T00:00:00Z' }, raw: { id: 'id.item.gold_ore_1001', name: 'Gold Ore', rarity: 'common', vendor_price: 5, primary: { weight: 0.5 } } },
  { id: 'id.item.arming_sword_2001', name: 'Arming Sword', rarity: 'uncommon', slot_type: 'Weapon', market: null, raw: { id: 'id.item.arming_sword_2001', name: 'Arming Sword' } },
]

it('lists items with market stats, shows every raw parameter, and tracks', () => {
  const track = vi.fn()
  vi.mocked(hooks.useTrackItem).mockReturnValue({ mutate: track, isPending: false } as never)
  render(<CatalogTable rows={rows as never} tracked={new Set(['Arming Sword'])} />)
  expect(screen.getByText('41.00')).toBeInTheDocument()
  expect(screen.getByText('已追踪')).toBeInTheDocument()
  fireEvent.click(screen.getByText('Gold Ore'))
  expect(screen.getByText('vendor_price')).toBeInTheDocument()
  expect(screen.getByText(/"weight": 0.5/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '追踪' }))
  expect(track.mock.calls[0][0]).toBe('id.item.gold_ore_1001')
})
