import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { CatalogTable } from './CatalogTable'

vi.mock('../../api/hooks')

const v = (id: string, rarity: string, extra = {}) => ({ id, rarity, gear_score: 5, vendor_price: 3, num_secondary_attributes: 0, icon_url: null, raw: { id, rarity, vendor_price: 3, ...extra } })
const rows = [
  { archetype: 'id.item.iron_ores', name: 'Iron Ore', item_type: 'misc', has_stats: false, icon_url: 'https://cdn.darkerdb.com/codex/abc', rarities: ['common'], listings_total: 58042, variants: [v('id.item.iron_ores_2001', 'common')] },
  { archetype: 'id.item.bandage', name: 'Bandage', item_type: 'utility', has_stats: false, icon_url: null, rarities: ['poor', 'common'], listings_total: 900, variants: [v('id.item.bandage_1001', 'poor'), v('id.item.bandage_2001', 'common', { duration: 4 })] },
  { archetype: 'id.item.arming_sword', name: 'Arming Sword', item_type: 'weapon', has_stats: true, icon_url: null, rarities: ['common', 'legendary'], listings_total: 95498, variants: [v('id.item.arming_sword_6001', 'legendary')] },
]

function setup(track = vi.fn()) {
  vi.mocked(hooks.useTrackItem).mockReturnValue({ mutate: track, isPending: false } as never)
  render(<MemoryRouter><CatalogTable rows={rows as never} tracked={new Set(['id.item.iron_ores_2001'])} /></MemoryRouter>)
  return track
}

it('shows one row per type with listings, icon and stats badge', () => {
  setup()
  expect(screen.getByText('58042')).toBeInTheDocument()
  expect(document.querySelector('img')?.getAttribute('src')).toBe('https://cdn.darkerdb.com/codex/abc')
  expect(screen.getByText('有词条')).toBeInTheDocument()
  expect(screen.getByText('已追踪')).toBeInTheDocument()
})

it('tracks a specific rarity of a stat-free type and shows all parameters', () => {
  const track = setup()
  fireEvent.click(screen.getByRole('button', { name: 'Bandage' }))
  expect(screen.getByText('duration')).toBeInTheDocument()
  fireEvent.click(screen.getAllByRole('button', { name: '追踪' })[1])
  expect(track.mock.calls[0][0]).toBe('id.item.bandage_2001')
})

it('sends equipment to the equipment market instead of tracking', () => {
  setup()
  expect(screen.getByRole('link', { name: '查看行情' })).toHaveAttribute('href', '/equipment?type=id.item.arming_sword')
})
