import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useParams } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { CatalogTable } from './CatalogTable'

vi.mock('../../api/hooks')

const v = (id: string, rarity: string, extra = {}) => ({ id, rarity, gear_score: 5, vendor_price: 3, num_secondary_attributes: 0, icon_url: null, raw: { id, rarity, vendor_price: 3, ...extra } })
const rows = [
  { archetype: 'id.item.iron_ores', name: 'Iron Ore', name_zh: '铁矿石', item_type: 'misc', has_stats: false, icon_url: 'https://cdn.darkerdb.com/codex/abc', rarities: ['common'], listings_total: 58042, variants: [v('id.item.iron_ores_2001', 'common')] },
  { archetype: 'id.item.bandage', name: 'Bandage', name_zh: '绷带', item_type: 'utility', has_stats: false, icon_url: null, rarities: ['poor', 'common'], listings_total: 900, variants: [v('id.item.bandage_1001', 'poor'), v('id.item.bandage_2001', 'common', { duration: 4 })] },
  { archetype: 'id.item.arming_sword', name: 'Arming Sword', name_zh: '武装剑', item_type: 'weapon', has_stats: true, icon_url: null, rarities: ['common', 'legendary'], listings_total: 95498, variants: [v('id.item.arming_sword_6001', 'legendary')] },
]

function DetailProbe() { return <p>detail:{useParams().archetype}</p> }

function setup(track = vi.fn()) {
  vi.mocked(hooks.useTrackItem).mockReturnValue({ mutate: track, isPending: false } as never)
  vi.mocked(hooks.useTypeHistory).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<MemoryRouter><Routes>
    <Route path="/" element={<CatalogTable rows={rows as never} tracked={new Set(['id.item.iron_ores_2001'])} />} />
    <Route path="/catalog/:archetype" element={<DetailProbe />} />
  </Routes></MemoryRouter>)
  return track
}

it('shows one row per type with listings, icon and stats badge', () => {
  setup()
  expect(screen.getByText('58042')).toBeInTheDocument()
  expect(document.querySelector('img')?.getAttribute('src')).toBe('https://cdn.darkerdb.com/codex/abc')
  expect(screen.getByText('有词条')).toBeInTheDocument()
  expect(screen.getByText('铁矿石')).toBeInTheDocument()
  expect(screen.getByText('武器')).toBeInTheDocument()
  expect(screen.getByText('粗劣 · 普通')).toBeInTheDocument()
  expect(screen.getByText('已追踪')).toBeInTheDocument()
})

it('opens the item detail page when a row is clicked', () => {
  setup()
  fireEvent.click(screen.getByText('粗劣 · 普通'))
  expect(screen.getByText('detail:id.item.bandage')).toBeInTheDocument()
})

it('sends equipment to the equipment market instead of tracking', () => {
  setup()
  expect(screen.getByRole('link', { name: '查看行情' })).toHaveAttribute('href', '/equipment?type=id.item.arming_sword')
})
