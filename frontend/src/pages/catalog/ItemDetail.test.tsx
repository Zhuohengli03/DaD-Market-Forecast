import { cleanup, render, screen, fireEvent, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import ItemDetail from './ItemDetail'

vi.mock('../../api/hooks')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

const v = (id: string, rarity: string, extra = {}) => ({ id, rarity, gear_score: 5, vendor_price: 3, icon_url: null, raw: { id, rarity, vendor_price: 3, ...extra } })
const bandage = { archetype: 'id.item.bandage', name: 'Bandage', name_zh: '绷带', category: 'consumable', item_type: 'utility', has_stats: false, icon_url: 'https://x/b.png',
  rarities: ['poor', 'common'], listings_total: 900, first_listed: '2024-09-06T10:00:00Z', season_listings: null, variants: [v('id.item.bandage_1001', 'poor'), v('id.item.bandage_2001', 'common', { duration: 4 })] }

function mount(track = vi.fn(), history = vi.fn()) {
  vi.mocked(hooks.useOverview).mockReturnValue({ data: [{ key: 'id.item.bandage', archetype: 'id.item.bandage', name: 'Bandage', name_zh: '绷带', category: 'consumable', icon_url: null, rows: 50, season_rows: 42,
    variants: [{ name: 'Bandage (poor)', label: '粗劣', label_en: 'Poor', rarity: 'poor', rows: 50, season_rows: 42, last_ts: null, latest: 12.5, change24: -0.1, spark: [] }] }] } as never)
  vi.mocked(hooks.useCatalogType).mockReturnValue({ data: bandage, isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Bandage (poor)', item_id: 'id.item.bandage_1001', rows: 50, season_rows: 42 }] } as never)
  vi.mocked(hooks.useTrackItem).mockReturnValue({ mutate: track, isPending: false } as never)
  vi.mocked(hooks.useTypeHistory).mockReturnValue({ mutate: history, isPending: false } as never)
  render(<MemoryRouter initialEntries={['/catalog/id.item.bandage']}><Routes><Route path="/catalog/:archetype" element={<ItemDetail />} /></Routes></MemoryRouter>)
}

it('shows every detail of one item: names, listing counts, rarities and all parameters', () => {
  mount()
  expect(screen.getByRole('heading', { name: /绷带/ })).toBeInTheDocument()
  expect(screen.getByText('900')).toBeInTheDocument()
  expect(screen.queryByText('2024-09-06')).not.toBeInTheDocument()   // DarkerDB's record start is not shown
  expect(screen.getAllByText('42').length).toBeGreaterThan(0)       // this season from local data
  expect(screen.getByText('duration')).toBeInTheDocument()           // highest rarity's parameters shown first
  expect(screen.getByRole('link', { name: '返回道具库' })).toHaveAttribute('href', '/catalog')
})

it('tracks another rarity and fetches season numbers on demand', () => {
  const track = vi.fn(); const history = vi.fn()
  mount(track, history)
  fireEvent.click(screen.getByRole('button', { name: '追踪' }))
  expect(track.mock.calls[0][0]).toBe('id.item.bandage_2001')
  fireEvent.click(screen.getByRole('button', { name: /查询本赛季挂单/ }))
  expect(history.mock.calls[0][0]).toBe('id.item.bandage')
})


it('shows the price of each tracked quality', () => {
  mount()
  expect(screen.getByText('各品质价格')).toBeInTheDocument()
  expect(screen.getByText('12.50')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '看走势' })).toHaveAttribute('href', '/market?item=Bandage%20(poor)')
})

it('after a lookup with no listings, shows 0 and "none" instead of dashes, without the lookup button', () => {
  mount()
  vi.mocked(hooks.useCatalogType).mockReturnValue({ data: { ...bandage, listings_total: 0, first_listed: null, season_listings: 0, season_first: null }, isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [] } as never)
  cleanup()
  render(<MemoryRouter initialEntries={['/catalog/id.item.bandage']}><Routes><Route path="/catalog/:archetype" element={<ItemDetail />} /></Routes></MemoryRouter>)
  const stats = screen.getByRole('region', { name: '挂单统计' })
  expect(within(stats).getAllByText('0')).toHaveLength(2)
  expect(within(stats).getByText('无')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /查询本赛季挂单/ })).not.toBeInTheDocument()
})


it('shows when this season\'s first listing was', () => {
  mount()
  vi.mocked(hooks.useCatalogType).mockReturnValue({ data: { ...bandage, season_listings: 12, season_first: '2026-10-09T00:03:10Z' }, isLoading: false } as never)
  cleanup()
  render(<MemoryRouter initialEntries={['/catalog/id.item.bandage']}><Routes><Route path="/catalog/:archetype" element={<ItemDetail />} /></Routes></MemoryRouter>)
  const stats = screen.getByRole('region', { name: '挂单统计' })
  expect(within(stats).getByText('本赛季首条')).toBeInTheDocument()
  expect(within(stats).getByText('2026-10-09 00:03Z')).toBeInTheDocument()
})
