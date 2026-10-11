import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import ItemPage from './ItemPage'

vi.mock('../../api/hooks')
vi.mock('echarts-for-react', () => ({ default: () => null }))
vi.mock('./ItemForecast', () => ({ ItemForecast: () => <p>forecast card</p> }))
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
vi.mock('../market/RefreshData', () => ({ RefreshData: () => <span>refresh</span> }))
vi.mock('../market/ListingsTable', () => ({ ListingsTable: () => <span>listings</span> }))

const v = (id: string, rarity: string) => ({ id, rarity, gear_score: null, vendor_price: 3, icon_url: null, raw: { id, rarity } })
const ruby = { archetype: 'id.item.ruby', name: 'Ruby', name_zh: '红宝石', category: 'gem', item_type: 'misc', has_stats: false, icon_url: null,
  rarities: ['common', 'rare'], listings_total: 100, season_listings: 10, season_first: null, variants: [v('id.item.ruby_2001', 'common'), v('id.item.ruby_4001', 'rare')] }
const SEASONS = [{ id: 's10', name: 'Season 10', start: '2026-07-17T00:00:00Z', end: '2026-10-09T03:00:00Z', note: '' },
  { id: 's11', name: 'Season 11', start: '2026-10-09T03:00:00Z', end: null, note: '' }]

function mount(items: unknown[]) {
  for (const [k, val] of Object.entries({
    useCatalogType: { data: ruby, isLoading: false }, useItems: { data: items }, useSeasons: { data: SEASONS },
    useSeries: { data: { points: [] }, isLoading: false }, useNextHour: { data: undefined, isError: true, error: new Error('x') },
    useOverview: { data: [] }, useTrackItem: { mutate: vi.fn() }, useTypeHistory: { mutate: vi.fn() }, useCatalogMeta: { data: {} },
    useItemAnnouncements: { data: { name: 'Ruby', mentions: [] } }, useStartForecast: { mutate: vi.fn(), isPending: false }, useAnnouncements: { data: { items: [] } }, useSeasonCompare: { data: { seasons: [] } }, useRuns: { data: [] },
  })) vi.mocked(hooks[k as keyof typeof hooks] as never as (...a: unknown[]) => unknown).mockReturnValue(val as never)
  render(<MemoryRouter initialEntries={['/item/id.item.ruby']}><Routes><Route path="/item/:archetype" element={<ItemPage />} /></Routes></MemoryRouter>)
}

it('puts one item on one page: header, quality switch, price chart, facts', () => {
  mount([{ name: 'Ruby (Normal) (common)', item_id: 'id.item.ruby_2001', rows: 5 }, { name: 'Ruby (Rare)', item_id: 'id.item.ruby_4001', rows: 2 }])
  expect(screen.getByRole('heading', { name: /红宝石/ })).toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: '品质' })).toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: '赛季' })).toHaveValue('s11')          // the latest season by default
  expect(screen.getByText('道具资料')).toBeInTheDocument()
})

it('an untracked item shows its facts and a track button only', () => {
  mount([])
  expect(screen.getAllByRole('button', { name: '追踪' }).length).toBeGreaterThan(0)
  expect(screen.queryByRole('combobox', { name: '赛季' })).not.toBeInTheDocument()
})

it('before the newest season trades, opens on the last season that did, and the header shows no price for it', () => {
  const closed = [SEASONS[0], { ...SEASONS[1], market_open: '2099-01-01' }]
  mount([{ name: 'Ruby (Normal) (common)', item_id: 'id.item.ruby_2001', rows: 5 }])
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: closed } as never)
  vi.mocked(hooks.useOverview).mockReturnValue({ data: [{ key: 'id.item.ruby', archetype: 'id.item.ruby', name: 'Ruby', name_zh: '红宝石', category: 'gem', icon_url: null, rows: 5, season_rows: 0,
    variants: [{ name: 'Ruby (Normal) (common)', label: '普通', label_en: 'Common', rarity: 'common', rows: 5, season_rows: 0, last_ts: null, latest: null, change24: null, spark: [] }] }] } as never)
  cleanup()
  render(<MemoryRouter initialEntries={['/item/id.item.ruby']}><Routes><Route path="/item/:archetype" element={<ItemPage />} /></Routes></MemoryRouter>)
  expect(screen.getByRole('combobox', { name: '赛季' })).toHaveValue('s10')
  expect(screen.queryByText('往届数据')).not.toBeInTheDocument()
  expect(hooks.useOverview).toHaveBeenLastCalledWith('s10')            // the header asks for the open season's prices
})

it('the season is picked once, in the header, and the price section follows it', () => {
  mount([{ name: 'Ruby (Normal) (common)', item_id: 'id.item.ruby_2001', rows: 5 }])
  const pickers = screen.getAllByRole('combobox', { name: '赛季' })
  expect(pickers).toHaveLength(1)
  expect(pickers[0].closest('section')).toHaveTextContent('红宝石')            // the first block, with the name
  fireEvent.change(pickers[0], { target: { value: 's10' } })
  expect(hooks.useSeries).toHaveBeenLastCalledWith('Ruby (Normal) (common)', expect.objectContaining({ season_id: 's10' }))
})
