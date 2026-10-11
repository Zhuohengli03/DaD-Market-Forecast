import { render, screen, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import Dashboard from './Dashboard'
import * as hooks from '../api/hooks'

vi.mock('../api/hooks')
vi.mock('../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
vi.mock('echarts-for-react', () => ({ default: () => null }))   // no canvas in the test DOM

const spark = Array(48).fill(10)
const v = (name: string, label: string, rarity: string | null, latest: number | null, change24: number | null, rows = 30) =>
  ({ name, label, label_en: label, rarity, rows, season_rows: rows, last_ts: rows ? '2026-10-09T10:00:00Z' : null, latest, change24, spark, season_id: rows ? 's2' : null, season_name: rows ? 'Season 2' : null })
const GROUPS = [
  { key: 'id.item.ruby', archetype: 'id.item.ruby', name: 'Ruby', name_zh: '红宝石', category: 'gem', icon_url: null, rows: 60, season_rows: 60,
    variants: [v('Ruby (Cracked) (poor)', '裂开', 'poor', 1.5, 0.2), v('Ruby (Flawed) (common)', '瑕疵', 'common', 3, -0.5)] },
  { key: 'id.item.gold_ores', archetype: 'id.item.gold_ores', name: 'Gold Ore', name_zh: '金矿', category: 'ore', icon_url: null, rows: 30, season_rows: 30,
    variants: [v('Gold Ore', '史诗', 'epic', 216.67, null)] },
]

const SEASONS = [{ id: 's1', name: 'Season 1', start: '2025-10-01T00:00:00Z', end: '2026-01-01T00:00:00Z', note: '' }, { id: 's2', name: 'Season 2', start: '2026-01-01T00:00:00Z', end: null, note: '' }]

function mount(items: object[] = [], groups: object[] = GROUPS) {
  vi.mocked(hooks.useOverview).mockReturnValue({ data: groups, isLoading: false } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: SEASONS } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: items, isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'sqlite', jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ isSuccess: false } as never)
  vi.mocked(hooks.useUntrackBulk).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Dashboard /></MemoryRouter></QueryClientProvider>)
}

it('shows one card per item type, with every quality inside', () => {
  mount()
  const ruby = screen.getByRole('link', { name: '红宝石' })
  expect(ruby).toHaveAttribute('href', '/item/id.item.ruby')
  expect(within(ruby).getByText('裂开')).toBeInTheDocument()
  expect(within(ruby).getByText('-50.0%')).toBeInTheDocument()
  expect(within(screen.getByRole('link', { name: '金矿' })).getByText('216.67')).toBeInTheDocument()
  expect(screen.getByText('矿石')).toBeInTheDocument()          // grouped by category
})

it('filters by search', () => {
  mount()
  fireEvent.change(screen.getByLabelText('搜索道具'), { target: { value: '瑕疵' } })
  return new Promise(r => setTimeout(r, 250)).then(() => {
    expect(screen.queryByRole('link', { name: '金矿' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '红宝石' })).toBeInTheDocument()
  })
})

it('keeps items without data in one card', () => {
  mount([{ name: 'Iron Ore', name_zh: '铁矿', rows: 0, last_ts: null, enabled: true }])
  expect(screen.getByText('1 个道具还没有数据')).toBeInTheDocument()
})

it('a card shows next hour range and a recent update, and opens the item page', () => {
  const groups = [{ ...GROUPS[0], recent_update: { title: 'Early Access Hotfix #127', date: '2026-09-29T00:00:00Z' },
    variants: GROUPS[0].variants.map(x => ({ ...x, next_hour: { p25: 17.4, p75: 23.4 } })) }]
  vi.mocked(hooks.useOverview).mockReturnValue({ data: groups, isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [], isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'sqlite', jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ isSuccess: false } as never)
  vi.mocked(hooks.useUntrackBulk).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Dashboard /></MemoryRouter></QueryClientProvider>)
  expect(screen.getByText('下一小时：17.40–23.40')).toBeInTheDocument()
  expect(screen.getByText('Hotfix #127')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '红宝石' })).toHaveAttribute('href', '/item/id.item.ruby')
})

it('hides a next hour range built on old data', () => {
  const groups = [{ ...GROUPS[0], variants: GROUPS[0].variants.map(x => ({ ...x, next_hour: { p25: 17.4, p75: 23.4, stale_hours: 10 } })) }]
  vi.mocked(hooks.useOverview).mockReturnValue({ data: groups, isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [], isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'sqlite', jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ isSuccess: false } as never)
  vi.mocked(hooks.useUntrackBulk).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Dashboard /></MemoryRouter></QueryClientProvider>)
  expect(screen.queryByText(/下一小时：/)).not.toBeInTheDocument()
})

it('a price on a card says what it is and which hour it is from', () => {
  const gold = { ...GROUPS[1], variants: GROUPS[1].variants.map(x => ({ ...x, latest: 148, latest_ts: '2026-10-09T02:00:00+00:00', latest_listings: 51 })) }
  vi.mocked(hooks.useOverview).mockReturnValue({ data: [gold], isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [], isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'sqlite', jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ isSuccess: false } as never)
  vi.mocked(hooks.useUntrackBulk).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Dashboard /></MemoryRouter></QueryClientProvider>)
  const card = screen.getByRole('link', { name: '金矿' })
  expect(within(card).getByText('148.00')).toBeInTheDocument()
  expect(within(card).getByText('每个的成交中位价')).toBeInTheDocument()
  expect(within(card).getByText('2026-10-09 02:00Z 那一小时 · 51 条挂单')).toBeInTheDocument()
})

it('shows every item, whatever season its data is from, with no season picker', () => {
  mount()
  expect(screen.queryByRole('combobox', { name: '赛季' })).not.toBeInTheDocument()
  expect(hooks.useOverview).toHaveBeenCalledWith(undefined, true)   // cards call it too, disabled until they switch season
})

it('each card opens on its own newest trading season, and can switch to another season by itself', () => {
  const s1Gold = { ...GROUPS[1], variants: GROUPS[1].variants.map(x => ({ ...x, season_id: 's1', season_name: 'Season 1' })) }
  mount([], [GROUPS[0], s1Gold])
  const goldPick = screen.getByRole('combobox', { name: '金矿 赛季' })
  expect(goldPick).toHaveValue('s1')
  expect(screen.getByRole('combobox', { name: '红宝石 赛季' })).toHaveValue('s2')
  // season 2 has no gold data: the card shows dashes, the ruby card is untouched
  vi.mocked(hooks.useOverview).mockImplementation(((season?: string, all?: boolean) => all || !season ? { data: [GROUPS[0], s1Gold], isLoading: false }
    : { data: [{ ...GROUPS[1], season_rows: 0, variants: GROUPS[1].variants.map(x => ({ ...x, season_rows: 0, latest: null })) }], isLoading: false }) as never)
  fireEvent.change(goldPick, { target: { value: 's2' } })
  const gold = screen.getByRole('link', { name: '金矿' })
  expect(within(gold).queryByText('216.67')).not.toBeInTheDocument()
  expect(within(gold).getAllByText('–').length).toBeGreaterThan(0)
  expect(within(screen.getByRole('link', { name: '红宝石' })).getByText('3.00')).toBeInTheDocument()
})

it('every item gets a card; one without trading data shows dashes and comes after those with data', () => {
  const noTrade = { ...GROUPS[1], key: 'id.item.cobalt', archetype: 'id.item.cobalt', name: 'Cobalt Ore', name_zh: '钴矿', season_rows: 0,
    variants: GROUPS[1].variants.map(x => ({ ...x, name: 'Cobalt Ore', season_rows: 0, latest: null, change24: null, season_id: null })) }
  mount([], [noTrade, GROUPS[1]])
  const cobalt = screen.getByRole('link', { name: '钴矿' })
  expect(within(cobalt).getAllByText('–').length).toBeGreaterThan(0)
  const names = screen.getAllByRole('link').map(a => a.getAttribute('aria-label')).filter(Boolean)
  expect(names.indexOf('金矿')).toBeLessThan(names.indexOf('钴矿'))
  expect(screen.queryByText(/还没有交易数据的道具/)).not.toBeInTheDocument()
})

it('a filter beside the search shows all items, only those with trading data, or only those without', () => {
  const noTrade = { ...GROUPS[1], key: 'id.item.cobalt', archetype: 'id.item.cobalt', name: 'Cobalt Ore', name_zh: '钴矿', season_rows: 0,
    variants: GROUPS[1].variants.map(x => ({ ...x, name: 'Cobalt Ore', season_rows: 0, latest: null, change24: null, season_id: null })) }
  mount([], [GROUPS[0], GROUPS[1], noTrade])
  const filter = screen.getByRole('combobox', { name: '筛选' })
  expect([...(filter as HTMLSelectElement).options].map(o => o.textContent)).toEqual(['全部（3）', '有交易数据（2）', '没有数据（1）'])
  fireEvent.change(filter, { target: { value: 'data' } })
  expect(screen.queryByRole('link', { name: '钴矿' })).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: '金矿' })).toBeInTheDocument()
  fireEvent.change(filter, { target: { value: 'nodata' } })
  expect(screen.getByRole('link', { name: '钴矿' })).toBeInTheDocument()
  expect(screen.queryByRole('link', { name: '金矿' })).not.toBeInTheDocument()
})
