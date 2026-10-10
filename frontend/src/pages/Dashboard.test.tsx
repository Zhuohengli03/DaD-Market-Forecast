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
  ({ name, label, label_en: label, rarity, rows, season_rows: rows, last_ts: rows ? '2026-10-09T10:00:00Z' : null, latest, change24, spark })
const GROUPS = [
  { key: 'id.item.ruby', archetype: 'id.item.ruby', name: 'Ruby', name_zh: '红宝石', category: 'gem', icon_url: null, rows: 60, season_rows: 60,
    variants: [v('Ruby (Cracked) (poor)', '裂开', 'poor', 1.5, 0.2), v('Ruby (Flawed) (common)', '瑕疵', 'common', 3, -0.5)] },
  { key: 'id.item.gold_ores', archetype: 'id.item.gold_ores', name: 'Gold Ore', name_zh: '金矿', category: 'ore', icon_url: null, rows: 30, season_rows: 30,
    variants: [v('Gold Ore', '史诗', 'epic', 216.67, null)] },
]

function mount(items: object[] = []) {
  vi.mocked(hooks.useOverview).mockReturnValue({ data: GROUPS, isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: items, isLoading: false } as never)
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'sqlite', jobs_running: [], collecting: {} } } as never)
  vi.mocked(hooks.useSetup).mockReturnValue({ isSuccess: false } as never)
  vi.mocked(hooks.useUntrackBulk).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Dashboard /></MemoryRouter></QueryClientProvider>)
}

it('shows one card per item type, with every quality inside', () => {
  mount()
  const ruby = screen.getByRole('link', { name: '红宝石' })
  expect(ruby).toHaveAttribute('href', '/catalog/id.item.ruby')
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
