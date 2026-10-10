import { act, render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import * as stream from '../../api/jobStream'
import { SetupWizard } from './SetupWizard'

vi.mock('../../api/hooks')
vi.mock('../../api/jobStream')
beforeEach(() => sessionStorage.clear())

const base = { initialized: false, collect_interval_seconds: 21600, rate_limit_per_minute: 60, auto_collect: false, api_key_configured: true, api_key_hint: '…by2r',
  data_source: 'csv', seasons: 8, tracked_items: 4, catalog_items: 2432,
  environment: { python_version: '3.13.5', python_ok: true, storage: 'sqlite', data_dir: '/x/data', db_host: null, db_database: null } }

function mount(setup = base) {
  const init = { mutate: vi.fn(), isPending: false }
  const schedule = { mutate: vi.fn(), isPending: false }
  vi.mocked(hooks.useSetup).mockReturnValue({ data: setup, isSuccess: true } as never)
  vi.mocked(hooks.useSaveApiKey).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useDatabaseSetup).mockReturnValue({ save: { mutate: vi.fn(), isPending: false }, clear: { mutate: vi.fn(), isPending: false } } as never)
  vi.mocked(hooks.useCatalogJob).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Iron Ore' }] } as never)
  vi.mocked(hooks.useCatalog).mockReturnValue({ data: { rows: [
    { archetype: 'id.item.iron_ores', name: 'Iron Ore', name_zh: '铁矿', category: 'ore', has_stats: false, rarities: ['common'], listings_total: 5, variants: [{ id: 'id.item.iron_ores_2001', rarity: 'common', raw: {} }] },
    { archetype: 'id.item.gold_ores', name: 'Gold Ore', name_zh: '金矿', category: 'ore', has_stats: false, rarities: ['epic'], listings_total: 7, variants: [{ id: 'id.item.gold_ores_5001', rarity: 'epic', raw: {} }] },
    { archetype: 'id.item.bandage', name: 'Bandage', name_zh: '绷带', category: 'consumable', has_stats: false, rarities: ['poor', 'common'], listings_total: 9, variants: [{ id: 'id.item.bandage_1001', rarity: 'poor', raw: {} }, { id: 'id.item.bandage_2001', rarity: 'common', raw: {} }] },
  ], total: 2, facets: { item_type: [] } } } as never)
  vi.mocked(hooks.useInitialize).mockReturnValue(init as never)
  vi.mocked(hooks.useSaveSchedule).mockReturnValue(schedule as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, disconnected: false, error: null, resultRef: null })
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><SetupWizard /></MemoryRouter></QueryClientProvider>)
  return { init, schedule }
}

it('opens on first run with four ordered steps and checks the environment', () => {
  mount()
  const dialog = screen.getByRole('dialog')
  expect(dialog).toHaveTextContent('1 环境')
  expect(dialog).toHaveTextContent('4 自动采集')
  expect(screen.getByText(/Python 3.13.5/)).toBeInTheDocument()
  expect(screen.getByLabelText('内置数据库（推荐）')).toBeChecked()
  expect(screen.getByText(/自动创建，不需要安装任何数据库/)).toBeInTheDocument()
})

it('does not open once initialized', () => {
  mount({ ...base, initialized: true })
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

it('walks to initialization and starts it with the chosen items', () => {
  const { init } = mount()
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))       // environment -> key
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))       // key already set -> init
  expect(screen.getByLabelText('铁矿')).toBeChecked()                     // already tracked: preselected
  fireEvent.click(screen.getByLabelText('绷带（普通）'))
  fireEvent.click(screen.getByRole('button', { name: '开始初始化' }))
  expect(init.mutate.mock.calls[0][0]).toEqual({ item_ids: ['id.item.iron_ores_2001', 'id.item.bandage_2001'], max_pages: 200 })
})

it('shows per-item results while initializing', () => {
  mount()
  vi.mocked(stream.useJobStream).mockReturnValue({ status: 'running', disconnected: false, error: null, resultRef: null,
    progress: { stage: 'collect', done: 2, total: 4, detail: { items: [
      { item_id: 'a', name: 'Iron Ore', name_zh: '铁矿', status: 'collected', pages: 3, collected: 120, forecast: null, run_id: null, hours: null },
      { item_id: 'b', name: 'Gold Ore', name_zh: '金矿', status: 'collecting', pages: 2, collected: 40, forecast: null, run_id: null, hours: null }] } } })
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  fireEvent.click(screen.getByRole('button', { name: '开始初始化' }))
  const init = vi.mocked(hooks.useInitialize).mock.results.at(-1)!.value as { mutate: ReturnType<typeof vi.fn> }
  act(() => init.mutate.mock.calls[0][1].onSuccess('job1'))
  expect(screen.getByText('采集到 120 条挂单')).toBeInTheDocument()
  expect(screen.getByText(/正在采集：已看 2 页，新增 40 条/)).toBeInTheDocument()
})

it('saves the collection interval and auto-collect on finish', () => {
  const { schedule } = mount({ ...base, initialized: false })
  for (let i = 0; i < 3; i++) fireEvent.click(screen.getByRole('button', { name: i === 2 ? '跳过初始化' : '下一步' }))
  fireEvent.change(screen.getByLabelText('天'), { target: { value: '0' } })
  fireEvent.change(screen.getByLabelText('时'), { target: { value: '0' } })
  fireEvent.change(screen.getByLabelText('分'), { target: { value: '1' } })
  fireEvent.change(screen.getByLabelText('秒'), { target: { value: '30' } })
  fireEvent.click(screen.getByRole('button', { name: '完成' }))
  expect(schedule.mutate.mock.calls[0][0]).toEqual({ interval_seconds: 90, auto_collect: true })
})

it('continues to the item picker as soon as the catalog download finishes', async () => {
  const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query')
  const qc = new QueryClient()
  const spy = vi.spyOn(qc, 'invalidateQueries')
  vi.mocked(hooks.useSetup).mockReturnValue({ data: { ...base, catalog_items: 0 }, isSuccess: true } as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: { stage: 'done', done: null, total: null, detail: { items: 2432, types: 798 } }, status: 'succeeded', disconnected: false, error: null, resultRef: null })
  render(<QueryClientProvider client={qc}><MemoryRouter><SetupWizard /></MemoryRouter></QueryClientProvider>)
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  expect(screen.getByPlaceholderText('搜索道具')).toBeInTheDocument()
  expect(spy).toHaveBeenCalledWith({ queryKey: ['setup'] })
})

it('refreshes the rest of the app when initialization finishes', async () => {
  const qc = new QueryClient()
  const spy = vi.spyOn(qc, 'invalidateQueries')
  mount()
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: { stage: 'done', done: 4, total: 4, detail: { items: [] } }, status: 'succeeded', disconnected: false, error: null, resultRef: null })
  render(<QueryClientProvider client={qc}><MemoryRouter><SetupWizard /></MemoryRouter></QueryClientProvider>)
  for (const k of ['setup', 'items', 'seasons', 'runs']) expect(spy).toHaveBeenCalledWith({ queryKey: [k] })
})


it('selects whole categories or everything in one click', () => {
  const { init } = mount()
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  fireEvent.click(screen.getByRole('button', { name: '清空' }))
  expect(screen.getByLabelText('铁矿')).not.toBeChecked()
  fireEvent.click(screen.getByRole('button', { name: '全选矿石（2）' }))
  expect(screen.getByLabelText('铁矿')).toBeChecked()
  expect(screen.getByLabelText('金矿')).toBeChecked()
  expect(screen.getByLabelText('绷带（普通）')).not.toBeChecked()
  fireEvent.click(screen.getByRole('button', { name: '全选（4）' }))
  fireEvent.click(screen.getByRole('button', { name: '开始初始化' }))
  expect(init.mutate.mock.calls[0][0].item_ids).toHaveLength(4)
})

it('stays open when initialization marks the app initialized, until the user finishes', () => {
  const qc = new QueryClient()
  vi.mocked(hooks.useSetup).mockReturnValue({ data: base, isSuccess: true } as never)
  mount()
  expect(screen.getAllByRole('dialog')).toHaveLength(1)
  vi.mocked(hooks.useSetup).mockReturnValue({ data: { ...base, initialized: true }, isSuccess: true } as never)
  fireEvent.click(screen.getAllByRole('button', { name: '下一步' })[0])      // any re-render with the new setup state
  expect(screen.getAllByRole('dialog')).toHaveLength(1)
  void qc
})

it('puts the most useful categories first and translates them', () => {
  mount()
  vi.mocked(hooks.useCatalog).mockReturnValue({ data: { rows: [
    { archetype: 'id.item.arrow', name: 'Arrow', category: 'arrow', has_stats: false, rarities: ['common'], listings_total: 1, variants: [{ id: 'id.item.arrow_2001', rarity: 'common', raw: {} }] },
    { archetype: 'id.item.iron_ores', name: 'Iron Ore', name_zh: '铁矿', category: 'ore', has_stats: false, rarities: ['common'], listings_total: 5, variants: [{ id: 'id.item.iron_ores_2001', rarity: 'common', raw: {} }] },
    { archetype: 'id.item.gold_ores', name: 'Gold Ore', name_zh: '金矿', category: 'ore', has_stats: false, rarities: ['epic'], listings_total: 7, variants: [{ id: 'id.item.gold_ores_5001', rarity: 'epic', raw: {} }] },
  ], total: 3, facets: { item_type: [] } } } as never)
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))
  const chips = screen.getAllByRole('button').map(b => b.textContent ?? '').filter(t => t.startsWith('全选') && !/^全选（/.test(t))
  expect(chips).toEqual(['全选矿石（2）', '全选箭（1）'])
})

it('never switches storage on a radio click; asks for confirmation and keeps Postgres settings', () => {
  const pgSetup = { ...base, environment: { ...base.environment, storage: 'postgres', db_host: 'localhost', db_database: 'darkerdb',
    db_saved: { host: 'localhost', port: 5432, database: 'darkerdb', user: 'zhuo' } } }
  mount(pgSetup as never)
  const db = vi.mocked(hooks.useDatabaseSetup).mock.results.at(-1)!.value as { clear: { mutate: ReturnType<typeof vi.fn> } }
  fireEvent.click(screen.getByLabelText('内置数据库（推荐）'))
  expect(db.clear.mutate).not.toHaveBeenCalled()
  expect(screen.getByText(/Postgres 里的数据不会删除/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '改用内置数据库' }))
  expect(db.clear.mutate).toHaveBeenCalled()
})

it('prefills the saved Postgres connection when switching back', () => {
  const paused = { ...base, environment: { ...base.environment, storage: 'sqlite', db_saved: { host: 'db.local', port: 5433, database: 'market', user: 'zhuo' } } }
  mount(paused as never)
  fireEvent.click(screen.getByLabelText('Postgres 数据库'))
  expect(screen.getByLabelText('地址')).toHaveValue('db.local')
  expect(screen.getByLabelText('用户名')).toHaveValue('zhuo')
})
