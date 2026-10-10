import { act, render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import * as stream from '../../api/jobStream'
import { SetupWizard } from './SetupWizard'

vi.mock('../../api/hooks')
vi.mock('../../api/jobStream')
beforeEach(() => sessionStorage.clear())

const base = { initialized: false, collect_interval_hours: 6, auto_collect: false, api_key_configured: true, api_key_hint: '…by2r',
  data_source: 'csv', seasons: 8, tracked_items: 4, catalog_items: 2432,
  environment: { python_version: '3.13.5', python_ok: true, storage: 'csv', data_dir: '/x/data', db_host: null, db_database: null } }

function mount(setup = base) {
  const init = { mutate: vi.fn(), isPending: false }
  const schedule = { mutate: vi.fn(), isPending: false }
  vi.mocked(hooks.useSetup).mockReturnValue({ data: setup, isSuccess: true } as never)
  vi.mocked(hooks.useSaveApiKey).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useDatabaseSetup).mockReturnValue({ save: { mutate: vi.fn(), isPending: false }, clear: { mutate: vi.fn(), isPending: false } } as never)
  vi.mocked(hooks.useCatalogJob).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Iron Ore' }] } as never)
  vi.mocked(hooks.useCatalog).mockReturnValue({ data: { rows: [
    { archetype: 'id.item.iron_ores', name: 'Iron Ore', name_zh: '铁矿', has_stats: false, rarities: ['common'], listings_total: 5, variants: [{ id: 'id.item.iron_ores_2001', rarity: 'common', raw: {} }] },
    { archetype: 'id.item.bandage', name: 'Bandage', name_zh: '绷带', has_stats: false, rarities: ['poor', 'common'], listings_total: 9, variants: [{ id: 'id.item.bandage_1001', rarity: 'poor', raw: {} }, { id: 'id.item.bandage_2001', rarity: 'common', raw: {} }] },
  ], total: 2, facets: { item_type: [] } } } as never)
  vi.mocked(hooks.useInitialize).mockReturnValue(init as never)
  vi.mocked(hooks.useSaveSchedule).mockReturnValue(schedule as never)
  vi.mocked(stream.useJobStream).mockReturnValue({ progress: null, status: null, disconnected: false, error: null, resultRef: null })
  render(<MemoryRouter><SetupWizard /></MemoryRouter>)
  return { init, schedule }
}

it('opens on first run with four ordered steps and checks the environment', () => {
  mount()
  const dialog = screen.getByRole('dialog')
  expect(dialog).toHaveTextContent('1 环境')
  expect(dialog).toHaveTextContent('4 自动采集')
  expect(screen.getByText(/Python 3.13.5/)).toBeInTheDocument()
  expect(screen.getByLabelText('本机文件（推荐）')).toBeChecked()
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
  fireEvent.change(screen.getByLabelText('多久采集一次'), { target: { value: '3' } })
  fireEvent.click(screen.getByRole('button', { name: '完成' }))
  expect(schedule.mutate.mock.calls[0][0]).toEqual({ interval_hours: 3, auto_collect: true })
})
