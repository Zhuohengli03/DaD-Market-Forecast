import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import Database from './Database'

vi.mock('../api/hooks')
vi.mock('../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))
afterEach(() => { vi.restoreAllMocks(); cleanup() })

const ITEMS = [
  { key: 'cobalt_ore', name: 'Cobalt Ore', name_zh: '钴矿石', tracked: true, rows: 3, first: '2026-08-01T00:00:00Z', last: '2026-10-09T02:00:00Z' },
  { key: 'old_ore', name: 'Old Ore', name_zh: null, tracked: false, rows: 1, first: '2025-01-01T00:00:00Z', last: '2025-01-01T00:00:00Z' },
]
const ROWS = [
  { id: 'a1', quantity: 2, price_per_unit: 10.5, price: 21, has_sold: true, created_at: '2026-10-09T02:00:00Z' },
  { id: 'a2', quantity: 1, price_per_unit: 11, price: 11, has_sold: false, created_at: '2026-10-09T01:00:00Z' },
]

function mount(del = vi.fn()) {
  vi.mocked(hooks.useDbItems).mockReturnValue({ data: { items: ITEMS, source: 'sqlite' }, isLoading: false } as never)
  vi.mocked(hooks.useDbRows).mockReturnValue({ data: { rows: ROWS, total: 2 }, isLoading: false } as never)
  vi.mocked(hooks.useDbDelete).mockReturnValue({ mutate: del, isPending: false } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [] } as never)
  render(<MemoryRouter><Database /></MemoryRouter>)
  return del
}

it('lists stored items, untracked ones too, and shows rows without any way to edit them', () => {
  mount()
  expect(screen.getByRole('button', { name: /钴矿石/ })).toBeInTheDocument()            // Chinese name when known
  expect(screen.getByRole('heading', { name: '钴矿石' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Old Ore/ })).toHaveTextContent('未追踪')
  const table = screen.getByRole('table')
  expect(within(table).getByText('10.50')).toBeInTheDocument()
  expect(within(table).queryAllByRole('textbox')).toHaveLength(0)        // read-only
  expect(within(table).queryAllByRole('spinbutton')).toHaveLength(0)
})

it('deletes the ticked rows after confirming', () => {
  const del = mount()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  fireEvent.click(screen.getByRole('checkbox', { name: 'a1' }))
  fireEvent.click(screen.getByRole('button', { name: '删除选中的 1 条' }))
  expect(del.mock.calls[0][0]).toEqual({ key: 'cobalt_ore', ids: ['a1'] })
})

it('deleting everything of an item needs a confirmation and says how many', () => {
  const del = mount()
  const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
  fireEvent.click(screen.getByRole('button', { name: '删除这个道具的全部 3 条' }))
  expect(confirm.mock.calls[0][0]).toContain('3')
  expect(del).not.toHaveBeenCalled()
})
