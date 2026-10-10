import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { ListingsTable } from './ListingsTable'

vi.mock('../../api/hooks')
const wrap = (ui: React.ReactElement) => render(<QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>)

it('renders empty state when no listings match', () => {
  vi.mocked(hooks.useListings).mockReturnValue({ data: { rows: [], total: 0 }, isLoading: false } as never)
  wrap(<ListingsTable item="Gold Ore" onPick={() => {}} />)
  expect(screen.getByText('没有符合条件的挂单')).toBeInTheDocument()
})

it('renders rows and pagination summary', () => {
  vi.mocked(hooks.useListings).mockReturnValue({ data: { rows: [{ item_id: '1', item: 'Gold Ore', quantity: 6, price_per_unit: 20.5, price: 123, has_sold: true, created_at: '2026-01-29T12:00:00+00:00' }], total: 250 }, isLoading: false } as never)
  wrap(<ListingsTable item="Gold Ore" onPick={() => {}} />)
  expect(screen.getByText('20.50')).toBeInTheDocument()
  expect(screen.getByText(/第 1 \/ 5 页，共 250 条/)).toBeInTheDocument()
})

it('returns to page 1 when the item changes', () => {
  vi.mocked(hooks.useListings).mockReturnValue({ data: { rows: [], total: 500 }, isLoading: false } as never)
  const { rerender } = wrap(<ListingsTable item="Gold Ore" onPick={() => {}} />)
  screen.getByRole('button', { name: '下一页' }).click()
  rerender(<QueryClientProvider client={new QueryClient()}><ListingsTable item="Iron Ore" onPick={() => {}} /></QueryClientProvider>)
  expect(vi.mocked(hooks.useListings).mock.calls.at(-1)?.[1]).toMatchObject({ page: 1 })
})
