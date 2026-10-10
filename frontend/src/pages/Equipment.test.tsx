import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import Equipment from './Equipment'

vi.mock('../api/hooks')
vi.mock('echarts-for-react', () => ({ default: () => null }))

it('lets the user add stat ranges and shows matching listings', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useSetup).mockReturnValue({ data: { api_key_configured: true } } as never)
  vi.mocked(hooks.useCatalog).mockReturnValue({ data: { rows: [{ archetype: 'id.item.arming_sword', name: 'Arming Sword', rarities: ['common', 'legendary'], has_stats: true }], total: 1, facets: { item_type: [] } } } as never)
  vi.mocked(hooks.useEquipmentStats).mockReturnValue({ data: { rarities: ['common', 'legendary'], stats: [
    { key: 'secondary_armor_penetration', kind: 'secondary', name: 'armor_penetration', label: 'Armor Penetration', percent: true, min: 1.5, max: 3, seen: 40 }] } } as never)
  vi.mocked(hooks.useEquipmentQuery).mockReturnValue({ mutate, isPending: false, data: { count: 1, listings: [
    { id: '1', item_id: 'x', name: 'Arming Sword', rarity: 'legendary', quantity: 1, price: 2333, price_per_unit: 2333, has_sold: true, listing_state: 'missing', created_at: '2026-10-09T10:00:00Z', attributes: { secondary_armor_penetration: 2.9 } }],
    daily: [], covered_from: '2026-10-09T10:00:00Z', covered_to: '2026-10-09T10:00:00Z', truncated: false } } as never)
  render(<MemoryRouter initialEntries={['/equipment?type=id.item.arming_sword']}><Equipment /></MemoryRouter>)
  fireEvent.change(screen.getByLabelText('稀有度'), { target: { value: 'legendary' } })
  fireEvent.click(screen.getByRole('button', { name: '添加词条条件' }))
  fireEvent.change(screen.getByLabelText('词条 1'), { target: { value: 'secondary_armor_penetration' } })
  fireEvent.change(screen.getByLabelText('词条 1 最小值'), { target: { value: '2' } })
  fireEvent.click(screen.getByRole('button', { name: '查询' }))
  expect(mutate.mock.calls[0][0]).toMatchObject({ archetype: 'id.item.arming_sword', rarity: 'legendary', filters: [{ kind: 'secondary', name: 'armor_penetration', min: 2, max: null }] })
  expect(screen.getByText('Armor Penetration 2.9%')).toBeInTheDocument()
  expect(screen.getByText('2333.00')).toBeInTheDocument()
})
