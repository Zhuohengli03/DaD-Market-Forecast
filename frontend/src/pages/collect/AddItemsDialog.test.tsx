import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { AddItemsDialog } from './AddItemsDialog'

vi.mock('../../api/hooks')

// cost estimates and credits load in the background; tests that care set their own values
beforeEach(() => {
  for (const h of ['useScheduleEstimate', 'useItemsEstimate', 'useQuota', 'useCollectAllEstimate'] as const) vi.mocked(hooks[h]).mockReturnValue({ data: undefined } as never)
})
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

const row = (arch: string, name: string, zh: string, category: string) => ({ archetype: arch, name, name_zh: zh, category, item_type: null, icon_url: null, has_stats: false, rarities: [], listings_total: 1, variants: [{ id: `${arch}_1001`, rarity: null, raw: {} }] })

it('lists only untracked items and adds a whole category at once', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', item_id: 'id.item.gold_ore_1001' }] } as never)
  vi.mocked(hooks.useTrackBulk).mockReturnValue({ mutate, isPending: false } as never)
  vi.mocked(hooks.useCatalog).mockReturnValue({ isLoading: false, data: { rows: [
    row('id.item.gold_ore', 'Gold Ore', '金矿石', 'ore'), row('id.item.iron_ore', 'Iron Ore', '铁矿石', 'ore'),
    row('id.item.cobalt_ore', 'Cobalt Ore', '钴矿石', 'ore'), row('id.item.bandage', 'Bandage', '绷带', 'consumable')] } } as never)
  render(<AddItemsDialog onClose={() => {}} />)
  expect(screen.queryByLabelText('金矿石')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /全选矿石/ }))
  fireEvent.click(screen.getByRole('button', { name: '添加 2 个' }))
  expect(mutate.mock.calls[0][0]).toEqual(['id.item.iron_ore_1001', 'id.item.cobalt_ore_1001'])
})
