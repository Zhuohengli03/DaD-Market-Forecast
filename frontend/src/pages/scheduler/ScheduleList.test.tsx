import { render, screen, fireEvent, within } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { ScheduleList } from './ScheduleList'

vi.mock('../../api/hooks')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

const v = (name: string, label: string, rarity: string) => ({ name, label, label_en: label, rarity, rows: 1, season_rows: 1, last_ts: null, latest: 1, change24: null, spark: [] })
const GROUPS = [
  { key: 'id.item.ruby', archetype: 'id.item.ruby', name: 'Ruby', name_zh: '红宝石', category: 'gem', icon_url: null, rows: 3, season_rows: 3,
    variants: [v('Ruby (Cracked) (poor)', '裂开', 'poor'), v('Ruby (Flawed) (common)', '瑕疵', 'common'), v('Ruby (Normal) (uncommon)', '普通', 'uncommon')] },
  { key: 'id.item.gold_ores', archetype: 'id.item.gold_ores', name: 'Gold Ore', name_zh: '金矿', category: 'ore', icon_url: null, rows: 1, season_rows: 1, variants: [v('Gold Ore', '史诗', 'epic')] },
]

function mount(scheduledOff: string[] = []) {
  const tick = vi.fn()
  vi.mocked(hooks.useOverview).mockReturnValue({ data: GROUPS, isLoading: false } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: GROUPS.flatMap(g => g.variants.map(x => ({ name: x.name, rows: 1, scheduled: !scheduledOff.includes(x.name) }))) } as never)
  vi.mocked(hooks.useScheduleEstimate).mockReturnValue({ data: { per_month: 0, unknown: 0, per_item: {} } } as never)
  vi.mocked(hooks.useSetScheduled).mockReturnValue({ mutate: tick, isPending: false } as never)
  for (const h of ['useStartCollect', 'useCollectAll', 'useSetItemSchedule', 'useSetScheduleBulk'] as const) vi.mocked(hooks[h]).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<ScheduleList defaultSeconds={21600} onJob={() => {}} />)
  return tick
}

it('shows one row per item type; ticking the type ticks every quality', () => {
  const tick = mount(['Ruby (Cracked) (poor)', 'Ruby (Flawed) (common)', 'Ruby (Normal) (uncommon)'])
  expect(screen.queryByText('裂开')).not.toBeInTheDocument()            // qualities stay folded
  fireEvent.click(screen.getByLabelText('定时采集 红宝石'))
  expect(tick.mock.calls[0][0]).toEqual({ names: ['Ruby (Cracked) (poor)', 'Ruby (Flawed) (common)', 'Ruby (Normal) (uncommon)'], scheduled: true })
})

it('opens a type to tick single qualities; a partly ticked type shows as mixed', () => {
  const tick = mount(['Ruby (Flawed) (common)'])
  const parent = screen.getByLabelText('定时采集 红宝石') as HTMLInputElement
  expect(parent.checked).toBe(false)
  expect(parent.indeterminate).toBe(true)
  expect(screen.getByText('2/3 个品质')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /红宝石/ }))
  fireEvent.click(screen.getByLabelText('定时采集 瑕疵'))
  expect(tick.mock.calls[0][0]).toEqual({ names: ['Ruby (Flawed) (common)'], scheduled: true })
})

it('counts ticked items and can untick everything at once', () => {
  const tick = mount(['Gold Ore'])
  expect(screen.getByText('已勾选 3 / 4 个')).toBeInTheDocument()
  expect(within(screen.getByText('金矿').closest('div')!.parentElement!).getByText('不定时采集')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '全部取消' }))
  expect(tick.mock.calls[0][0].scheduled).toBe(false)
  expect(tick.mock.calls[0][0].names).toHaveLength(4)
})
