import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { RelatedUpdates } from './RelatedUpdates'

vi.mock('../../api/hooks')

const M = [
  { id: '2', date: '2026-02-26T00:00:00Z', title: 'Early Access Hotfix #109', url: 'https://steam/2', snippet: 'Gold Ore now has a chance to appear in the Mysterious Rooms.', change3d: 0.18 },
  { id: '1', date: '2024-09-24T00:00:00Z', title: 'Early Access Hotfix #65', url: 'https://steam/1', snippet: 'Fixed an issue where the material of gold ore could not be displayed.', change3d: null },
]

it('lists the updates that name the item, with the name highlighted, a link and the price change around it', () => {
  vi.mocked(hooks.useItemAnnouncements).mockReturnValue({ data: { name: 'Gold Ore', mentions: M }, isLoading: false } as never)
  vi.mocked(hooks.useAnnouncements).mockReturnValue({ data: { items: [], error: null } } as never)
  const onPick = vi.fn()
  render(<RelatedUpdates archetype="id.item.gold_ores" item="Gold Ore" onPick={onPick} />)
  expect(screen.getByText('抢先体验热修复 #109')).toBeInTheDocument()
  expect(screen.getAllByText(/gold ore/i, { selector: 'mark' })).toHaveLength(2)
  expect(screen.getByText('+18%')).toBeInTheDocument()
  expect(screen.getByText('–')).toBeInTheDocument()
  expect(screen.getAllByRole('link', { name: 'Steam' })[0]).toHaveAttribute('href', 'https://steam/2')
  fireEvent.click(screen.getByText('抢先体验热修复 #109'))
  expect(onPick).toHaveBeenCalledWith('2026-02-26T00:00:00Z')
})

it('says when no update names the item, and when updates could not be fetched', () => {
  vi.mocked(hooks.useItemAnnouncements).mockReturnValue({ data: { name: 'Gold Ore', mentions: [] }, isLoading: false } as never)
  vi.mocked(hooks.useAnnouncements).mockReturnValue({ data: { items: [], error: '公告暂时无法更新（ConnectionError）' } } as never)
  render(<RelatedUpdates archetype="id.item.gold_ores" item="Gold Ore" onPick={vi.fn()} />)
  expect(screen.getByText(/没有补丁直接提到/)).toBeInTheDocument()
  expect(screen.getByText(/公告暂时无法更新/)).toBeInTheDocument()
})
