import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import ForecastList from './ForecastList'

vi.mock('../api/hooks')

const run = (run_id: string, item: string, season_id: string, created_at: string, selected: string, horizon_hours = 24) =>
  ({ run_id, item, season_id, season_name: season_id === 's10' ? 'Season 10' : 'Season 11', created_at, selected, horizon_hours, skill_vs_naive: 0.1, beats_baseline: selected !== 'naive', source: 'sqlite', warnings: 0 })
const RUNS = [
  run('r3', 'Gold Ore', 's10', '2026-10-10T18:00:00Z', 'ensemble'),
  run('r2', 'Gold Ore', 's10', '2026-10-09T18:00:00Z', 'naive', 72),
  run('r1', 'Cobalt Ore', 's10', '2026-10-08T11:00:00Z', 'naive', 72),
  run('r0', 'Iron Ore', 's11', '2026-10-11T01:00:00Z', 'ridge'),
]
const ITEMS = [
  { name: 'Gold Ore', name_zh: '金矿', item_id: 'id.item.gold_ores_5001', icon_url: 'https://x/gold.png', rows: 10 },
  { name: 'Cobalt Ore', name_zh: '钴矿', item_id: 'id.item.cobalt_ores_4001', icon_url: null, rows: 10 },
  { name: 'Iron Ore', name_zh: '铁矿', item_id: 'id.item.iron_ores_2001', icon_url: null, rows: 10 },
]
const SEASONS = [{ id: 's10', name: 'Season 10', start: '2026-07-17T00:00:00Z', end: '2026-10-09T03:00:00Z', note: '' },
  { id: 's11', name: 'Season 11', start: '2026-10-09T03:00:00Z', end: null, note: '' }]

function Where() { const l = useLocation(); return <p>at:{l.pathname}{l.search}</p> }
function mount(url = '/forecast?season=s10', seasons: object[] = SEASONS) {
  vi.mocked(hooks.useRuns).mockReturnValue({ data: RUNS, isSuccess: true } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: ITEMS, isSuccess: true } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: seasons } as never)
  render(<MemoryRouter initialEntries={[url]}><Routes>
    <Route path="/forecast" element={<ForecastList />} /><Route path="/item/:a" element={<Where />} /></Routes></MemoryRouter>)
}

it('lists the items forecast in the chosen season, newest first, with picture, count, latest run and the model used', () => {
  mount()
  const rows = screen.getAllByRole('row').slice(1)
  expect(rows).toHaveLength(2)                                       // Iron Ore was forecast in season 11 only
  expect(within(rows[0]).getByText('金矿')).toBeInTheDocument()
  expect(rows[0].querySelector('img')).toHaveAttribute('src', 'https://x/gold.png')   // decorative: alt=""
  expect(within(rows[0]).getByText('2')).toBeInTheDocument()          // two runs
  expect(within(rows[0]).getByText('多模型加权')).toBeInTheDocument()   // the newest run's model
  expect(within(rows[1]).getByText(/沿用上一小时/)).toBeInTheDocument()
})

it('a row opens that item page on its latest forecast', () => {
  mount()
  fireEvent.click(screen.getByText('金矿'))
  expect(screen.getByText('at:/item/id.item.gold_ores?q=Gold+Ore&season=s10&run=r3')).toBeInTheDocument()
})

it('defaults to the season of the newest forecast; search narrows it down', () => {
  mount('/forecast')
  expect(screen.getByRole('combobox', { name: '赛季' })).toHaveValue('s11')
  expect(screen.getByText('铁矿')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('搜索道具'), { target: { value: '金' } })
  expect(screen.getByText('没有匹配的道具')).toBeInTheDocument()
})

it('says when a season has no forecasts yet', () => {
  mount('/forecast?season=s9', [...SEASONS, { id: 's9', name: 'Season 9', start: '2026-04-28T00:00:00Z', end: '2026-07-17T00:00:00Z', note: '' }])
  expect(screen.queryAllByRole('row')).toHaveLength(0)
  expect(screen.getByText(/第 9 赛季还没有预测过的道具/)).toBeInTheDocument()
})

it('an old link to one forecast still opens it', () => {
  mount('/forecast?run=r1')
  expect(screen.getByText('at:/item/id.item.cobalt_ores?q=Cobalt+Ore&season=s10&run=r1')).toBeInTheDocument()
})
