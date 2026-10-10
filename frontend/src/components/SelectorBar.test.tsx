import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import { SelectorBar } from './SelectorBar'

vi.mock('../api/hooks')

function Where() { return <p data-testid="url">{useLocation().search}</p> }

function mount(url = '/market') {
  vi.mocked(hooks.useItems).mockReturnValue({ data: [{ name: 'Gold Ore', name_zh: '金矿' }] } as never)
  vi.mocked(hooks.useSeasons).mockReturnValue({ data: [
    { id: 's7', name: '第 7 赛季', start: '2025-01-01T00:00:00Z', end: '2025-03-01T00:00:00Z' },
    { id: 's11', name: 'Season 11', start: '2026-10-09T00:00:00Z', end: null }] } as never)
  render(<MemoryRouter initialEntries={[url]}><SelectorBar /><Where /></MemoryRouter>)
}

it('switches to another season', () => {
  mount()
  const select = screen.getByLabelText('赛季') as HTMLSelectElement
  expect(select.value).toBe('s11')                                  // latest season by default
  fireEvent.change(select, { target: { value: 's7' } })
  expect(select.value).toBe('s7')
  expect(screen.getByTestId('url').textContent).toContain('season=s7')
  expect(screen.getByRole('option', { name: '第 11 赛季' })).toBeInTheDocument()   // old English names translated
})

it('can pick a custom time range instead of a season', () => {
  mount('/market?season=s7')
  const select = screen.getByLabelText('赛季') as HTMLSelectElement
  fireEvent.change(select, { target: { value: '' } })
  expect(select.value).toBe('')
  expect(screen.getByText(/开始/)).toBeInTheDocument()
})
