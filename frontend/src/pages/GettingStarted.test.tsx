import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import { GettingStarted } from './GettingStarted'

vi.mock('../api/hooks')

function setup(setup: object, items: object[]) {
  vi.mocked(hooks.useSetup).mockReturnValue({ data: setup, isSuccess: true } as never)
  vi.mocked(hooks.useItems).mockReturnValue({ data: items, isSuccess: true } as never)
  return render(<MemoryRouter><GettingStarted /></MemoryRouter>)
}

it('lists the steps in order and links the next unfinished one', () => {
  setup({ api_key_configured: true, seasons: 0, tracked_items: 0 }, [])
  const steps = screen.getAllByRole('listitem').map(li => li.textContent)
  expect(steps[0]).toMatch(/填写 DarkerDB API 密钥/)
  expect(steps[1]).toMatch(/导入赛季/)
  expect(steps[2]).toMatch(/选择要追踪的道具/)
  expect(steps[3]).toMatch(/采集数据/)
  expect(screen.getByRole('link', { name: '去导入赛季' })).toHaveAttribute('href', '/settings')
})

it('disappears once everything is ready', () => {
  const { container } = setup({ api_key_configured: true, seasons: 8, tracked_items: 2 }, [{ name: 'Iron Ore', rows: 100 }])
  expect(container).toBeEmptyDOMElement()
})
