import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { Shell } from './Shell'

vi.mock('../../api/hooks')

it('orders navigation by workflow: get data first, then analyse, settings last', () => {
  vi.mocked(hooks.useStatus).mockReturnValue({ data: { source: 'csv', jobs_running: [] } } as never)
  render(<MemoryRouter><Shell /></MemoryRouter>)
  const nav = screen.getByRole('navigation')
  const links = within(nav).getAllByRole('link').map(a => a.textContent)
  expect(links).toEqual(['概览', '道具库', '数据采集', '定时采集', '价格走势', '价格预测', '装备行情', '设置'])
  expect(within(nav).getByText('① 获取数据')).toBeInTheDocument()
  expect(within(nav).getByText('② 查看与分析')).toBeInTheDocument()
})
