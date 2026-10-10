import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import i18n from '../../i18n'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { fmtInterval, ItemScheduleEditor, toInput, validate } from './ItemScheduleEditor'

vi.mock('../../api/hooks')
beforeEach(() => { vi.mocked(hooks.useSetScheduleBulk).mockReturnValue({ mutate: vi.fn(), isPending: false } as never) })

it('formats intervals and reads stored schedules', () => {
  expect(fmtInterval(93784, i18n.t)).toBe('1天2时3分4秒')
  expect(fmtInterval(3600, i18n.t)).toBe('1时')
  expect(toInput('23:00')).toEqual({ mode: 'daily', times: ['23:00'] })
  expect(toInput({ mode: 'interval', seconds: 90 })).toMatchObject({ mode: 'interval', seconds: 90 })
  expect(toInput(undefined)).toEqual({ mode: 'default' })
  expect(validate({ mode: 'interval', seconds: 20 })).toBe('tooShort')
  expect(validate({ mode: 'daily', times: [] })).toBe('noTimes')
})

it('saves fixed daily times for one item', () => {
  const mutate = vi.fn()
  vi.mocked(hooks.useSetItemSchedule).mockReturnValue({ mutate, isPending: false } as never)
  render(<ItemScheduleEditor name="Gold Ore" schedule={null} defaultSeconds={3600} onDone={() => {}} />)
  expect(screen.getByText('用默认间隔（每 1时）')).toBeInTheDocument()
  fireEvent.click(screen.getByLabelText('每天固定时间'))
  fireEvent.change(screen.getByLabelText('时间 1'), { target: { value: '21:30' } })
  fireEvent.click(screen.getByRole('button', { name: '保存' }))
  expect(mutate.mock.calls[0][0]).toEqual({ name: 'Gold Ore', mode: 'daily', times: ['20:00', '21:30'] })
})

it('blocks a custom interval under 30 seconds', () => {
  vi.mocked(hooks.useSetItemSchedule).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<ItemScheduleEditor name="Gold Ore" schedule={{ mode: 'interval', seconds: 60 }} defaultSeconds={3600} onDone={() => {}} />)
  fireEvent.change(screen.getByLabelText('分'), { target: { value: '0' } })
  fireEvent.change(screen.getByLabelText('秒'), { target: { value: '10' } })
  expect(screen.getByRole('alert')).toHaveTextContent('间隔最少 30 秒')
  expect(screen.getByRole('button', { name: '保存' })).toBeDisabled()
})


it('applies one schedule to every quality of a type', () => {
  const bulk = vi.fn()
  vi.mocked(hooks.useSetItemSchedule).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  vi.mocked(hooks.useSetScheduleBulk).mockReturnValue({ mutate: bulk, isPending: false } as never)
  render(<ItemScheduleEditor name="Ruby (Cracked) (poor)" names={['Ruby (Cracked) (poor)', 'Ruby (Flawed) (common)']} schedule={null} defaultSeconds={3600} onDone={() => {}} />)
  fireEvent.click(screen.getByLabelText('每天固定时间'))
  fireEvent.click(screen.getByRole('button', { name: '保存' }))
  expect(bulk.mock.calls[0][0]).toEqual({ names: ['Ruby (Cracked) (poor)', 'Ruby (Flawed) (common)'], mode: 'daily', times: ['08:00', '20:00'] })
})
