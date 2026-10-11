import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { DefaultInterval } from './DefaultInterval'

vi.mock('../../api/hooks')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

function mount(running: boolean) {
  const save = vi.fn()
  vi.mocked(hooks.useSaveSchedule).mockReturnValue({ mutate: save, isPending: false } as never)
  vi.mocked(hooks.useScheduleEstimate).mockReturnValue({ data: { per_month: 24000, unknown: 0, per_item: {} } } as never)
  render(<DefaultInterval seconds={21600} running={running} monthlyQuota={10000} />)
  return save
}

it('saves a new default interval without turning scheduled collection on or off', () => {
  const save = mount(false)
  expect(screen.getByRole('button', { name: '保存' })).toBeDisabled()          // unchanged
  fireEvent.change(screen.getByLabelText('时'), { target: { value: '3' } })
  expect(screen.getByText(/按这个间隔，定时采集每月约 24,000 次/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '保存' }))
  expect(save.mock.calls[0][0]).toEqual({ interval_seconds: 10800, auto_collect: false })
})

it('refuses intervals under 30 seconds', () => {
  mount(true)
  for (const [label, v] of [['时', '0'], ['秒', '10']]) fireEvent.change(screen.getByLabelText(label), { target: { value: v } })
  expect(screen.getByRole('alert')).toHaveTextContent('30 秒')
  expect(screen.getByRole('button', { name: '保存' })).toBeDisabled()
})
