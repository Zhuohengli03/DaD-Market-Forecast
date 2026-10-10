import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { ScheduleForm, splitSeconds, requestsPerMinute } from './ScheduleForm'

vi.mock('../../api/hooks')

it('splits seconds into days/hours/minutes/seconds', () => {
  expect(splitSeconds(93784)).toEqual({ d: 1, h: 2, m: 3, s: 4 })
  expect(requestsPerMinute(30, 10)).toBe(20)
})

it('refuses intervals under 30 seconds', () => {
  const save = { mutate: vi.fn(), isPending: false }
  vi.mocked(hooks.useSaveSchedule).mockReturnValue(save as never)
  render(<ScheduleForm initialSeconds={3600} initialAuto trackedItems={2} rateLimit={60} submitLabel="保存" />)
  fireEvent.change(screen.getByLabelText('时'), { target: { value: '0' } })
  fireEvent.change(screen.getByLabelText('秒'), { target: { value: '20' } })
  expect(screen.getByRole('alert')).toHaveTextContent('最少 30 秒')
  expect(screen.getByRole('button', { name: '保存' })).toBeDisabled()
})

it('always reminds about the DarkerDB plan and warns when the estimate exceeds the rate limit', () => {
  vi.mocked(hooks.useSaveSchedule).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<ScheduleForm initialSeconds={30} initialAuto trackedItems={40} rateLimit={60} submitLabel="保存" />)
  expect(screen.getByText(/请按你的 DarkerDB API 订阅方案设置/)).toBeInTheDocument()
  expect(screen.getByText(/每分钟至少 80 次请求/)).toBeInTheDocument()
  expect(screen.getByText(/超过了每分钟 60 次的额度/)).toBeInTheDocument()
})
