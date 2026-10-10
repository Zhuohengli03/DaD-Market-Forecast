import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { NetworkCard } from './NetworkCard'

vi.mock('../../api/hooks')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

function mount(data: object) {
  const save = vi.fn(); const restart = vi.fn()
  vi.mocked(hooks.useNetwork).mockReturnValue({ data } as never)
  vi.mocked(hooks.useSaveNetwork).mockReturnValue({ mutate: save, isPending: false } as never)
  vi.mocked(hooks.useRestart).mockReturnValue({ mutate: restart, isPending: false } as never)
  render(<NetworkCard />)
  return { save, restart }
}

it('turns on LAN access and changes the port without touching any file', () => {
  const { save } = mount({ lan: false, port: 5001, running_port: 5001, running_lan: false, restart_required: false, lan_urls: [] })
  fireEvent.click(screen.getByLabelText(/允许同一局域网/))
  expect(screen.getByText(/只在家里等可信的网络下打开/)).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('端口'), { target: { value: '5077' } })
  fireEvent.click(screen.getByRole('button', { name: '保存' }))
  expect(save.mock.calls[0][0]).toEqual({ lan: true, port: 5077 })
})

it('rejects a port outside 1024–65535', () => {
  mount({ lan: false, port: 5001, running_port: 5001, running_lan: false, restart_required: false, lan_urls: [] })
  fireEvent.change(screen.getByLabelText('端口'), { target: { value: '80' } })
  expect(screen.getByRole('alert')).toHaveTextContent('1024 到 65535')
  expect(screen.getByRole('button', { name: '保存' })).toBeDisabled()
})

it('offers a restart when saved settings are not running yet', () => {
  const { restart } = mount({ lan: true, port: 5077, running_port: 5001, running_lan: false, restart_required: true, lan_urls: ['http://192.168.1.5:5077'] })
  fireEvent.click(screen.getByRole('button', { name: '立即重启' }))
  expect(restart).toHaveBeenCalled()
})
