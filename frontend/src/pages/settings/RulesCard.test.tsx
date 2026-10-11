import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { RulesCard } from './RulesCard'

vi.mock('../../api/hooks')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

const BUNDLED = { source: 'bundled', version: 1, updated: '2026-10-11', synced_at: null, url: '', key_saved: false, key_hint: null, error: null }

function mount(data: object, save = vi.fn(), sync = vi.fn()) {
  vi.mocked(hooks.useRules).mockReturnValue({ data } as never)
  vi.mocked(hooks.useSaveRulesSource).mockReturnValue({ mutate: save, isPending: false } as never)
  vi.mocked(hooks.useSyncRules).mockReturnValue({ mutate: sync, isPending: false } as never)
  render(<RulesCard />)
  return { save, sync }
}

it('says which rules are in use and that without a service they are maintained by hand', () => {
  mount(BUNDLED)
  expect(screen.getByText(/版本 1（2026-10-11）/)).toBeInTheDocument()
  expect(screen.getByText(/随程序附带，自己维护/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '立即同步' })).toBeDisabled()
})

it('saves a rules service with its key, then syncs', () => {
  const { save } = mount(BUNDLED)
  fireEvent.change(screen.getByLabelText('规则服务地址'), { target: { value: 'https://rules.example.com/v1/rules' } })
  fireEvent.change(screen.getByLabelText('授权码'), { target: { value: 'LIC-1' } })
  fireEvent.click(screen.getByRole('button', { name: '保存' }))
  expect(save.mock.calls[0][0]).toEqual({ url: 'https://rules.example.com/v1/rules', key: 'LIC-1' })
})

it('shows a synced source and a failed sync', () => {
  const { sync } = mount({ ...BUNDLED, source: 'synced', version: 3, synced_at: '2026-10-11T03:00:00Z', url: 'https://r.example', key_saved: true, key_hint: '5678', error: '规则暂时无法同步（ConnectionError）' })
  expect(screen.getByText(/已同步/)).toBeInTheDocument()
  expect(screen.getByText(/规则暂时无法同步/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '立即同步' }))
  expect(sync).toHaveBeenCalled()
})
