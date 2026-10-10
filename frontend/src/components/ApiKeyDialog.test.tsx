import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../i18n'
import * as hooks from '../api/hooks'
import { ApiError } from '../api/client'
import { ApiKeyDialog } from './ApiKeyDialog'

vi.mock('../api/hooks')
beforeEach(() => sessionStorage.clear())

function setup(configured: boolean, mutate = vi.fn()) {
  vi.mocked(hooks.useSetup).mockReturnValue({ data: { api_key_configured: configured, api_key_hint: null, data_source: 'csv', seasons: 0, tracked_items: 4, catalog_items: 0 }, isSuccess: true } as never)
  vi.mocked(hooks.useSaveApiKey).mockReturnValue({ mutate, isPending: false } as never)
  return render(<ApiKeyDialog />)
}

it('opens on first run when no key is configured', () => {
  setup(false)
  expect(screen.getByRole('dialog')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /darkerdb\.com/ })).toHaveAttribute('href', 'https://darkerdb.com/dashboard/api-keys')
})

it('stays closed when a key exists', () => {
  setup(true)
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

it('submits the trimmed key and shows the server error inline', () => {
  const mutate = vi.fn((_k: string, o: { onError: (e: Error) => void }) => o.onError(new ApiError(422, 'DarkerDB 拒绝了这个 key')))
  setup(false, mutate)
  fireEvent.change(screen.getByLabelText('API key'), { target: { value: '  abc  ' } })
  fireEvent.click(screen.getByRole('button', { name: '验证并保存' }))
  expect(mutate.mock.calls[0][0]).toBe('abc')
  expect(screen.getByRole('alert')).toHaveTextContent('DarkerDB 拒绝了这个 key')
})

it('focuses the key field and closes on Escape', () => {
  setup(false)
  expect(screen.getByLabelText('API key')).toHaveFocus()
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

it('can be dismissed for now', () => {
  setup(false)
  fireEvent.click(screen.getByRole('button', { name: '稍后再说' }))
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})
