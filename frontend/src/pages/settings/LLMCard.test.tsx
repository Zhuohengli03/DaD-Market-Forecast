import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import { LLMCard } from './LLMCard'

vi.mock('../../api/hooks')
vi.mock('../../components/ui/Toast', () => ({ useToast: () => ({ push: vi.fn() }) }))

it('saves Bailian settings in the app, keeping the stored key when the field is left empty', () => {
  const save = vi.fn()
  vi.mocked(hooks.useLLM).mockReturnValue({ data: { configured: true, key_hint: '…1234', model: 'qwen-plus', region: 'cn', workspace: '', price_in: null, price_out: null, base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1' } } as never)
  vi.mocked(hooks.useSaveLLM).mockReturnValue({ mutate: save, isPending: false } as never)
  vi.mocked(hooks.useTestLLM).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)
  render(<LLMCard />)
  expect(screen.getByPlaceholderText('已保存（…1234），留空则不修改')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('地域'), { target: { value: 'intl' } })
  fireEvent.change(screen.getByLabelText('输入'), { target: { value: '0.8' } })
  fireEvent.change(screen.getByLabelText('输出'), { target: { value: '2' } })
  fireEvent.click(screen.getByRole('button', { name: '保存' }))
  expect(save.mock.calls[0][0]).toEqual({ model: 'qwen-plus', region: 'intl', workspace: '', price_in: 0.8, price_out: 2 })
})
