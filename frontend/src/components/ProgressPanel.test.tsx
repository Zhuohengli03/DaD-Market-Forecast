import { render, screen } from '@testing-library/react'
import '../i18n'
import { ProgressPanel } from './ProgressPanel'

it('shows forecast backtest progress as a bar with a plain sentence', () => {
  render(<ProgressPanel kind="forecast" status="running" progress={{ stage: 'backtest', done: 3, total: 7, detail: { model: 'ridge' } }} error={null} disconnected={false} />)
  expect(screen.getByText('正在用历史数据检验各方法（3/7：线性回归）')).toBeInTheDocument()
  expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '43')
})

it('summarises a finished collection', () => {
  render(<ProgressPanel kind="collect" status="succeeded" progress={{ stage: 'done', done: null, total: null, detail: { new: 120 } }} error={null} disconnected={false} />)
  expect(screen.getByText('采集完成：新增 120 条挂单')).toBeInTheDocument()
})

it('explains failures without technical output', () => {
  render(<ProgressPanel kind="collect" status="failed" progress={{ stage: 'error', done: null, total: null, detail: { code: 'auth' } }} error="exit code 2" disconnected={false} />)
  expect(screen.getByRole('alert')).toHaveTextContent('DarkerDB 拒绝了你的 API key')
  expect(screen.queryByText('exit code 2')).not.toBeInTheDocument()
})

it('shows a waiting state before progress arrives', () => {
  render(<ProgressPanel kind="catalog" status="queued" progress={null} error={null} disconnected={false} />)
  expect(screen.getByText('准备中…')).toBeInTheDocument()
})
