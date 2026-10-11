import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { vi } from 'vitest'
import '../i18n'
import CollectManage, { SchedulerRedirect } from './CollectManage'

vi.mock('./Collect', () => ({ default: ({ embedded }: { embedded?: boolean }) => <p>collect now{embedded ? ' (embedded)' : ''}</p> }))
vi.mock('./Scheduler', () => ({ default: ({ embedded }: { embedded?: boolean }) => <p>schedule{embedded ? ' (embedded)' : ''}</p> }))

function Where() { const l = useLocation(); return <p>at:{l.pathname}{l.hash}</p> }

it('one page for spending credits on data: collect now on top, scheduled collection below', () => {
  render(<MemoryRouter><CollectManage /></MemoryRouter>)
  expect(screen.getByRole('heading', { level: 1, name: '采集管理' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { level: 2, name: '立即采集' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { level: 2, name: '定时采集' })).toBeInTheDocument()
  expect(screen.getByText('collect now (embedded)')).toBeInTheDocument()
  expect(screen.getByText('schedule (embedded)')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '定时采集' })).toHaveAttribute('href', '#schedule')
})

it('the old scheduled-collection address opens that part of the page', () => {
  render(<MemoryRouter initialEntries={['/scheduler']}><Routes>
    <Route path="/scheduler" element={<SchedulerRedirect />} /><Route path="/collect" element={<Where />} /></Routes></MemoryRouter>)
  expect(screen.getByText('at:/collect#schedule')).toBeInTheDocument()
})
