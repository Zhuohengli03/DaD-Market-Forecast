import { render, screen } from '@testing-library/react'
import '../../i18n'
import { PriceNote } from './PriceNote'

const v = { latest: 148, latest_ts: '2026-10-09T02:00:00+00:00', latest_listings: 51 }

it('says what the price is, which hour it is from and how many listings it comes from', () => {
  render(<PriceNote v={v} />)
  expect(screen.getByText('每个的成交中位价')).toBeInTheDocument()
  expect(screen.getByText('2026-10-09 02:00Z 那一小时 · 51 条挂单')).toBeInTheDocument()
  expect(screen.getByText('每个的成交中位价').closest('[title]')).toHaveAttribute('title', expect.stringContaining('中间那条'))
})

it('says nothing when there is no price', () => {
  const { container } = render(<PriceNote v={{ latest: null, latest_ts: null, latest_listings: null }} />)
  expect(container).toBeEmptyDOMElement()
})
