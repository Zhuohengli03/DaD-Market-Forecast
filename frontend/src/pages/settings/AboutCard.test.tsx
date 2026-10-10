import { render, screen } from '@testing-library/react'
import '../../i18n'
import { AboutCard } from './AboutCard'

it('names the license and links to the source code', () => {
  render(<AboutCard />)
  expect(screen.getByText('本软件开源，许可证为 AGPL-3.0-or-later。')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '源代码（GitHub）' })).toHaveAttribute('href', 'https://github.com/Zhuohengli03/DaD-Market-Forecast')
})
