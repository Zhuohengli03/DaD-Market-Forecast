import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { CatalogRedirect } from './App'

function Where() { return <p>at:{useLocation().pathname}</p> }

it('old item links open the item page', () => {
  render(<MemoryRouter initialEntries={['/catalog/id.item.ruby']}><Routes>
    <Route path="/catalog/:archetype" element={<CatalogRedirect />} />
    <Route path="/item/:archetype" element={<Where />} />
  </Routes></MemoryRouter>)
  expect(screen.getByText('at:/item/id.item.ruby')).toBeInTheDocument()
})
