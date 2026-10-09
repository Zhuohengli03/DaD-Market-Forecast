import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './i18n'
import { Shell } from './components/layout/Shell'
import { ToastProvider } from './components/ui/Toast'
import Dashboard from './pages/Dashboard'
import Market from './pages/Market'
import Forecast from './pages/Forecast'

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 10_000 } } })

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Shell />}>
              <Route index element={<Dashboard />} />
              <Route path="/market" element={<Market />} />
              <Route path="/forecast" element={<Forecast />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  )
}
