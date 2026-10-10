import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './i18n'
import { Shell } from './components/layout/Shell'
import { ToastProvider } from './components/ui/Toast'
import Dashboard from './pages/Dashboard'
import Market from './pages/Market'
import Forecast from './pages/Forecast'
import Collect from './pages/Collect'
import Scheduler from './pages/Scheduler'
import Settings from './pages/Settings'
import Catalog from './pages/Catalog'
import Equipment from './pages/Equipment'
import { ApiKeyDialog } from './components/ApiKeyDialog'

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 10_000 } } })

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <BrowserRouter>
          <ApiKeyDialog />
          <Routes>
            <Route element={<Shell />}>
              <Route index element={<Dashboard />} />
              <Route path="/market" element={<Market />} />
              <Route path="/forecast" element={<Forecast />} />
              <Route path="/catalog" element={<Catalog />} />
              <Route path="/equipment" element={<Equipment />} />
              <Route path="/collect" element={<Collect />} />
              <Route path="/scheduler" element={<Scheduler />} />
              <Route path="/settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  )
}
