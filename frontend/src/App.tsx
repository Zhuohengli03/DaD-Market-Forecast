import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes, useParams } from 'react-router-dom'
import './i18n'
import { Shell } from './components/layout/Shell'
import { ToastProvider } from './components/ui/Toast'
import Dashboard from './pages/Dashboard'
import Market from './pages/Market'
import ForecastList from './pages/ForecastList'
import CollectManage, { SchedulerRedirect } from './pages/CollectManage'
import Database from './pages/Database'
import Settings from './pages/Settings'
import Catalog from './pages/Catalog'
import ItemPage from './pages/item/ItemPage'
import Equipment from './pages/Equipment'
import Reports from './pages/Reports'
import Holdout from './pages/Holdout'
import { SetupWizard } from './components/wizard/SetupWizard'

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 10_000 } } })

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <BrowserRouter>
          <SetupWizard />
          <Routes>
            <Route element={<Shell />}>
              <Route index element={<Dashboard />} />
              <Route path="/market" element={<Market />} />
              <Route path="/forecast" element={<ForecastList />} />
              <Route path="/holdout" element={<Holdout />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/catalog" element={<Catalog />} />
              <Route path="/item/:archetype" element={<ItemPage />} />
              <Route path="/catalog/:archetype" element={<CatalogRedirect />} />
              <Route path="/equipment" element={<Equipment />} />
              <Route path="/collect" element={<CollectManage />} />
              <Route path="/scheduler" element={<SchedulerRedirect />} />
              <Route path="/database" element={<Database />} />
              <Route path="/settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  )
}

/** Old item links (/catalog/<type>) open the item page. */
export function CatalogRedirect() {
  const { archetype = '' } = useParams()
  return <Navigate replace to={`/item/${encodeURIComponent(archetype)}`} />
}
