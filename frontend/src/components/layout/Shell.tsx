import { NavLink, Outlet } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Activity, BarChart3, Database, Download, LayoutDashboard, Package, Settings, Clock } from 'lucide-react'
import { useStatus } from '../../api/hooks'
import { setLang } from '../../i18n'
import { Badge } from '../ui/Badge'

const NAV = [
  { to: '/', key: 'dashboard', icon: LayoutDashboard },
  { to: '/market', key: 'market', icon: BarChart3 },
  { to: '/forecast', key: 'forecast', icon: Activity },
  { to: '/catalog', key: 'catalog', icon: Package },
  { to: '/collect', key: 'collect', icon: Download },
  { to: '/scheduler', key: 'scheduler', icon: Clock },
  { to: '/settings', key: 'settings', icon: Settings },
] as const

export function Shell() {
  const { t, i18n } = useTranslation()
  const status = useStatus()
  const running = status.data?.jobs_running.length ?? 0
  return (
    <div className="flex h-full">
      <aside className="w-56 shrink-0 bg-[var(--sidebar)] text-gray-200 flex flex-col">
        <div className="px-4 py-5 border-b border-white/10"><div className="font-semibold text-white">Darker Market</div><div className="text-xs text-gray-400 mt-0.5">{t('app.tagline')}</div><div className="text-xs text-gray-500 mt-1 flex items-center gap-1"><Database size={12} />{status.data ? (status.data.source === 'postgres' ? 'Postgres' : 'CSV') : '…'}</div></div>
        <nav className="flex-1 py-2">
          {NAV.map(n => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => `flex items-center gap-3 px-4 py-2.5 text-sm ${isActive ? 'bg-white/10 text-white' : 'text-gray-300 hover:bg-white/5'}`}>
              <n.icon size={16} />{t(`nav.${n.key}`)}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 border-t border-white/10 flex items-center justify-between text-xs">
          <span>{running > 0 ? <Badge tone="warn">{running} {t('common.running')}</Badge> : <span className="text-gray-500">{t('dashboard.noJobs')}</span>}</span>
          <div className="flex gap-1">
            {(['zh', 'en'] as const).map(l => <button key={l} type="button" onClick={() => setLang(l)} className={`px-1.5 py-0.5 rounded ${i18n.language === l ? 'bg-white/20 text-white' : 'text-gray-400'}`}>{l.toUpperCase()}</button>)}
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-6"><Outlet /></main>
    </div>
  )
}
