import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Navigate, useLocation } from 'react-router-dom'
import Collect from './Collect'
import Scheduler from './Scheduler'

/** Everything that spends DarkerDB credits on data, in one place: collect now, then scheduled collection. */
export default function CollectManage() {
  const { t } = useTranslation()
  const { hash } = useLocation()
  useEffect(() => { if (hash) document.getElementById(hash.slice(1))?.scrollIntoView() }, [hash])
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline gap-4">
        <h1 className="text-xl font-semibold">{t('collectManage.title')}</h1>
        <nav className="flex gap-3 text-sm"><a href="#now" className="text-[var(--accent)] hover:underline">{t('collectManage.now')}</a>
          <a href="#schedule" className="text-[var(--accent)] hover:underline">{t('scheduler.title')}</a></nav>
      </div>
      <section id="now" className="space-y-3"><h2 className="text-lg font-semibold">{t('collectManage.now')}</h2><Collect embedded /></section>
      <section id="schedule" className="space-y-3"><h2 className="text-lg font-semibold">{t('scheduler.title')}</h2><Scheduler embedded /></section>
    </div>
  )
}

/** The old scheduled-collection address. */
export function SchedulerRedirect() {
  return <Navigate replace to="/collect#schedule" />
}
