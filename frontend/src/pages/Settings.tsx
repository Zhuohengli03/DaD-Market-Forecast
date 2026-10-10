import { useTranslation } from 'react-i18next'
import { useSetup, useStatus } from '../api/hooks'
import { ScheduleForm } from '../components/wizard/ScheduleForm'
import { OPEN_WIZARD_EVENT } from '../components/wizard/SetupWizard'
import { Card } from '../components/ui/Card'
import { setLang } from '../i18n'
import { Button } from '../components/ui/Button'
import { ApiKeyCard } from './settings/ApiKeyCard'
import { ItemsConfigForm } from './settings/ItemsConfigForm'
import { SeasonEditor } from './settings/SeasonEditor'

export default function Settings() {
  const { t, i18n } = useTranslation()
  const status = useStatus()
  const setup = useSetup()
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between"><h1 className="text-xl font-semibold">{t('settings.title')}</h1>
        <Button variant="ghost" size="sm" onClick={() => window.dispatchEvent(new Event(OPEN_WIZARD_EVENT))}>{t('wizard.reopen')}</Button></div>
      <Card title={t('settings.apiKey')}><ApiKeyCard /></Card>
      {setup.data && <Card title={t('settings.interval')}><ScheduleForm key={`${setup.data.collect_interval_hours}-${setup.data.auto_collect}`} initialHours={setup.data.collect_interval_hours} initialAuto={setup.data.auto_collect} submitLabel={t('common.save')} /></Card>}
      <Card title={t('settings.seasons')}><p className="text-sm text-[var(--muted)] mb-3">{t('settings.seasonsIntro')}</p><SeasonEditor /></Card>
      <Card title={t('settings.items')}><p className="text-sm text-[var(--muted)] mb-3">{t('settings.itemsIntro')}</p><ItemsConfigForm /></Card>
      <Card title={t('settings.dataSource')}>
        <dl className="text-sm grid grid-cols-[160px_1fr] gap-y-1"><dt className="text-[var(--muted)]">{t('settings.dataSource')}</dt><dd>{status.data ? (status.data.source === 'postgres' ? t('settings.dataSourcePg') : t('settings.dataSourceCsv')) : '…'}</dd><dt className="text-[var(--muted)]">{t('settings.python')}</dt><dd className="font-mono text-xs">{status.data?.python}</dd></dl>
      </Card>
      <Card title={t('settings.language')}><div className="flex gap-2">{(['zh', 'en'] as const).map(l => <Button key={l} size="sm" variant={i18n.language === l ? 'primary' : 'ghost'} onClick={() => setLang(l)}>{l === 'zh' ? '中文' : 'English'}</Button>)}</div></Card>
    </div>
  )
}
