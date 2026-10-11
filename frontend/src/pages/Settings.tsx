import { useTranslation } from 'react-i18next'
import { useSetup, useStatus } from '../api/hooks'
import { OPEN_WIZARD_EVENT } from '../components/wizard/SetupWizard'
import { Card } from '../components/ui/Card'
import { setLang } from '../i18n'
import { Button } from '../components/ui/Button'
import { ApiKeyCard } from './settings/ApiKeyCard'
import { QuotaCard } from './settings/QuotaCard'
import { AboutCard } from './settings/AboutCard'
import { NetworkCard } from './settings/NetworkCard'
import { LLMCard } from './settings/LLMCard'
import { RulesCard } from './settings/RulesCard'
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
      {setup.data && <Card title={t('quota.title')}><QuotaCard key={setup.data.monthly_quota} initial={setup.data.monthly_quota} /></Card>}
      <Card title={t('llm.title')}><LLMCard /></Card>
      <Card title={t('rules.title')}><RulesCard /></Card>
      <Card title={t('settings.seasons')}><p className="text-sm text-[var(--muted)] mb-3">{t('settings.seasonsIntro')}</p><SeasonEditor /></Card>
      <Card title={t('settings.items')}><p className="text-sm text-[var(--muted)] mb-3">{t('settings.itemsIntro')}</p><ItemsConfigForm /></Card>
      <Card title={t('settings.dataSource')}>
        <dl className="text-sm grid grid-cols-[160px_1fr] gap-y-1"><dt className="text-[var(--muted)]">{t('settings.dataSource')}</dt><dd>{status.data ? (status.data.source === 'postgres' ? t('settings.dataSourcePg') : status.data.source === 'sqlite' ? t('settings.dataSourceSqlite') : t('settings.dataSourceCsv')) : '…'}</dd><dt className="text-[var(--muted)]">{t('settings.python')}</dt><dd className="font-mono text-xs">{status.data?.python}</dd></dl>
      </Card>
      <Card title={t('network.title')}><NetworkCard /></Card>
      <Card title={t('about.title')}><AboutCard /></Card>
      <Card title={t('settings.language')}><div className="flex gap-2">{(['zh', 'en'] as const).map(l => <Button key={l} size="sm" variant={i18n.language === l ? 'primary' : 'ghost'} onClick={() => setLang(l)}>{l === 'zh' ? '中文' : 'English'}</Button>)}</div></Card>
    </div>
  )
}
