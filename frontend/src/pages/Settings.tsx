import { useTranslation } from 'react-i18next'
import { useStatus } from '../api/hooks'
import { Card } from '../components/ui/Card'
import { setLang } from '../i18n'
import { Button } from '../components/ui/Button'
import { ItemsConfigForm } from './settings/ItemsConfigForm'
import { SeasonEditor } from './settings/SeasonEditor'

export default function Settings() {
  const { t, i18n } = useTranslation()
  const status = useStatus()
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">{t('settings.title')}</h1>
      <Card title={t('settings.seasons')}><SeasonEditor /></Card>
      <Card title={t('settings.items')}><ItemsConfigForm /></Card>
      <Card title={t('settings.dataSource')}>
        <dl className="text-sm grid grid-cols-[160px_1fr] gap-y-1"><dt className="text-[var(--muted)]">{t('settings.dataSource')}</dt><dd>{status.data?.source ?? '…'}</dd><dt className="text-[var(--muted)]">{t('settings.python')}</dt><dd className="font-mono text-xs">{status.data?.python}</dd></dl>
      </Card>
      <Card title={t('settings.language')}><div className="flex gap-2">{(['zh', 'en'] as const).map(l => <Button key={l} size="sm" variant={i18n.language === l ? 'primary' : 'ghost'} onClick={() => setLang(l)}>{l === 'zh' ? '中文' : 'English'}</Button>)}</div></Card>
    </div>
  )
}
