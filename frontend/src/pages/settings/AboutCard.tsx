import { useTranslation } from 'react-i18next'
import { LICENSE_NAME, SOURCE_URL, SPONSOR_URL } from '../../lib/about'

export function AboutCard() {
  const { t } = useTranslation()
  const link = 'text-[var(--accent)] hover:underline'
  return (
    <div className="space-y-2 text-sm">
      <p>{t('about.license', { license: LICENSE_NAME })}</p>
      <p className="text-[var(--muted)]">{t('about.licenseHint')}</p>
      <div className="flex flex-wrap gap-4">
        <a className={link} href={SOURCE_URL} target="_blank" rel="noreferrer">{t('about.source')}</a>
        <a className={link} href={`${SOURCE_URL}/blob/main/LICENSE`} target="_blank" rel="noreferrer">{t('about.licenseText')}</a>
        <a className={link} href={SPONSOR_URL} target="_blank" rel="noreferrer">{t('about.sponsor')}</a>
      </div>
      <p className="text-xs text-[var(--muted)]">{t('about.notAffiliated')}</p>
    </div>
  )
}
