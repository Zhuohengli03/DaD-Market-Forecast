import { useTranslation } from 'react-i18next'

export function Spinner({ size = 18 }: { size?: number }) {
  const { t } = useTranslation()
  return <span className="inline-block animate-spin rounded-full border-2 border-current border-t-transparent" style={{ width: size, height: size }} aria-label={t('common.loading')} />
}
