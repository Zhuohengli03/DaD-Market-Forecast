import { useTranslation } from 'react-i18next'
import { Card } from '../components/ui/Card'
import { SeasonHoldout } from './forecast/SeasonHoldout'

/** Season holdout on its own page: how well the forecaster did on an ended season it had not seen the end of. */
export default function Holdout() {
  const { t } = useTranslation()
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">{t('holdout.title')}</h1>
      <Card><SeasonHoldout /></Card>
    </div>
  )
}
