import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSetup } from '../../api/hooks'
import { ApiKeyForm } from '../../components/ApiKeyForm'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'

export function ApiKeyCard() {
  const { t } = useTranslation()
  const setup = useSetup(); const toast = useToast()
  const [editing, setEditing] = useState(false)
  const configured = !!setup.data?.api_key_configured
  if (!editing && configured) {
    return <div className="flex items-center justify-between text-sm"><span><Badge tone="ok">✓</Badge> {t('apiKey.current', { hint: setup.data?.api_key_hint })}</span>
      <Button size="sm" variant="ghost" onClick={() => setEditing(true)}>{t('apiKey.change')}</Button></div>
  }
  return (
    <div className="space-y-2">
      {!configured && <p className="text-sm text-[var(--warn)]">{t('apiKey.missing')}</p>}
      <ApiKeyForm onSaved={() => { setEditing(false); toast.push(t('apiKey.saved')) }} extra={configured ? <Button variant="ghost" onClick={() => setEditing(false)}>{t('common.cancel')}</Button> : undefined} />
    </div>
  )
}
