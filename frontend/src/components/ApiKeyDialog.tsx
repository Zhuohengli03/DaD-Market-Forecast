import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { KeyRound } from 'lucide-react'
import { useSetup } from '../api/hooks'
import { ApiKeyForm } from './ApiKeyForm'
import { Button } from './ui/Button'
import { useToast } from './ui/Toast'

/** First-run prompt: shown whenever no DarkerDB key is configured, until saved or dismissed for this session. */
export function ApiKeyDialog() {
  const { t } = useTranslation()
  const setup = useSetup()
  const toast = useToast()
  const [dismissed, setDismissed] = useState(() => { try { return sessionStorage.getItem('apiKeyDismissed') === '1' } catch { return false } })
  const open = setup.isSuccess && !setup.data?.api_key_configured && !dismissed
  if (!open) return null
  const dismiss = () => { setDismissed(true); try { sessionStorage.setItem('apiKeyDismissed', '1') } catch { /* private mode */ } }
  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center p-4">
      <div role="dialog" aria-modal="true" aria-labelledby="api-key-title" className="bg-white rounded-xl shadow-xl max-w-lg w-full p-6 space-y-4">
        <div className="flex items-center gap-3"><KeyRound className="text-[var(--accent)]" /><h2 id="api-key-title" className="text-lg font-semibold">{t('apiKey.title')}</h2></div>
        <p className="text-sm">{t('apiKey.why')}</p>
        <ApiKeyForm onSaved={() => toast.push(t('apiKey.saved'))} extra={<Button variant="ghost" onClick={dismiss}>{t('apiKey.later')}</Button>} />
      </div>
    </div>
  )
}
