import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNetwork, useRestart, useSaveNetwork } from '../../api/hooks'
import { Button } from '../../components/ui/Button'
import { inputCls } from '../../components/ui/Field'
import { useToast } from '../../components/ui/Toast'

/** Wait until the restarted server answers, then open it (the port may have changed). */
export async function waitForServer(url: string, tries = 60, delayMs = 1000): Promise<boolean> {
  for (let i = 0; i < tries; i++) {
    await new Promise(r => setTimeout(r, delayMs))
    try {
      await fetch(`${url}/api/status`, { mode: 'no-cors', cache: 'no-store' })   // any answer means it is up
      return true
    } catch { /* still restarting */ }
  }
  return false
}

/** LAN access and port, saved to the app's settings file and applied by restarting from here. */
export function NetworkCard() {
  const { t } = useTranslation()
  const net = useNetwork(); const save = useSaveNetwork(); const restart = useRestart(); const toast = useToast()
  const [lan, setLan] = useState(false)
  const [port, setPort] = useState('5001')
  const [phase, setPhase] = useState<'idle' | 'restarting' | 'failed'>('idle')
  useEffect(() => { if (net.data) { setLan(net.data.lan); setPort(String(net.data.port)) } }, [net.data])
  if (!net.data) return <p className="text-sm">{t('common.loading')}</p>
  const n = Number(port)
  const badPort = !Number.isInteger(n) || n < 1024 || n > 65535
  const dirty = lan !== net.data.lan || n !== net.data.port
  const doRestart = () => restart.mutate(undefined, {
    onSuccess: async () => {
      setPhase('restarting')
      const url = `${window.location.protocol}//${window.location.hostname}:${net.data!.port}`
      if (await waitForServer(url)) window.location.href = url
      else setPhase('failed')
    },
    onError: e => toast.push((e as Error).message, 'error'),
  })
  return (
    <div className="space-y-3 text-sm">
      <label className="flex items-start gap-2">
        <input type="checkbox" className="mt-1" checked={lan} onChange={e => setLan(e.target.checked)} />
        <span><span className="font-medium">{t('network.lan')}</span>
          <span className="block text-xs text-[var(--muted)]">{t('network.lanHint')}</span></span>
      </label>
      {lan && <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs">{t('network.lanWarn')}</p>}
      <label className="flex flex-wrap items-center gap-2">{t('network.port')}
        <input type="number" min={1024} max={65535} aria-label={t('network.port')} className={`${inputCls} w-28`} value={port} onChange={e => setPort(e.target.value)} />
        <span className="text-xs text-[var(--muted)]">{t('network.portHint')}</span>
      </label>
      {badPort && <p role="alert" className="text-xs text-[var(--bad)]">{t('network.badPort')}</p>}
      <Button size="sm" disabled={!dirty || badPort} loading={save.isPending}
        onClick={() => save.mutate({ lan, port: n }, { onSuccess: () => toast.push(t('network.saved')), onError: e => toast.push((e as Error).message, 'error') })}>{t('common.save')}</Button>
      {net.data.lan && net.data.lan_urls.length > 0 && !net.data.restart_required && (
        <p className="text-xs">{t('network.openOn')} {net.data.lan_urls.map(u => <code key={u} className="ml-1 px-1 rounded bg-gray-100">{u}</code>)}</p>)}
      {net.data.restart_required && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2 space-y-2">
          <p>{t('network.needRestart')}</p>
          {phase === 'restarting' ? <p className="text-xs">{t('network.restarting')}</p>
            : phase === 'failed' ? <p role="alert" className="text-xs text-[var(--bad)]">{t('network.restartFailed')}</p>
            : <Button size="sm" loading={restart.isPending} onClick={doRestart}>{t('network.restartNow')}</Button>}
        </div>)}
    </div>
  )
}
