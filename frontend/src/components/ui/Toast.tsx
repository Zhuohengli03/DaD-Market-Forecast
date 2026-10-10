import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'

type Kind = 'ok' | 'error' | 'warn'
interface Toast { id: number; msg: string; kind: Kind }
const Ctx = createContext<{ push: (msg: string, kind?: Kind) => void }>({ push: () => {} })

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])
  const push = useCallback((msg: string, kind: Kind = 'ok') => {
    const id = Date.now() + Math.random()
    setItems(t => [...t, { id, msg, kind }])
    setTimeout(() => setItems(t => t.filter(x => x.id !== id)), kind === 'error' ? 6000 : 3000)
  }, [])
  const tone = { ok: 'bg-emerald-600', error: 'bg-red-600', warn: 'bg-amber-600' }
  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2" role="status">
        {items.map(t => <div key={t.id} className={`${tone[t.kind]} text-white px-4 py-2 rounded-lg shadow-lg text-sm max-w-sm`}>{t.msg}</div>)}
      </div>
    </Ctx.Provider>
  )
}
export const useToast = () => useContext(Ctx)
