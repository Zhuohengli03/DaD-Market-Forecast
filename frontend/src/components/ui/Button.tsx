import type { ButtonHTMLAttributes } from 'react'
import { Spinner } from './Spinner'

type Props = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' | 'danger'; size?: 'sm' | 'md'; loading?: boolean }
export function Button({ variant = 'primary', size = 'md', loading, className = '', children, disabled, ...rest }: Props) {
  const base = 'inline-flex items-center gap-2 rounded-md font-medium transition disabled:opacity-50 disabled:cursor-not-allowed'
  const sizes = { sm: 'px-2.5 py-1 text-xs', md: 'px-3.5 py-2 text-sm' }
  const variants = { primary: 'bg-[var(--accent)] text-white hover:brightness-110', ghost: 'border border-[var(--line)] bg-white hover:bg-gray-50', danger: 'bg-[var(--bad)] text-white hover:brightness-110' }
  return <button type="button" className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} disabled={disabled || loading} {...rest}>{loading && <Spinner size={14} />}{children}</button>
}
