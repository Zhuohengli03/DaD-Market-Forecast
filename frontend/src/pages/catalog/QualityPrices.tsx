import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useOverview } from '../../api/hooks'
import { Card } from '../../components/ui/Card'
import { Sparkline } from '../../components/charts/Sparkline'
import { fmtNum, fmtPct } from '../../lib/format'
import { variantLabel } from '../dashboard/TypeCard'

/** Latest price, 24h change and 48h trend of every tracked quality of one item type. */
export function QualityPrices({ archetype }: { archetype: string }) {
  const { t, i18n } = useTranslation()
  const group = useOverview().data?.find(g => g.archetype === archetype)
  if (!group) return null
  return (
    <Card title={t('catalog.detail.prices')}>
      <p className="text-xs text-[var(--muted)] mb-2">{t('catalog.detail.pricesHint')}</p>
      <table className="w-full text-sm">
        <thead className="text-left text-[var(--muted)] text-xs"><tr><th className="py-1 font-normal">{t('dashboard.quality')}</th><th className="text-right font-normal">{t('dashboard.latestPrice')}</th><th className="text-right font-normal">{t('dashboard.change24Short')}</th><th className="w-48 font-normal pl-4">{t('dashboard.sparkHint')}</th><th className="text-right font-normal">{t('dashboard.seasonRowsShort')}</th><th></th></tr></thead>
        <tbody>{group.variants.map(v => (
          <tr key={v.name} className="border-t border-[var(--line)]">
            <td className="py-1.5">{variantLabel(v, i18n.language)}</td>
            <td className="text-right tabular-nums font-medium">{v.rows ? fmtNum(v.latest) : '–'}</td>
            <td className={`text-right tabular-nums ${v.change24 == null ? 'text-[var(--muted)]' : v.change24 >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}`}>{v.change24 == null ? '–' : fmtPct(v.change24)}</td>
            <td className="pl-4">{v.rows ? <Sparkline values={v.spark} /> : <span className="text-xs text-[var(--muted)]">{t('dashboard.noDataShort')}</span>}</td>
            <td className="text-right tabular-nums text-[var(--muted)]">{v.season_rows}</td>
            <td className="text-right whitespace-nowrap pl-3">{v.rows > 0 && <Link to={`/market?item=${encodeURIComponent(v.name)}`} className="text-[var(--accent)] hover:underline">{t('catalog.detail.viewTrend')}</Link>}</td>
          </tr>))}</tbody>
      </table>
    </Card>
  )
}
