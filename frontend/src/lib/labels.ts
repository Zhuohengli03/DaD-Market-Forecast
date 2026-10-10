import type { TFunction } from 'i18next'

export const modelLabel = (name: string, t: TFunction) => t(`models.${name}`, { defaultValue: name })
export const rarityLabel = (r: string | null | undefined, t: TFunction) => (r ? t(`rarity.${r}`, { defaultValue: r }) : '')
export const typeLabel = (x: string | null | undefined, t: TFunction) => (x ? t(`itemType.${x}`, { defaultValue: x }) : '')
export const paramLabel = (key: string, t: TFunction) => t(`param.${key}`, { defaultValue: key })

/** Chinese name when the interface is Chinese and DarkerDB provided one; English otherwise. */
export function itemName(row: { name: string; name_zh?: string | null }, lang: string): string {
  return lang.startsWith('zh') && row.name_zh ? row.name_zh : row.name
}

/** Server messages are written "中文 / English"; show the half that matches the interface. */
export function localizeMessage(msg: string, lang: string): string {
  const i = msg.indexOf(' / ')
  if (i < 0) return msg
  return lang.startsWith('zh') ? msg.slice(0, i) : msg.slice(i + 3)
}

const VALUE_DICTS: Record<string, string> = { rarity: 'rarity', item_type: 'itemType', slot_type: 'slot' }

/** Display a raw DarkerDB parameter value in the interface language where we know the vocabulary. */
export function paramValue(key: string, value: unknown, t: TFunction): string {
  if (typeof value === 'boolean') return value ? t('yes') : t('no')
  if (value === null || value === undefined) return ''
  const s = String(value)
  if (VALUE_DICTS[key]) return t(`${VALUE_DICTS[key]}.${s}`, { defaultValue: s })
  if (key === 'season_label' || key === 'patch_name') {
    const m = /^Season (\d+)$/.exec(s)
    if (m) return t('patch.seasonLabel', { n: m[1] })
  }
  return s
}

export function patchLabel(title: string | null | undefined, t: TFunction): string {
  if (!title) return '–'
  const m = /Season #?(\d+)/.exec(title)
  return m ? t('patch.season', { n: m[1] }) : title
}

export const modelHelp = (name: string, t: TFunction) => t(`modelHelp.${name}`, { defaultValue: '' })

const OLD_MODEL_ZH: Record<string, string> = { naive: '沿用上一小时', seasonal_naive_24: '沿用昨天同一时刻', ets: '指数平滑', arima: 'ARIMA 时间序列', ridge: '线性回归', hgb: '梯度提升树', rf: '随机森林', ensemble: '多模型加权' }

// forecast runs saved before the interface was translated carry English warnings
const OLD_WARNINGS: [RegExp, (m: RegExpExecArray) => string][] = [
  [/^(\d+) gap hours forward-filled for model input \(not scored\)$/, m => `有 ${m[1]} 个小时没有挂单，已用前一小时价格补齐（只用于建模，不计入评分）`],
  [/^(\d+) forward-filled hour-steps excluded from backtest scoring$/, m => `${m[1]} 个前向填充的时间点没有计入回测评分`],
  [/^intervals beyond h=(\d+) reuse the last backtested width$/, m => `超过 ${m[1]} 小时的部分，区间宽度沿用回测到的最后一个值`],
  [/^evaluation horizon shrunk from (\d+)h to (\d+)h to obtain \d+ backtest origins \(series has (\d+) hours\)$/, m => `数据只有 ${m[3]} 小时，回测评估长度从 ${m[1]} 小时缩短到 ${m[2]} 小时，以获得足够的回测次数`],
  [/^only (\d+) backtest origins available \(wanted (\d+)\)$/, m => `只能做 ${m[1]} 次回测（建议至少 ${m[2]} 次），结果参考价值有限`],
  [/^latest listing is (\d+) hours old; the forecast starts from then, not from now\. Collect fresh data first\.$/, m => `最新的挂单是 ${m[1]} 小时前的，预测从那时起算，不是从现在。请先采集新数据。`],
  [/^showing the naive forecast: (.*)$/, m => `显示基准预测：${localizeWarning(m[1], 'zh')}`],
  [/^only (\d+) backtest origins; at least (\d+) are needed to trust a winner$/, m => `只有 ${m[1]} 次回测，至少需要 ${m[2]} 次才能判断哪个方法更准`],
  [/^best model \((\S+)\) was only ([\d.]+%) better, within noise \(needs > ([\d.]+%)\)$/, m => `最好的方法（${OLD_MODEL_ZH[m[1]] ?? m[1]}）只比基准准 ${m[2]}，在随机波动范围内（需要超过 ${m[3]}）`],
  [/^no model beat repeating the last hour$/, () => '没有任何方法比“沿用上一小时”更准'],
  [/^(\S+) failed on the full season \((.*)\); falling back to the naive forecast$/, m => `${m[1]} 在完整赛季数据上运行失败（${m[2]}），改用基准预测`],
]

/** Show a forecast warning in Chinese, including English ones stored by older versions. */
export function localizeWarning(msg: string, lang: string): string {
  if (!lang.startsWith('zh')) return msg
  for (const [re, fn] of OLD_WARNINGS) {
    const m = re.exec(msg)
    if (m) return fn(m)
  }
  return msg
}

/** "Season 2" (older saved runs) -> "第 2 赛季"; Chinese names pass through. */
export function seasonLabel(name: string | null | undefined, t: TFunction): string {
  if (!name) return '–'
  const m = /^Season #?(\d+)$/.exec(name.trim())
  return m ? t('patch.seasonLabel', { n: m[1] }) : name
}

const RARITY_CODES = new Set(['poor', 'common', 'uncommon', 'rare', 'epic', 'legendary', 'unique', 'artifact'])

/** A tracked item's own note, unless it is just a raw rarity code that older versions stored there. */
export function itemNote(description: string | null | undefined): string {
  const d = (description ?? '').trim()
  return RARITY_CODES.has(d.toLowerCase()) ? '' : d
}
