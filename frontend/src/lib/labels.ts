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
