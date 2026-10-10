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

export const modelHelp = (name: string, t: TFunction) => t(`modelHelp.${name}`, { defaultValue: '' })
