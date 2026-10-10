import type { TFunction } from 'i18next'

export const modelLabel = (name: string, t: TFunction) => t(`models.${name}`, { defaultValue: name })
export const modelHelp = (name: string, t: TFunction) => t(`modelHelp.${name}`, { defaultValue: '' })
