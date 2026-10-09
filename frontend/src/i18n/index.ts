import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import zh from './zh'
import en from './en'

const saved = (() => { try { return localStorage.getItem('lang') } catch { return null } })()
i18n.use(initReactI18next).init({ resources: { zh: { translation: zh }, en: { translation: en } }, lng: saved || 'zh', fallbackLng: 'zh', interpolation: { escapeValue: false } })
export function setLang(lang: 'zh' | 'en') { i18n.changeLanguage(lang); try { localStorage.setItem('lang', lang) } catch { /* ignore */ } }
export default i18n
