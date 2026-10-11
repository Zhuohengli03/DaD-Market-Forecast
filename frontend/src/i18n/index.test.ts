import i18n, { setLang } from './index'
import zh from './zh'
import en from './en'

function flatten(value: object, prefix = ''): Record<string, string> {
  return Object.fromEntries(Object.entries(value).flatMap(([key, v]) => typeof v === 'object'
    ? Object.entries(flatten(v, `${prefix}${key}.`)) : [[prefix + key, v]]))
}

it('has Chinese translations and matching placeholders for every English message', () => {
  const chinese = flatten(zh)
  for (const [key, english] of Object.entries(flatten(en))) {
    expect(chinese[key], key).toBeDefined()
    expect(chinese[key].match(/{{\w+}}/g)?.sort() ?? [], key).toEqual(english.match(/{{\w+}}/g)?.sort() ?? [])
  }
})

it('updates the document language and title when switching languages', () => {
  try {
    setLang('en')
    expect(document.documentElement.lang).toBe('en')
    expect(document.title).toContain('Market tracking')
    setLang('zh')
    expect(document.documentElement.lang).toBe('zh-CN')
    expect(document.title).toContain('行情追踪与预测')
    expect(i18n.t('apiKey.label')).toBe('API 密钥')
  } finally { setLang('zh') }
})
