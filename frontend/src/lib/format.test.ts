import { fmtDate, fmtNum, fmtPct, fmtRelative, fmtTs, fmtTsUtc } from './format'

it('formats numbers, percents and timestamps', () => {
  expect(fmtNum(12.3456)).toBe('12.35')
  expect(fmtNum(null)).toBe('–')
  expect(fmtPct(0.1234)).toBe('+12.3%')
  expect(fmtPct(-0.05)).toBe('-5.0%')
  expect(fmtTs('2026-01-20T00:00:00+00:00')).toMatch(/\d/)
})

it('fmtTsUtc renders the UTC wall-clock with a Z suffix', () => {
  expect(fmtTsUtc('2025-10-01T00:00:00+00:00')).toBe('2025-10-01 00:00Z')
  expect(fmtTsUtc(null)).toBe('–')
})

describe('fmtRelative', () => {
  const ago = (ms: number) => new Date(Date.now() - ms).toISOString()
  it('says how long ago in Chinese', async () => {
    const i18n = (await import('../i18n')).default
    await i18n.changeLanguage('zh')
    expect(fmtRelative(ago(8 * 6e4))).toBe('8 分钟前')
    expect(fmtRelative(ago(3 * 3.6e6))).toBe('3 小时前')
    expect(fmtRelative(ago(3 * 86.4e6))).toBe('3 天前')
    expect(fmtRelative(ago(10e3))).toBe('刚刚')
    expect(fmtRelative(null)).toBe('–')
  })
  it('and in English', async () => {
    const i18n = (await import('../i18n')).default
    await i18n.changeLanguage('en')
    expect(fmtRelative(ago(8 * 6e4))).toBe('8 min ago')
    await i18n.changeLanguage('zh')
  })
})

it('shows calendar dates in UTC, like season boundaries', () => {
  expect(fmtDate('2026-07-31T00:00:20Z')).toBe('2026-07-31')        // not 07-30 in a timezone behind UTC
  expect(fmtDate('2026-10-09T23:59:00Z')).toBe('2026-10-09')
  expect(fmtDate(null)).toBe('–')
})

describe('fmtInZone', () => {
  it('shows a UTC instant on the wall clock of the chosen city', async () => {
    const { fmtInZone } = await import('./format')
    expect(fmtInZone('2026-09-20T14:00:00Z', 'America/New_York')).toBe('09-20 10:00')   // EDT, UTC-4
    expect(fmtInZone('2026-12-20T14:00:00Z', 'America/New_York')).toBe('12-20 09:00')   // EST, UTC-5
    expect(fmtInZone('2026-09-20T14:00:00Z', 'UTC')).toBe('09-20 14:00')
    expect(fmtInZone(null, 'UTC')).toBe('–')
  })
})
