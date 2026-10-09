import { fmtNum, fmtPct, fmtTs, fmtTsUtc } from './format'

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
