import { fmtNum, fmtPct, fmtTs } from './format'

it('formats numbers, percents and timestamps', () => {
  expect(fmtNum(12.3456)).toBe('12.35')
  expect(fmtNum(null)).toBe('–')
  expect(fmtPct(0.1234)).toBe('+12.3%')
  expect(fmtPct(-0.05)).toBe('-5.0%')
  expect(fmtTs('2026-01-20T00:00:00+00:00')).toMatch(/\d/)
})
