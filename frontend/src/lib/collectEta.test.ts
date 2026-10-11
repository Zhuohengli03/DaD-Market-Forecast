import '../i18n'
import { collectEta, fmtWait } from './collectEta'

const T0 = Date.parse('2026-10-10T00:00:00Z')

it('estimates progress and time left from pages done, the expected pages and the pace so far', () => {
  // 50 of about 200 pages after 50 s: one page a second, 150 s to go
  expect(collectEta(50, 200, '2026-10-10T00:00:00Z', T0 + 50_000)).toEqual({ total: 200, share: 0.25, secondsLeft: 150 })
  // just started: assume DarkerDB's 60 requests a minute
  expect(collectEta(0, 120, '2026-10-10T00:00:00Z', T0 + 1_000).secondsLeft).toBe(120)
  // past the estimate (the estimate rounds up, but can still be short): no false precision
  expect(collectEta(250, 200, '2026-10-10T00:00:00Z', T0 + 250_000)).toEqual({ total: null, share: null, secondsLeft: null })
  expect(collectEta(10, null, '2026-10-10T00:00:00Z', T0 + 10_000)).toEqual({ total: null, share: null, secondsLeft: null })
})

it('says the wait in words', () => {
  expect(fmtWait(20)).toBe('不到 1 分钟')
  expect(fmtWait(150)).toBe('约 3 分钟')
  expect(fmtWait(4000)).toBe('约 1 小时 7 分钟')
})
