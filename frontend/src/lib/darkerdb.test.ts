import i18n from '../i18n'
import { firstRecordLabel } from './darkerdb'

it('marks dates that are only DarkerDB\'s record start', () => {
  expect(firstRecordLabel('2026-07-31T00:00:20Z', i18n.t)).toBe('2026-07-31（记录起点）')
  expect(firstRecordLabel('2026-10-09T03:00:00Z', i18n.t)).toBe('2026-10-09')
})
