import i18n from '../i18n'
import { modelLabel } from './labels'

it('gives models plain-language names', () => {
  expect(modelLabel('naive', i18n.t)).toBe('沿用上一小时')
  expect(modelLabel('seasonal_naive_24', i18n.t)).toBe('沿用昨天同一时刻')
  expect(modelLabel('mystery', i18n.t)).toBe('mystery')
})
