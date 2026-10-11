import { localizeError } from './errors'

it('translates season validation and missing saved results without changing identifiers', () => {
  expect(localizeError('end must be after start', 'zh')).toBe('结束时间必须晚于开始时间')
  expect(localizeError('run gold_ore-123 not found', 'zh')).toBe('找不到预测记录 gold_ore-123，可能已被删除')
  expect(localizeError('market opening must be within the season', 'zh')).toContain('赛季范围内')
})

it('translates field-level validation, keeps limits and preserves unknown diagnostics', () => {
  expect(localizeError('name: Field required; horizon_hours: Input should be greater than or equal to 1', 'zh'))
    .toBe('名称：此项必填；预测时长：数值不能小于 1')
  expect(localizeError('OddFailure: unknown diagnostic', 'zh')).toBe('OddFailure: unknown diagnostic')
  expect(localizeError('end must be after start', 'en')).toBe('end must be after start')
})
