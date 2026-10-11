import { localizeMessage } from './labels'

// Compatibility with existing backend responses and saved jobs. Keep unknown
// details intact so a translation never conceals the actual failure.
const MESSAGES: Record<string, string> = {
  'invalid request': '请求参数不正确，请检查填写的内容',
  'start is required': '请填写开始时间',
  'end must be after start': '结束时间必须晚于开始时间',
  'market opening must be within the season': '交易开放日期必须在赛季范围内',
  'range overlaps season': '时间范围与已有赛季重叠',
  "start must be before end within the season's market range": '开始时间必须早于结束时间，且范围需位于赛季可交易期间',
  'Field required': '此项必填',
  'Input should be a valid integer': '请输入整数',
  'Input should be a valid number': '请输入数字',
  'Input should be a valid boolean': '请选择是或否',
  'Input should be a valid string': '请输入文本',
  'min must not exceed max': '最小值不能大于最大值',
  'Failed to fetch': '连接失败，请检查网络和本地服务',
}

const PATTERNS: [RegExp, (m: RegExpExecArray) => string][] = [
  [/^range overlaps season (.+)$/, m => `时间范围与赛季 ${m[1]} 重叠`],
  [/^season (.+) not found$/, m => `找不到赛季 ${m[1]}`],
  [/^run (.+) not found$/, m => `找不到预测记录 ${m[1]}，可能已被删除`],
  [/^holdout (.+) not found$/, m => `找不到回测记录 ${m[1]}，可能已被删除`],
  [/^unknown models (.+)$/, m => `无法识别预测方法：${m[1]}`],
  [/^resolution must be one of (.+)$/, m => `请选择有效的时间粒度：${m[1]}`],
  [/^(?:invalid timestamp|bad timestamp):? (.+)$/, m => `时间格式不正确：${m[1]}`],
  [/^Input should be greater than or equal to (.+)$/, m => `数值不能小于 ${m[1]}`],
  [/^Input should be less than or equal to (.+)$/, m => `数值不能大于 ${m[1]}`],
  [/^String should have at least (\d+) characters?$/, m => `至少输入 ${m[1]} 个字符`],
]

const FIELDS: Record<string, string> = {
  name: '名称', start: '开始时间', end: '结束时间', market_open: '交易开放日期',
  item: '道具', season_id: '赛季', horizon_hours: '预测时长', coverage: '区间可信度',
  resolution: '时间粒度', train_frac: '训练比例', port: '端口', key: '密钥',
}

export function localizeError(message: string, lang: string): string {
  const msg = localizeMessage(message, lang)
  if (!lang.startsWith('zh')) return msg
  if (MESSAGES[msg]) return MESSAGES[msg]
  for (const [pattern, translate] of PATTERNS) {
    const match = pattern.exec(msg)
    if (match) return translate(match)
  }
  // Validation errors prefix the field name, e.g. "horizon_hours: Field required".
  return msg.split('; ').map(part => {
    const match = /^(\w+(?:\.\w+)*): (.+)$/.exec(part)
    if (!match) return part
    const detail = localizeError(match[2], lang)
    return detail === match[2] && !FIELDS[match[1]] ? part : `${FIELDS[match[1]] ?? match[1]}：${detail}`
  }).join('；')
}
