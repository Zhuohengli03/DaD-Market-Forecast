export default {
  nav: { dashboard: '概览', market: '市场', forecast: '预测', collect: '采集', scheduler: '定时任务', settings: '设置' },
  common: { item: '物品', season: '赛季', start: '开始', end: '结束', resolution: '分辨率', run: '运行', running: '运行中', save: '保存', cancel: '取消', delete: '删除', add: '添加', refresh: '刷新', loading: '加载中…', none: '暂无数据', error: '出错了', source: '数据源', unassigned: '未分配', all: '全部', yes: '是', no: '否', actions: '操作', download: '下载 CSV', jobs: '任务', log: '日志' },
  dashboard: { title: '概览', latest: '最新小时中位价', change24: '24 小时变化', lastSeen: '最近挂单', rows: '条挂单', runningJobs: '正在运行的任务', noJobs: '当前没有任务' },
  market: { title: '市场', listings: '挂单', sold: '是否成交', minQty: '最小数量', maxQty: '最大数量', price: '单价', total: '总价', qty: '数量', time: '时间', page: '第 {{page}} 页 / 共 {{pages}} 页', emptyListings: '没有符合条件的挂单', customRange: '自定义范围' },
  forecast: { title: '预测', horizon: '预测步长', models: '参与模型', coverage: '区间覆盖率', start: '开始预测', history: '历史运行', noRuns: '还没有运行记录', beats: '优于基线', notBeats: '未优于基线，显示基线预测', skill: '相对复读上一小时的提升', selected: '选用模型', backtest: '回测', origins: '回测起点', evalHorizon: '评估步长', coverageActual: '回测实际覆盖率', perHorizon: '分步长误差', importance: '特征重要性', references: '叠加其他赛季参考', actuals: '叠加后来的实际价格', warnings: '警告', pending: '正在运行，日志在下方', failed: '运行失败', disconnected: '日志连接中断，请刷新查看任务状态', model: '模型', mae: 'MAE', maeH1: 'MAE h=1', maeH24: 'MAE h=24', mase: 'MASE', dir: '方向命中率', rmse: 'RMSE', deleteRun: '已删除此次运行' },
  collect: { title: '采集', collectNow: '立即采集', recent: '最近任务', started: '已启动采集' },
  scheduler: { title: '定时任务', status: '调度器状态', on: '运行中', off: '已停止', startBtn: '启动调度器', stopBtn: '停止调度器', task: '任务', schedule: '计划', enabled: '启用', runNow: '立即运行' },
  settings: { title: '设置', seasons: '赛季', seasonName: '名称', seasonStart: '开始 (UTC)', seasonEnd: '结束 (UTC，留空为进行中)', note: '备注', items: '物品配置', itemName: '名称', file: '采集脚本', csv: 'CSV 文件', category: '分类', description: '描述', enabled: '启用', dataSource: '数据源', python: 'Python 解释器', language: '语言', saved: '已保存' },
}
