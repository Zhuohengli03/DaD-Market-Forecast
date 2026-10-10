# Darker Market：预测流水线重建与交互式前后端设计

日期：2026-10-09
状态：待用户评审

## 1. 背景与目标

现有 `src/analysis/Machine_learning_analysis.py` 的预测结果不可信，根因有七条（已逐条对照代码确认）：

1. 目标泄露：`price_quantity_ratio`、`price_quantity_interaction` 用同一行的总价构造，目标 `price_per_unit` 是它们的确定函数。
2. 按挂单逐条回归，滞后特征是"上一条挂单"而非上一小时。
3. "7 日预测"只是把测试集最后 7 行逐行喂给模型，没有向前推演。
4. ARIMA 的"步"是挂单条数，与 ML 的"步"口径不同，却按手写权重融合。
5. 置信区间来自泄露模型的测试残差，"95%"只是名义。
6. 离群点用全量数据做 z-score；交叉验证用未打乱的 KFold；"当前价格"取最后一条挂单。
7. 跨赛季概念漂移未处理。

现有 Web 壳（`app.py` + `frontend/`）只是把命令行输出塞进页面。

目标：

- 新写一条诚实的小时级预测流水线，七条问题全部消除，结果带可验证的回测证据。
- 新写一个像软件的浏览器单页应用和配套后端：可交互的市场浏览、预测运行与回看、采集与调度控制、配置管理。

非目标：

- 不改采集脚本 `src/api/*_API.py`、数据库层 `src/database/`、调度器 `src/scheduler/` 的逻辑。
- 不做多用户、登录、远程部署。单人本地使用。
- 不做桌面打包（后续可在此基础上加 Tauri）。

## 2. 已确认的决策

| 决策 | 结论 |
|---|---|
| 应用形态 | 浏览器单页应用，本地 localhost |
| 数据源 | Postgres 为主，连不上退回项目根目录 CSV |
| 赛季边界 | 只手动配置（API 数据是连续的，空档不代表 wipe） |
| 时间粒度 | 小时级序列，预测未来 24~168 小时 |
| 外部评审 | 不用 CCB（codex/gemini），评审在本会话内完成 |
| 时区 | 存储与计算用 UTC，界面按浏览器本地时区显示 |
| 赛季作用域 | 全局，不分物品 |

## 3. 代码布局

```
Darker Market/
├── backend/                    # FastAPI 应用（替换 app.py）
│   ├── main.py                 # 应用工厂、路由挂载、静态托管
│   ├── settings.py             # 环境变量 → Settings
│   ├── schemas.py              # Pydantic 请求/响应模型，统一信封
│   ├── data/
│   │   ├── repository.py       # ListingRepository 协议 + 工厂（探测 DB，失败退 CSV）
│   │   ├── postgres_repo.py
│   │   ├── csv_repo.py
│   │   └── seasons.py          # seasons.json 读写
│   ├── services/
│   │   ├── series.py           # 小时聚合、重采样（1h/4h/1d）
│   │   ├── jobs.py             # 任务注册表、日志环形缓冲、SSE 发布
│   │   ├── collection.py       # 启动采集子进程
│   │   ├── scheduler_ctl.py    # 调度器子进程控制、任务列表
│   │   └── forecast_runs.py    # runs/ 目录读写
│   └── api/
│       ├── items.py  series.py  seasons.py  forecast.py  jobs.py  scheduler.py  config.py
├── src/forecast/               # 预测流水线（纯 Python，不依赖 FastAPI）
│   ├── aggregate.py            # 挂单 → 小时序列
│   ├── features.py             # 仅用过去的特征
│   ├── models/
│   │   ├── base.py             # Forecaster 协议：fit(series) / predict(h)
│   │   ├── baselines.py        # naive、seasonal_naive_24
│   │   ├── statistical.py      # ETS、ARIMA
│   │   └── ml.py               # Ridge、HistGradientBoosting、RandomForest（递归多步）
│   ├── backtest.py             # 滚动起点回测
│   ├── intervals.py            # 按步长的经验分位数区间
│   ├── select.py               # 模型选择与集成判定
│   ├── reference.py            # 其他赛季参考曲线对齐
│   ├── pipeline.py             # run_forecast(request) → ForecastResult
│   └── cli.py                  # 命令行入口（替换 src/web/run_analysis_cli.py）
├── frontend/                   # React + TypeScript + Vite（替换现有原生页面）
├── runs/                       # 预测运行结果 JSON（git 忽略）
├── seasons.json                # 赛季配置
└── tests/
    ├── forecast/
    └── backend/
```

删除：`app.py`、`src/web/`、`src/analysis/Machine_learning_analysis.py`、现有 `frontend/` 内容。删除时机：新流水线与新前端通过测试之后，在同一次提交里完成。

## 4. 数据层

### 4.1 仓储接口

```python
class ListingRepository(Protocol):
    def list_items(self) -> list[ItemInfo]            # 名称、表/文件、行数、最早/最新 created_at
    def fetch(self, item: str, start: datetime | None, end: datetime | None,
              sold: bool | None, min_qty: int | None, max_qty: int | None,
              limit: int, offset: int) -> tuple[list[Listing], int]   # 行 + 总数
    def fetch_all(self, item: str, start, end) -> pd.DataFrame        # 给聚合和预测用
    def source_name(self) -> str                                       # "postgres" | "csv"
```

- `Listing` 字段与现有表一致：`item_id, item, quantity, price_per_unit, price, has_sold, created_at`。
- `PostgresListingRepository`：表名按物品名 `lower().replace(' ', '_')`，过滤条件参数化查询。
- `CsvListingRepository`：读 `items_config.json` 里每个物品的 `csv`，用 pandas 读取并缓存，按文件 mtime 失效。
- 工厂 `make_repository(settings)`：先尝试 Postgres（`connect_timeout=3`），失败则记日志并返回 CSV 实现。当前来源通过 `/api/status` 暴露。

### 4.2 赛季

`seasons.json`：

```json
{ "seasons": [
  { "id": "s1", "name": "Season 1", "start": "2025-10-06T00:00:00Z", "end": null, "note": "" }
] }
```

- `end` 为 null 表示进行中。赛季不允许重叠；保存时校验。
- 挂单归属赛季按 `created_at ∈ [start, end)`。
- 不属于任何赛季的数据在界面上归入"未分配"，可以浏览，不能用于预测。

## 5. 小时序列

`src/forecast/aggregate.py::hourly_series(df) -> pd.DataFrame`，索引为 UTC 整点，列：

| 列 | 定义 |
|---|---|
| `median_price` | 该小时内挂单 `price_per_unit` 中位数（预测目标） |
| `min_price`, `max_price` | 最低、最高单价 |
| `p25`, `p75` | 四分位 |
| `listings` | 挂单条数 |
| `sold` | `has_sold` 为真的条数 |
| `volume` | `quantity` 之和 |

- 没有挂单的小时保留为 NaN 行（不删除，不填充）。界面显示为缺口。
- 建模时对 `median_price` 的缺口做前向填充，最多填 6 小时；更长的缺口把序列切断，只取最后一段连续数据训练。切断事实写入结果的 `warnings`。
- 不做基于全量均值/标准差的离群点删除。中位数聚合已抵消捆绑挂单（如 2 个 0.5）造成的极端值。
- 重采样到 4h/1d 只用于界面显示，不用于建模。

## 6. 预测流水线

### 6.1 输入

```python
@dataclass(frozen=True)
class ForecastRequest:
    item: str
    season_id: str
    horizon_hours: int            # 24 ≤ h ≤ 168
    models: tuple[str, ...]       # 子集，默认全部
    coverage: float = 0.9         # 区间覆盖率
    backtest_origin_step: int = 6 # 回测起点间隔（小时）
```

### 6.2 步骤

1. 取赛季内该物品全部挂单，聚合成小时序列。有效点数不足 72 小时则拒绝，返回明确错误。
2. 特征（仅用过去，见 6.3）。
3. 滚动起点回测（6.4），得到每个模型每个步长的误差分布。
4. 模型选择与集成判定（6.5）。
5. 用赛季内全部数据重新拟合入选模型，向前递归预测 `horizon_hours`。
6. 区间：按步长取回测残差分位数（6.6）。
7. 参考曲线：其他赛季按"赛季开始后第 N 小时"对齐（6.7）。
8. 组装 `ForecastResult`，写入 `runs/<run_id>.json`。

### 6.3 特征

对小时序列的每一行 t，特征只允许引用 t-1 及更早的行：

- `median_price` 滞后 1, 2, 3, 6, 12, 24
- `volume`、`listings` 滞后 1, 6, 24
- 以滞后 1 的价格为基础的滚动均值和标准差，窗口 6、24
- 滞后 1 的价格相对滞后 24 的变化率
- 日历：小时（0–23，用 sin/cos）、星期几
- `hours_since_season_start`

不存在任何引用 t 行 `price`、`price_per_unit`、`quantity` 的特征。测试 `test_no_same_row_leakage`：把第 t 行目标改成任意值，断言第 t 行全部特征不变。

### 6.4 滚动起点回测

- 起点集合：从序列第 `max(48, 0.6·N)` 行开始，每隔 `backtest_origin_step` 小时一个起点，直到 `N - horizon_hours`。起点少于 5 个则缩短评估 horizon 并记入 `warnings`。
- 每个起点：用起点前全部数据拟合，向前预测 `horizon_hours`，与真实值比较。
- 记录：每个模型、每个步长 h 的误差数组。
- 汇总指标：MAE、RMSE、MASE（相对 naive 的 MAE 比值）、方向命中率；按三个视角给出：h=1、h=24、全程平均。
- 基线 `naive`（复读上一小时）、`seasonal_naive_24`（复读 24 小时前）始终参与，不可关闭。

ML 模型的递归多步：预测 t+1 后，把预测值填回序列，重新计算 t+1 行的特征，再预测 t+2。`volume`、`listings` 在未来未知，按训练窗口末 24 小时的同小时均值填充。

### 6.5 模型选择与集成

- 按全程平均 MAE 排序。
- 集成候选：按回测 MAE 倒数加权前 3 名。只有当集成在回测中的 MAE 低于每一个成员时才采用集成，否则用最好的单模型。
- `skill_vs_naive = 1 - MAE_selected / MAE_naive`。若 ≤ 0，`selected = "naive"`，结果里 `beats_baseline = false`。界面据此直接告知。

### 6.6 区间

对入选模型，取回测中步长 h 的残差 `r_h`，区间为 `[pred_h + Q_{(1-c)/2}(r_h), pred_h + Q_{(1+c)/2}(r_h)]`。残差少于 10 个的步长向相邻步长借样本。同时报告回测内的实际覆盖率。

### 6.7 参考曲线

对其他每个赛季：同一物品的小时序列按 `hours_since_season_start` 对齐，取与当前赛季末尾同一偏移往后 `horizon_hours` 的片段。只返回中位价，标为 `reference`，不参与任何计算。

### 6.8 输出

`ForecastResult`（即 `runs/<run_id>.json`）：

```
run_id, created_at, request,
source: "postgres" | "csv",
series_tail: [{ts, median_price, p25, p75, volume}]   # 最近 168 小时
forecast: [{ts, h, pred, lower, upper}]
selected: { name, members: [{name, weight}] }
backtest: {
  origins: int, horizon: int,
  models: [{ name, mae_h1, mae_h24, mae_mean, rmse_mean, mase_mean, direction_acc, coverage }],
  per_horizon: { model_name: [mae_h1 … mae_hH] }
}
beats_baseline: bool, skill_vs_naive: float
feature_importance: [{name, value}]      # 树模型入选时
references: [{season_id, name, points: [{h, median_price}]}]
warnings: [str]
```

### 6.9 旧模块的处置

`src/analysis/Machine_learning_analysis.py` 整体删除。它的命令行菜单功能（选物品、跑分析）由 `src/forecast/cli.py` 提供：`python -m src.forecast.cli --item "Iron Ore" --season s3 --horizon 72`。

## 7. 后端接口

所有响应统一信封：`{ "ok": bool, "data": ..., "error": str | null, "meta": {...} | null }`。参数用 Pydantic 校验，非法输入返回 422 并带字段级信息。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 数据源、各物品最新时间与行数、运行中任务 |
| GET | `/api/items` | 物品列表（来自 items_config.json 与仓储合并） |
| GET | `/api/series/{item}` | 查询参数：season_id/start/end、resolution=1h\|4h\|1d；返回聚合序列 |
| GET | `/api/listings/{item}` | 查询参数：season_id/start/end/sold/min_qty/max_qty/page/page_size；分页挂单 |
| GET/POST/PUT/DELETE | `/api/seasons` | 赛季 CRUD，校验不重叠 |
| POST | `/api/forecast` | 提交预测任务，返回 job_id |
| GET | `/api/forecast/runs` | 历史运行列表（摘要） |
| GET | `/api/forecast/runs/{run_id}` | 完整结果 |
| GET | `/api/forecast/runs/{run_id}/actuals` | 该次预测区间内后来的实际小时中位价，用于对比 |
| DELETE | `/api/forecast/runs/{run_id}` | 删除一次运行 |
| GET | `/api/jobs` | 任务列表 |
| GET | `/api/jobs/{id}` | 状态、结果引用、日志尾部 |
| GET | `/api/jobs/{id}/events` | SSE：日志行与状态变化 |
| POST | `/api/collect/{item}` | 启动采集子进程，返回 job_id |
| GET | `/api/scheduler` | 状态与任务列表 |
| POST | `/api/scheduler/start` `/stop` | 启停调度器子进程 |
| POST | `/api/scheduler/run` | 立即运行某任务，返回 job_id |
| GET/PUT | `/api/config/items` | items_config.json 读写，PUT 做结构校验 |
| GET | `/{path}` | 生产模式托管 `frontend/dist` |

任务模型：`Job{id, kind: forecast|collect|task, status: queued|running|succeeded|failed, created_at, finished_at, log: deque(maxlen=2000), result_ref}`。预测任务在线程池运行；采集与调度器任务是子进程，stdout 逐行进日志。SSE 用 `sse-starlette`。同一物品同一时刻只允许一个采集任务。

## 8. 前端

技术栈：React 18、TypeScript、Vite、react-router、TanStack Query、ECharts（echarts-for-react）、Tailwind CSS、i18next（中/英）。状态：服务端状态全部交给 TanStack Query，本地只保留选中的物品/赛季/范围（URL 查询参数持久化）。

页面：

1. **概览**：每物品卡片（最新小时中位价、24h 变化、最近采集时间、小型趋势线），数据源徽标，运行中任务条。
2. **市场**：顶部物品、赛季、时间范围、分辨率；主图：中位价折线 + p25–p75 带 + 最低最高细线，副图：成交量柱与挂单数；支持 dataZoom 和刷选；下方挂单表，可按数量、是否成交、价格过滤，分页，点击表格行在图上高亮对应小时。
3. **预测**：左栏历史运行列表（物品、赛季、步长、是否优于基线、时间）；主区：参数表单 → 运行 → 实时日志 → 结果。结果区：历史 + 预测 + 区间带图，可叠加参考赛季曲线，若有后来实际值则叠加对比；回测表格（模型 × 指标，基线行固定在顶）；分步长误差曲线；区间覆盖率；特征重要性条形图；"优于基线 / 未优于基线"的醒目状态；下载 CSV。
4. **采集**：物品卡片 + 立即采集，实时日志面板，最近任务列表。
5. **定时任务**：调度器状态、启停、任务表（名称、计划、启用、立即运行）。
6. **设置**：物品配置表单（名称、脚本、CSV、分类、启用）、赛季编辑器（表格 + 表单，冲突即时提示）、数据源状态、语言。

交互约定：所有写操作有即时反馈（toast）和失败原因；长任务通过 SSE 更新，不轮询；表单在前端和后端都校验；空状态和错误状态有明确文案。

## 9. 测试

后端与流水线（pytest，anaconda base 环境）：

- `aggregate`：固定挂单样本 → 期望小时行；缺口小时为 NaN。
- `features`：无同行泄露测试；滞后列与手算一致。
- `backtest`：合成序列（正弦 + 噪声）上 `seasonal_naive_24` 的 MAE 显著低于 `naive`；起点数量与步长正确。
- `intervals`：构造残差，验证分位数与覆盖率计算。
- `select`：集成不优于成员时回退单模型；无模型优于 naive 时 `beats_baseline=false`。
- `pipeline`：合成序列端到端跑通，输出 schema 完整，`HistGradientBoosting` 在含日内周期的合成序列上优于 naive。
- `seasons`：重叠校验、归属判断。
- `csv_repo`：过滤与分页。
- API：TestClient 覆盖每个路由的成功与 422 路径；SSE 读到日志行。

前端：vitest + testing-library 测关键组件（参数表单校验、回测表格渲染、赛季编辑冲突提示）与 API 客户端；Playwright 一条冒烟流程：打开预测页、选物品与赛季、运行、看到"优于基线/未优于基线"状态和图表。

覆盖率目标：`src/forecast` 与 `backend` 各 80% 以上。

## 10. 运行方式

- 依赖：`requirements.txt` 新增 fastapi、uvicorn、sse-starlette、pydantic-settings；移除 matplotlib、seaborn（不再绘图）。
- 开发：`uvicorn backend.main:app --reload --port 5001`；`cd frontend && npm run dev`（Vite 代理 `/api` 到 5001）。
- 生产：`npm run build` 后 `python -m backend`，单端口提供页面与接口。
- `.env` 继续提供数据库与 API key。`runs/` 加入 `.gitignore`。
- README 的"Web 控制台"和"机器学习分析"章节改写为新用法。

## 11. 风险与应对

- 每赛季数据约 10 天，小时点约 240 个，168 小时步长的回测起点会很少。应对：起点不足时自动缩短评估步长并在结果中告知；界面默认步长 72。
- Postgres 表名由物品名推导，物品名含特殊字符会出错。应对：仓储层做白名单校验，只允许字母、数字、空格。
- 采集脚本是子进程且依赖 anaconda 解释器。应对：后端用 `sys.executable` 启动，并在 `/api/status` 显示解释器路径，README 说明用 anaconda base 启动后端。
