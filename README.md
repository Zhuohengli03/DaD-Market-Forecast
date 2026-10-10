# Darker Market

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Node](https://img.shields.io/badge/Node-20+-green.svg)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Track and forecast **Dark and Darker** marketplace prices on your own machine, with data from the [DarkerDB API](https://darkerdb.com/documentation).

在自己电脑上追踪和预测 **Dark and Darker（暗黑地牢）** 交易行价格，数据来自 [DarkerDB API](https://darkerdb.com/documentation)。

[中文说明](#中文说明) · [English](#english)

---

## English

### What it does
- **Item catalog**: every tradable item type from DarkerDB (all rarities of one name are one type) with all parameters. Items without random stats (ores, consumables, materials) can be tracked and forecast.
- **Equipment market**: for weapons, armor and accessories, add any number of stat conditions (e.g. armor penetration 2–5 and headshot bonus ≥ 3) and see matching listings, sales and daily prices, live from DarkerDB.
- **Price history**: hourly median price with the middle-50% band, volume, zoom, range statistics and a filterable listing table.
- **Forecast**: predicts the next 24–168 hours from the current season only. Every run first replays history and compares each method with "repeat the last hour". A model is used only when it is clearly better, and the page says so either way.
- **Collection**: collect on demand or once a day automatically. Data stays on your machine as CSV files in `data/`, or in Postgres if you configure one.

### Requirements
- Python 3.10+ and Node.js 20+
- A free DarkerDB account and API key ([create one here](https://darkerdb.com/dashboard/api-keys), scopes `darkerdb.live` and `darkerdb.data`)
- No database needed

### Quick start
```bash
git clone https://github.com/<you>/darker-market.git
cd darker-market
python -m pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python -m backend
```
Open **http://localhost:5001**. On first launch a dialog asks for your DarkerDB API key; it is verified and saved to `.env` in the project folder (never committed).

Then:
1. **Settings → Seasons**: press *Import seasons from DarkerDB*. The market resets every season, so forecasts never mix seasons.
2. **Item catalog**: press *Sync catalog*, then *Check tradable items* once (about 15 minutes, it hides items that never trade). *Track* stat-free items you care about; four ores are tracked by default. For equipment, use *View market*.
3. **Collect data**: press *Collect now* on an item (or turn on *Scheduled collection*).
4. **Forecast**: pick an item and a horizon, press *Run forecast*.

### How the forecast stays honest
- Listings become an hourly median price. Every feature looks only at earlier hours (a test proves it).
- Methods are scored by a rolling backtest against two baselines: repeat the last hour and repeat yesterday.
- A model is used only if it beats "repeat last hour" by more than 2% on at least 5 backtests. Otherwise the baseline is shown with the reason.
- Shaded intervals come from real backtest errors, and the page reports how often they actually contained the price.
- If your newest data is old, the page warns you that the forecast starts from then.

### Development
```bash
python -m pip install -r requirements-dev.txt   # adds pytest and pytest-cov
python -m backend                               # API on :5001
cd frontend && npm run dev                      # UI on :5173 with hot reload (proxies /api)
python -m pytest -q --cov=src --cov=backend     # backend and forecast tests with coverage
cd frontend && npm test                         # unit tests
npx playwright install chromium                 # once, before the first browser test run
PYTHON=python npm run e2e                       # browser tests on synthetic data, no key needed
```
`PYTHON` must point to the interpreter where you installed the requirements (for example a venv's `python`).
Command-line tools:
```bash
python -m src.api.collect --item "Iron Ore"            # collect one item
python -m src.api.collect --item "Iron Ore" --probe    # print one raw API row
python -m src.forecast.cli --item "Iron Ore" --season <season-id> --horizon 72
```
More: [Quick start details](docs/guides/QUICK_START.md) · [Configuration](docs/guides/ENV_SETUP_GUIDE.md) · [Architecture](docs/PROJECT_SUMMARY.md) · [API reference](docs/api/API_REFERENCE.md)

---

## 中文说明

### 功能
- **道具库**：DarkerDB 上所有可交易的道具类型（同名道具的各稀有度算一种），带全部参数。没有随机词条的道具（矿石、消耗品、材料）可以追踪和预测。
- **装备行情**：武器、防具、饰品可以添加任意多个词条条件（如破甲 2~5、爆头加成 ≥ 3），实时查看符合条件的挂单、成交和每日价格。
- **价格走势**：每小时中位价、中间 50% 价格带、成交量，可缩放、框选统计，挂单明细可筛选。
- **价格预测**：只用当前赛季的数据预测未来 24 到 168 小时。每次先在历史上回测，和"沿用上一小时价格"比较，只有明显更准才采用模型，并在页面上说明结果。
- **数据采集**：手动或每天定时采集。数据保存在本机 `data/` 目录的 CSV 里；配置了 Postgres 则存数据库。

### 需要准备
- Python 3.10+ 和 Node.js 20+
- DarkerDB 账号和 API key（[在这里创建](https://darkerdb.com/dashboard/api-keys)，勾选 `darkerdb.live` 和 `darkerdb.data`）
- 不需要安装数据库

### 快速开始
```bash
git clone https://github.com/<you>/darker-market.git
cd darker-market
python -m pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python -m backend
```
浏览器打开 **http://localhost:5001**。第一次打开会弹窗让你填写 DarkerDB API key，验证通过后保存到项目目录的 `.env`（不会被提交到 git）。

然后：
1. **设置 → 赛季**：点「从 DarkerDB 导入赛季」。每个赛季市场都会重置，预测不会跨赛季混用数据。
2. **道具库**：点「同步道具库」，再点一次「检查可交易道具」（约 15 分钟，会隐藏从来没人交易过的道具）。给关心的无词条道具点「追踪」，默认已追踪 4 种矿石。装备点「查看行情」。
3. **数据采集**：点道具上的"立即采集"，或在"定时采集"里开启每天自动采集。
4. **价格预测**：选道具和预测时长，点"开始预测"。

### 常见问题
- **弹窗说 key 无效**：确认是 v2 key，并勾选了上面两个权限。旧的 v1 key 已经不能用了。
- **预测说"没有找到比基准明显更准的方法"**：这是正常结果，说明这段时间价格接近随机波动，最好的猜测就是当前价格。
- **想让局域网里的手机访问**：在 `.env` 里设 `HOST=0.0.0.0`，只在可信网络下这样做。

## License
[MIT](LICENSE). Not affiliated with IRONMACE or DarkerDB. 与 IRONMACE 和 DarkerDB 无关联。
