# Darker Market

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Node](https://img.shields.io/badge/Node-20+-green.svg)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![爱发电](https://img.shields.io/badge/%E7%88%B1%E5%8F%91%E7%94%B5-Sponsor-946ce6.svg)](https://afdian.com/a/felixcode)

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
- No database to install: a built-in SQLite database (`data/market.db`) is created automatically; Postgres is optional

### Quick start
```bash
git clone https://github.com/<you>/darker-market.git
cd darker-market
./start.sh            # Windows: double-click start.bat
```
The script checks Python 3.10+ and Node.js 20+, creates a private Python environment (`.venv`), installs everything, builds the pages, starts the app and opens **http://localhost:5001**. The first run takes a few minutes; later runs start in seconds and only reinstall what changed.

A setup wizard then walks you through four steps:
1. **Environment**: checks Python and lets you keep data in the built-in database (recommended, created automatically) or a Postgres database.
2. **API key**: paste your DarkerDB key; it is verified and saved to `.env` (never committed).
3. **Initialize**: pick items to track; the app syncs the item catalog, imports seasons, collects this season's listings and tries a first forecast, showing progress for each item.
4. **Auto-collect**: choose how often to collect, in days/hours/minutes/seconds (minimum 30 seconds). Set it according to your DarkerDB API plan: requests over your quota fail. Collection resumes automatically every time you start the app.

Change the interval later in **Settings**, or run the wizard again from there.

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
- 不需要安装数据库：自动创建内置的 SQLite 数据库（`data/market.db`），Postgres 可选

### 快速开始
```bash
git clone https://github.com/<you>/darker-market.git
cd darker-market
./start.sh            # Windows：双击 start.bat
```
脚本会检查 Python 3.10+ 和 Node.js 20+，建立独立的 Python 环境（`.venv`）、安装依赖、构建页面、启动程序并打开 **http://localhost:5001**。第一次需要几分钟，之后几秒就能启动，只有内容变化时才重新安装或构建。

打开页面后，设置向导会带你完成四步：
1. **环境**：检查 Python，选择数据存在内置数据库（推荐，自动创建）还是 Postgres。
2. **API key**：粘贴你的 DarkerDB key，验证通过后保存到 `.env`（不会被提交到 git）。
3. **初始化**：选择要追踪的道具，程序会同步道具库、导入赛季、采集本赛季的挂单并尝试第一次预测，每个道具单独显示进度。
4. **自动采集**：按天/时/分/秒设置多久采集一次（最少 30 秒）。请按你的 DarkerDB API 订阅方案设置，超出额度的请求会失败。之后每次启动程序都会自动恢复采集。

之后可以在「设置」里修改采集间隔，或重新运行设置向导。

### 常见问题
- **弹窗说 key 无效**：确认是 v2 key，并勾选了上面两个权限。旧的 v1 key 已经不能用了。
- **预测说"没有找到比基准明显更准的方法"**：这是正常结果，说明这段时间价格接近随机波动，最好的猜测就是当前价格。
- **想让局域网里的手机访问**：在 `.env` 里设 `HOST=0.0.0.0`，只在可信网络下这样做。

## Support / 支持这个项目

如果 Darker Market 对你有帮助，可以在 [爱发电](https://afdian.com/a/felixcode) 请作者喝一杯麦酒，微信和支付宝都能付款。赞助完全自愿，软件本身一直免费。

If Darker Market saves you time, you can support it on [afdian](https://afdian.com/a/felixcode), a Chinese creator-support site (similar to Patreon). Sponsorship is optional; the app is and stays free.

## License
[MIT](LICENSE). Not affiliated with IRONMACE or DarkerDB. 与 IRONMACE 和 DarkerDB 无关联。
