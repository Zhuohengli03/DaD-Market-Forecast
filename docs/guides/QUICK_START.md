# Quick start in detail / 详细上手指南

## 1. Install / 安装
```bash
python -m pip install -r requirements.txt     # use a virtualenv or conda env if you prefer
cd frontend && npm install && npm run build && cd ..
```
`psycopg2-binary` is installed but only used if you configure Postgres.

## 2. Start / 启动
```bash
python -m backend        # http://localhost:5001
```
The first start creates `data/` with your tracked items (seeded from `items_config.json`), an empty season list, and a folder for forecast runs. Everything in `data/` and `.env` is ignored by git.

## 3. API key / 填写 API key
The app opens a dialog. Create a key at https://darkerdb.com/dashboard/api-keys with the `darkerdb.live` and `darkerdb.data` scopes and paste it. The key is checked against DarkerDB before it is saved. You can change it later in **Settings**.

## 4. Seasons / 赛季
**Settings → Seasons**: add the current season with its start time in UTC and an empty end. Add past seasons too if you have their data; they appear as dotted reference lines on forecasts.

## 5. Pick items / 选择道具
**Item catalog → Sync catalog** downloads every item and its parameters (rarity, type, gear score, vendor price, attribute ranges, everything DarkerDB returns). **Scan this season's market** reads the newest listings since the season start (40 pages by default, about a minute at DarkerDB's 60 requests/min limit) and marks which items are trading. Press **Track** on the items you want.

## 6. Collect and forecast / 采集与预测
**Collect data → Collect now** fetches listings for one item. A forecast needs at least 72 continuous hours of data in the selected season, so collect for a few days (or turn on **Scheduled collection**) before forecasting a new item.

## Troubleshooting / 排错
| Symptom | Fix |
|---|---|
| Dialog says the key was rejected | Use a v2 key with both scopes. v1 keys stopped working. |
| "need at least 72 contiguous hours" | Collect more data, or check the season start. Gaps longer than 6 hours cut the series. |
| Forecast shows the baseline | Normal when prices move randomly. The banner explains why. |
| `python -m backend` cannot import sklearn | Install the requirements into the same Python you run. |
| Port 5001 in use | `PORT=5050 python -m backend` |
