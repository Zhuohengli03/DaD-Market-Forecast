# Darker Market guide

Everything is configured inside the app; no code, config files or terminal needed.

## Install and start
1. Install [Python 3.10+](https://www.python.org/downloads/) and [Node.js 20+](https://nodejs.org/).
2. Download this project (**Code → Download ZIP** on GitHub, or clone it).
3. Run `start.sh` on macOS / Linux, or double-click `start.bat` on Windows.

The first start installs dependencies and takes a few minutes; later starts take seconds. Your browser opens **http://localhost:5001**.

## Setup wizard
It opens on first use; reopen it any time from **Settings → Run the setup wizard again**.

| Step | What happens |
|---|---|
| Environment | Keep data in the built-in database (recommended, created for you) or your own Postgres |
| API key | Paste your DarkerDB key ([create one](https://darkerdb.com/dashboard/api-keys) with `darkerdb.live` and `darkerdb.data`) |
| Initialize | Pick items to track; the app downloads the catalog, imports seasons, collects this season and tries a first forecast |
| Auto-collect | How often to collect (30 seconds minimum), with an estimate of the DarkerDB credits it uses per month |

## Pages
| Page | For |
|---|---|
| Overview | One card per item type with every quality together: latest price, 24h change, 7-day trend and next hour's sale range; each card can switch season by itself |
| Item page | Open it from an overview card or the catalog. The header picks the season; below: price chart (with the game updates that name the item), next hour's sale range, any two seasons compared, related updates, one-click forecast and its full result |
| Price history | Any time range, with every listing |
| Forecasts | Items forecast in a season; click one to see that forecast |
| Season holdout | Test the forecaster on an ended season |
| Market reports | Weekly reports written with Alibaba Cloud Bailian |
| Equipment market | Filter gear by stat ranges and see listings and daily prices, live |
| Item catalog | Every tradable item; click a row for its item page |
| Collection | Collect now on top (add or remove items, collect all, by season, only those without data), scheduled collection below (on/off, default interval, which items, their own times) |
| Local database | Look at and delete the listings stored here; no editing |
| Settings | API key, DarkerDB credits, AI analysis, seasons, network (LAN access and port), language |

## DarkerDB credits
The free plan has 10,000 a month. Credits left show in the bottom-left corner; every action that calls DarkerDB shows its (rounded-up) cost first and asks before going over. Task history shows what each run actually used.

## Why the forecast can be trusted
- Listings become an hourly median; every feature looks only at earlier hours (a test proves it).
- Methods are backtested against two baselines: repeat the last hour, and repeat yesterday.
- A model is used only if it beats the baseline by more than 2% over at least 5 backtests; otherwise the baseline is shown with the reason.
- Intervals come from real backtest errors, and the page reports how often they held.

## FAQ
- **"Invalid key"**: use a v2 key with both scopes above.
- **"Get latest data" finds nothing new**: DarkerDB has nothing newer yet; the page shows when its newest listing is from.
- **"No method clearly beats the baseline"**: prices are close to random right now, so the current price is the best guess.
- **Use it from your phone**: Settings → Network, allow LAN access, then Restart now. Only on a trusted network.
- **Port already in use**: pick another in Settings → Network, then Restart now.

## Developers
```bash
python -m pip install -r requirements-dev.txt
python -m backend                               # API on :5001
cd frontend && npm run dev                      # UI on :5173 with hot reload
python -m pytest -q                             # backend and forecast tests
cd frontend && npm test && npm run e2e          # unit and browser tests
python tools/readme_banner.py                   # regenerate the README animation
```
More: [Configuration reference](guides/ENV_SETUP_GUIDE.md) · [Architecture](PROJECT_SUMMARY.md) · [API reference](api/API_REFERENCE.md)
