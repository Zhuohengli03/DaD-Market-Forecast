# Configuration reference / 配置说明

All settings are environment variables; the app also reads `.env` in the project root. Copy `env.template` to start, or let the first-run dialog create `.env` for you.

| Variable | Default | Meaning |
|---|---|---|
| `DARKER_MARKET_API_KEY` | empty | DarkerDB v2 key (`darkerdb.live` + `darkerdb.data`). Set through the app's dialog or Settings page. |
| `DB_USER` | empty | Leave empty to use the built-in SQLite database `data/market.db`. Set it to use Postgres. |
| `DB_HOST` / `DB_PORT` / `DB_DATABASE` / `DB_PASSWORD` | `localhost` / `5432` / `darkerdb` / empty | Postgres connection, used only when `DB_USER` is set. |
| `HOST` | `127.0.0.1` | Address the web server binds to. `0.0.0.0` exposes it to your network. |
| `PORT` | `5001` | Web server port. |
| `DARKER_DATA_DIR` | `./data` | Where collected CSVs and user state live. (The older `DATA_DIR` variable is ignored.) |

## Files the app manages / 程序自己维护的文件
| Path | Content |
|---|---|
| `.env` | your API key (mode 600) |
| `data/items.json` | tracked items and their daily collection time |
| `data/seasons.json` | season start/end times |
| `data/catalog.json` | synced DarkerDB catalog and last market scan |
| `data/market.db` | built-in database with all collected listings (when not using Postgres) |
| `data/settings.json` | setup-wizard choices: collection interval, auto-collect |
| `data/runs/*.json` | saved forecasts |

## Using Postgres / 使用 Postgres
Create a database, set the `DB_*` variables, and restart. Collectors create one table per item (`iron_ore`, `arming_sword`, …). If Postgres is unreachable, the app falls back to the built-in database and says so in Settings. Older CSV exports are imported into the built-in database once, automatically.
