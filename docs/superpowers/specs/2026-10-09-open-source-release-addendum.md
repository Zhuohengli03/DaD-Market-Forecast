# Addendum: open-source release, item catalog, API-key onboarding

Date: 2026-10-09. Extends `2026-10-09-forecast-rebuild-and-web-app-design.md`.

## Audience
People who clone the repo from GitHub. They have Python and Node, a DarkerDB account, and no Postgres.
Every page and message is written for them: plain words first, jargon explained in place.

## Decisions
1. **User state lives in `data/` (git-ignored).** `data/items.json`, `data/seasons.json`, `data/catalog.json`,
   `data/runs/`, `data/<item>.csv`. On first start, `items_config.json` (tracked default) seeds `data/items.json`;
   an existing root `seasons.json` / root CSVs / `runs/` are still read so the maintainer's data keeps working.
2. **Postgres is optional.** Collectors write to Postgres when `DB_USER` is set and reachable, otherwise append
   to `data/<item>.csv`. The web app reads whichever exists (unchanged repository fallback).
3. **API key onboarding.** `GET /api/setup` reports whether a key is configured. `PUT /api/setup/api-key`
   validates the key against `/v2/items?limit=1`, then writes `DARKER_MARKET_API_KEY` into `.env`
   (atomic, mode 600). The frontend shows a blocking-but-dismissable dialog when no key is configured, and a
   key card in Settings (masked, last 4 chars).
4. **Item catalog.** DarkerDB v2 exposes no tradable flag and no "items on market" list. So:
   - *Sync catalog*: page `/v2/items?limit=200` to the end, store every raw field, stamp the current patch
     (`/v2/patches/current`).
   - *Scan market*: walk the newest `/v2/market` pages since the latest season start (or 7 days), bounded by a
     page budget, and record per item: listings, min / median unit price, last seen. Items seen are "tradable
     this season".
   - The Catalog page shows all parameters (raw key/value drawer), filters by name / rarity / type / tradable,
     icons from the keyless `/v2/items/{id}/icon`, and a "Track" button that adds the item to `data/items.json`.
5. **Tracked items are generic.** An entry is `{name, item_id, category, description, enabled}`; `file`/`csv`
   are optional legacy fields. Collection always goes through `src/api/collect.py --item-id`.
   The scheduler builds its task list from enabled tracked items.
6. **Seasons ship empty.** Pages that need a season guide the user to Settings.
7. **Server binds 127.0.0.1**, internal errors return a reference id instead of exception text.

## Unverified until a valid key exists
Field names of `/v2/items` and `/v2/market` rows beyond those the docs name. Code reads them defensively
(`item_id` | `item.id`, `name` | `item.name`) and keeps raw rows so nothing is lost.

## Revision 2026-10-10 (after live API access)
User clarified: "items" means item **types** (DarkerDB archetypes: every rarity of one name is one type).
1. Catalog lists types (798), grouped locally from the 2432 synced variants by `archetype`.
2. Only tradable types are shown, with a note saying so. DarkerDB's `is_tradable` is true for every item, so a
   background check asks `/v2/market?archetype=<type>&limit=1` once per type (~15 min at 60 req/min, resumable);
   types with zero listings ever are hidden. Until the check runs, all types show with a notice.
3. Types without random stats (no variant has `num_secondary_attributes > 0`; ores, consumables, materials) keep
   tracking, collection, price history and forecasts. Tracking is per rarity variant.
4. Types with random stats get an **Equipment market** page instead of forecasts: choose rarity, add any number of
   stat filters (each a min/max range, primary or secondary), optionally only sold listings; the backend queries
   DarkerDB live (`archetype`, `rarity`, `has_sold`, `primary[x]`, `secondary[x]` = `a:b` / `>=a` / `<=b`) and returns
   matching listings plus daily price statistics. Stat labels come from `/v2/attributes`.
5. "Sold" is DarkerDB's inference (listing vanished before expiry); the UI says so.
6. Season boundaries can be imported from `/v2/patches?kind=season`.
