# Item-centric UI and patch-note markers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "overview → one item page" the main path: every price, season, forecast and game-update fact about an item on one page, with Steam patch notes that name the item marked on its price chart.

**Architecture:** A new backend service downloads Steam patch announcements into `data/announcements.json` and matches item names in them. A new `/item/:archetype` page composes existing pieces (price chart, quality prices, next-hour range, item facts, listings) plus two new ones (season comparison, related updates). The overview card gains a 7-day sparkline, a next-hour range line and an update badge, and links to the item page.

**Tech Stack:** Python 3.10+, FastAPI, pandas, requests; React 18 + TypeScript, TanStack Query, ECharts, vitest, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-item-centric-ui-design.md`

## Global Constraints

- All UI text in Chinese (zh.ts) with an English twin (en.ts); patch snippets stay in English (source text).
- No DarkerDB credits spent by anything in this plan; Steam is free and keyless.
- Steam failures never break a page: fall back to the local cache and say "公告暂时无法更新".
- Times are UTC in data and charts.
- Don't commit or push unless the user asks; another tool edits this repo — touch only the files listed per task.
- Run the full suites at the end of each task: `/opt/anaconda3/bin/python -m pytest -q tests` and `cd frontend && npx tsc --noEmit -p . && npx vitest run`.

## Review Focus

1. A patch title that is not a patch (tournament results, events) must not become a chart marker.
2. Common words ("Arrow", "Ball", "Rope") match too much — matching is whole-word and case-insensitive; the page states the limit.
3. Season comparison when a season has no data, or only a few days: the line is short or absent, the table row shows "–", nothing crashes.
4. An untracked item opened from the catalog: the item page shows facts and "追踪" only, no empty chart errors.
5. Old links `/catalog/<archetype>` keep working (redirect), and the overview links to `/item/<archetype>`.

---

### Task 1: Steam patch announcements — download, store, match

**Files:**
- Create: `backend/services/announcements.py`
- Create: `backend/api/announcements.py`
- Modify: `backend/main.py` (store on state, router list, refresh at startup)
- Test: `tests/backend/test_announcements.py`

**Interfaces:**
- Produces: `AnnouncementStore(path)` with `.items() -> list[dict]`, `.updated_at() -> str | None`, `.merge(new: list[dict])`;
  `fetch_steam(get, since_ts: int | None) -> list[dict]` (`get(url, params) -> dict`, injectable);
  `is_patch(title: str) -> bool`; `plain_text(bbcode: str) -> str`;
  `mentions(name: str, items: list[dict]) -> list[dict]` → `{id, date, title, url, snippet}` newest first.
  Endpoints: `GET /api/announcements` (patches, newest first, `meta.updated_at`, `meta.error`),
  `GET /api/announcements/item/{archetype}` (mentions for that type's English name),
  `POST /api/announcements/refresh`.
  Announcement dict: `{"id": str, "date": ISO UTC str, "title": str, "url": str, "text": str}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backend/test_announcements.py
"""Steam patch notes: keep patches only, store them locally, find the ones that name an item."""
from backend.services.announcements import AnnouncementStore, fetch_steam, is_patch, mentions, plain_text

H = {"X-Darker-Market": "1"}


def _post(gid, date, title, contents):
    return {"gid": gid, "date": date, "title": title, "url": f"https://steam/{gid}", "contents": contents,
            "feedname": "steam_community_announcements"}


def test_only_patches_count():
    assert is_patch("Hotfix #126") and is_patch("Early Access Season #11") and is_patch("Patch Notes 0.17")
    assert not is_patch("EAS10 Warlord Tournament – Final Results") and not is_patch("Tavern Table #5 is now LIVE!")


def test_bbcode_becomes_plain_text():
    assert plain_text("[h1]Changes[/h1][list][*]Gold Ore now drops more[/list][img]x.png[/img]") == "Changes\n- Gold Ore now drops more"


def test_fetch_pages_back_and_keeps_patches():
    pages = [{"appnews": {"newsitems": [_post("3", 1791543651, "Early Access Season #11", "x"),
                                        _post("2", 1791338686, "Tavern Table #5 is now LIVE!", "y")]}},
             {"appnews": {"newsitems": [_post("1", 1780000000, "Hotfix #120", "Gold Ore price fix")]}},
             {"appnews": {"newsitems": []}}]
    calls = []

    def get(url, params):
        calls.append(params)
        return pages[len(calls) - 1]
    got = fetch_steam(get, since_ts=None)
    assert [g["id"] for g in got] == ["3", "1"] and got[1]["text"] == "Gold Ore price fix"
    assert calls[1]["enddate"] == 1791338686 - 1            # pages back from the oldest post seen


def test_store_merges_and_survives_a_bad_file(tmp_path):
    s = AnnouncementStore(tmp_path / "a.json")
    s.merge([{"id": "1", "date": "2026-01-01T00:00:00Z", "title": "Hotfix #1", "url": "u", "text": "t"}])
    s.merge([{"id": "1", "date": "2026-01-01T00:00:00Z", "title": "Hotfix #1", "url": "u", "text": "t"},
             {"id": "2", "date": "2026-02-01T00:00:00Z", "title": "Hotfix #2", "url": "u", "text": "t"}])
    assert [a["id"] for a in s.items()] == ["2", "1"] and s.updated_at()
    (tmp_path / "a.json").write_text("{broken")
    assert AnnouncementStore(tmp_path / "a.json").items() == []


def test_mentions_whole_words_any_case_and_plurals():
    items = [{"id": "1", "date": "2026-01-01T00:00:00Z", "title": "Hotfix #1", "url": "u",
              "text": "Intro.\nGOLD ORES now stack to 10.\nMarigold Oreo is not it."},
             {"id": "2", "date": "2026-02-01T00:00:00Z", "title": "Hotfix #2", "url": "u", "text": "Nothing here."}]
    m = mentions("Gold Ore", items)
    assert [x["id"] for x in m] == ["1"] and m[0]["snippet"] == "GOLD ORES now stack to 10."


def test_api_lists_patches_and_item_mentions(client, settings, monkeypatch):
    from backend.services import announcements as ann
    store = client.app.state.announcements
    store.merge([{"id": "9", "date": "2026-01-25T00:00:00Z", "title": "Hotfix #9", "url": "u", "text": "Test Ore drops doubled."}])
    assert client.get("/api/announcements").json()["data"][0]["id"] == "9"
    # Test Ore is tracked by name only (no catalog type): the archetype path falls back to the item name
    d = client.get("/api/announcements/item/Test Ore").json()["data"]
    assert d["name"] == "Test Ore" and d["mentions"][0]["snippet"] == "Test Ore drops doubled."
    monkeypatch.setattr(ann, "fetch_steam", lambda get, since_ts: (_ for _ in ()).throw(OSError("offline")))
    r = client.post("/api/announcements/refresh", headers=H).json()
    assert r["data"]["error"] and len(client.get("/api/announcements").json()["data"]) == 1   # cache kept
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/backend/test_announcements.py`
Expected: collection error, `backend.services.announcements` does not exist.

- [ ] **Step 3: Implement the service**

```python
# backend/services/announcements.py
"""Steam patch notes for Dark and Darker, kept locally, searched for item names (no DarkerDB credits)."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

STEAM_NEWS = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
APP_ID = 2016590
OLDEST = int(datetime(2024, 9, 1, tzinfo=timezone.utc).timestamp())   # season 4
REFRESH_HOURS = 6
_PATCH = re.compile(r"hotfix|patch|update|season", re.I)
_NOT_PATCH = re.compile(r"tournament|results|tavern table|giveaway|event", re.I)


def is_patch(title: str) -> bool:
    return bool(_PATCH.search(title)) and not _NOT_PATCH.search(title)


def plain_text(bbcode: str) -> str:
    s = re.sub(r"\[img\].*?\[/img\]", "", bbcode, flags=re.S | re.I)
    s = re.sub(r"\[\*\]", "\n- ", s)
    s = re.sub(r"\[/?(h\d|list|olist|p)\]", "\n", s, flags=re.I)
    s = re.sub(r"\[/?[a-z0-9]+(=[^\]]*)?\]", "", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    lines = [ln.strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_steam(get, since_ts: int | None) -> list[dict]:
    """Page back from the newest post until `since_ts` (or season 4); keep patch posts only."""
    out, end = [], None
    stop = since_ts or OLDEST
    while True:
        params = {"appid": APP_ID, "count": 100, "maxlength": 0, "format": "json",
                  "feeds": "steam_community_announcements"}
        if end is not None:
            params["enddate"] = end
        posts = (get(STEAM_NEWS, params).get("appnews") or {}).get("newsitems") or []
        if not posts:
            break
        for p in posts:
            if p["date"] < stop:
                return out
            if is_patch(p.get("title", "")):
                out.append({"id": str(p["gid"]), "date": _iso(p["date"]), "title": p["title"],
                            "url": p.get("url", ""), "text": plain_text(p.get("contents", ""))})
        end = min(p["date"] for p in posts) - 1
    return out


class AnnouncementStore:
    def __init__(self, path: Path):
        self._path = Path(path)

    def _read(self) -> dict:
        try:
            d = json.loads(self._path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def items(self) -> list[dict]:
        return sorted(self._read().get("items", []), key=lambda a: a["date"], reverse=True)

    def updated_at(self) -> str | None:
        return self._read().get("updated_at")

    def merge(self, new: list[dict]) -> None:
        by_id = {a["id"]: a for a in self._read().get("items", [])}
        by_id.update({a["id"]: a for a in new})
        data = {"updated_at": _iso(int(datetime.now(timezone.utc).timestamp())), "items": list(by_id.values())}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)


def mentions(name: str, items: list[dict]) -> list[dict]:
    """Patches whose text names the item: whole words, any case, plural allowed; the matching line as snippet."""
    words = r"\s+".join(re.escape(w) for w in name.split())
    pat = re.compile(rf"(?<![A-Za-z]){words}(e?s)?(?![A-Za-z])", re.I)
    out = []
    for a in items:
        line = next((ln for ln in a["text"].splitlines() if pat.search(ln)), None)
        if line:
            out.append({k: a[k] for k in ("id", "date", "title", "url")} | {"snippet": line.lstrip("- ").strip()})
    return sorted(out, key=lambda x: x["date"], reverse=True)


def refresh(store: AnnouncementStore, get) -> str | None:
    """Fetch posts newer than the newest stored; returns an error message or None."""
    newest = store.items()[0]["date"] if store.items() else None
    since = int(datetime.fromisoformat(newest.replace("Z", "+00:00")).timestamp()) if newest else None
    try:
        store.merge(fetch_steam(get, since))
        return None
    except Exception as exc:   # offline / Steam down: keep the cache
        return f"公告暂时无法更新（{type(exc).__name__}）"


def due(store: AnnouncementStore) -> bool:
    u = store.updated_at()
    if not u:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(u.replace("Z", "+00:00"))
    return age.total_seconds() > REFRESH_HOURS * 3600
```

- [ ] **Step 4: Implement the API and wire it**

```python
# backend/api/announcements.py
from fastapi import APIRouter, HTTPException, Request

from ..schemas import ok
from ..services import announcements as ann

router = APIRouter(tags=["announcements"])


def steam_get(url, params):
    import requests
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _type_name(st, archetype: str) -> str:
    """English name shared by a type's qualities (Ruby for every ruby), else the tracked item name itself."""
    from ..data.catalog import shared_name
    variants = st.catalog.types().get(archetype)
    if variants:
        return shared_name([v.get("name") or archetype for v in variants])
    if any(it["name"] == archetype for it in st.items.list()):
        return archetype
    raise HTTPException(404, f"没有这个道具 / unknown item {archetype}")


@router.get("/announcements")
def list_announcements(request: Request):
    store = request.app.state.announcements
    items = [{k: a[k] for k in ("id", "date", "title", "url")} for a in store.items()]
    return ok(items, meta={"updated_at": store.updated_at(), "error": getattr(request.app.state, "announcements_error", None)})


@router.get("/announcements/item/{archetype}")
def item_mentions(archetype: str, request: Request):
    st = request.app.state
    name = _type_name(st, archetype)
    return ok({"name": name, "mentions": ann.mentions(name, st.announcements.items())})


@router.post("/announcements/refresh")
def refresh(request: Request):
    st = request.app.state
    st.announcements_error = ann.refresh(st.announcements, steam_get)
    return ok({"error": st.announcements_error, "count": len(st.announcements.items())})
```

In `backend/main.py`:
- `ROUTER_MODULES`: append `"announcements"`.
- In `_attach_services`: `from .services.announcements import AnnouncementStore` and
  `st.announcements = AnnouncementStore(st.settings.data_dir / "announcements.json")`; `st.announcements_error = None`.
- In `lifespan`, after startup work: start a daemon thread that runs
  `ann.refresh(app.state.announcements, steam_get)` when `ann.due(app.state.announcements)`, storing the result in
  `app.state.announcements_error`. Skip when `settings` says tests (guard: only when `app.state.settings.frontend_dist` is set,
  the same condition the app uses for a real run — check `main.py` for the existing flag and reuse it).

- [ ] **Step 5: Run the tests**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/backend/test_announcements.py` → Expected: 6 passed.
Then the full backend suite → all pass.

- [ ] **Step 6: Try it live (no credits)**

Run: `curl -s -X POST -H "X-Darker-Market: 1" http://127.0.0.1:5001/api/announcements/refresh` after a restart,
then `curl -s http://127.0.0.1:5001/api/announcements/item/id.item.gold_ores` → Expected: a list of patches naming Gold Ore
(may be empty if none name it; check one snippet by eye).

---

### Task 2: Item page `/item/:archetype` composed of existing pieces

**Files:**
- Create: `frontend/src/pages/item/ItemPage.tsx`, `frontend/src/pages/item/ItemHeader.tsx`, `frontend/src/pages/item/PriceSection.tsx`
- Modify: `frontend/src/pages/catalog/ItemDetail.tsx` (export `ListingStats`, `VariantTable`; keep `Params` export)
- Modify: `frontend/src/App.tsx` (route + redirect), `frontend/src/pages/Market.tsx` (remove `NextHourRange`)
- Modify: `frontend/src/i18n/zh.ts`, `frontend/src/i18n/en.ts` (`item.*` keys)
- Test: `frontend/src/pages/item/ItemPage.test.tsx`

**Interfaces:**
- Consumes: `useCatalogType(archetype)`, `useItems()`, `useSeasons()`, `useSeries(item, {season_id, resolution})`,
  `useNextHour(item, seasonId)`, `RefreshData`, `QualityPrices`, `PriceChart`, `ListingsTable` (from `pages/market`), `Params`.
- Produces: `ItemPage` (default export) reading `:archetype` and URL params `q` (quality item name), `season`, `res`.
  `PriceSection({ item, seasonId, marks? })` — `marks` added in Task 4.

- [ ] **Step 1: Failing test**

```tsx
// frontend/src/pages/item/ItemPage.test.tsx
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'
import '../../i18n'
import * as hooks from '../../api/hooks'
import ItemPage from './ItemPage'

vi.mock('../../api/hooks')
vi.mock('echarts-for-react', () => ({ default: () => null }))
vi.mock('../market/RefreshData', () => ({ RefreshData: () => <span>refresh</span> }))
vi.mock('../market/ListingsTable', () => ({ ListingsTable: () => <span>listings</span> }))

const v = (id: string, rarity: string) => ({ id, rarity, gear_score: null, vendor_price: 3, icon_url: null, raw: { id, rarity } })
const ruby = { archetype: 'id.item.ruby', name: 'Ruby', name_zh: '红宝石', category: 'gem', item_type: 'misc', has_stats: false, icon_url: null,
  rarities: ['common', 'rare'], listings_total: 100, season_listings: 10, season_first: null, variants: [v('id.item.ruby_2001', 'common'), v('id.item.ruby_4001', 'rare')] }
const SEASONS = [{ id: 's10', name: 'Season 10', start: '2026-07-17T00:00:00Z', end: '2026-10-09T03:00:00Z', note: '' },
  { id: 's11', name: 'Season 11', start: '2026-10-09T03:00:00Z', end: null, note: '' }]

function mount(items: unknown[]) {
  for (const [k, val] of Object.entries({
    useCatalogType: { data: ruby, isLoading: false }, useItems: { data: items }, useSeasons: { data: SEASONS },
    useSeries: { data: { points: [] }, isLoading: false }, useNextHour: { data: undefined, isError: true, error: new Error('x') },
    useOverview: { data: [] }, useTrackItem: { mutate: vi.fn() }, useTypeHistory: { mutate: vi.fn() }, useCatalogMeta: { data: {} },
    useItemAnnouncements: { data: { name: 'Ruby', mentions: [] } }, useSeasonCompare: { data: { seasons: [] } }, useRuns: { data: [] },
  })) vi.mocked(hooks[k as keyof typeof hooks] as never as (...a: unknown[]) => unknown).mockReturnValue(val as never)
  render(<MemoryRouter initialEntries={['/item/id.item.ruby']}><Routes><Route path="/item/:archetype" element={<ItemPage />} /></Routes></MemoryRouter>)
}

it('puts one item on one page: header, quality switch, price chart, facts', () => {
  mount([{ name: 'Ruby (Normal) (common)', item_id: 'id.item.ruby_2001', rows: 5 }, { name: 'Ruby (Rare)', item_id: 'id.item.ruby_4001', rows: 2 }])
  expect(screen.getByRole('heading', { name: /红宝石/ })).toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: '品质' })).toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: '赛季' })).toHaveValue('s11')          // the latest season by default
  expect(screen.getByText('道具资料')).toBeInTheDocument()
})

it('an untracked item shows its facts and a track button only', () => {
  mount([])
  expect(screen.getAllByRole('button', { name: '追踪' }).length).toBeGreaterThan(0)
  expect(screen.queryByRole('combobox', { name: '赛季' })).not.toBeInTheDocument()
})
```

- [ ] **Step 2: Run, verify it fails** — `cd frontend && npx vitest run src/pages/item` → fails: `./ItemPage` not found
  (and `useItemAnnouncements` / `useSeasonCompare` are added in Tasks 3–4; add them now as hooks that call the
  endpoints so the mock names exist: see Task 3 Step 3 and Task 4 Step 3 for their bodies).

- [ ] **Step 3: Implement**

`ItemPage.tsx`:
- Read `archetype` from params; `row = useCatalogType(archetype).data`; `tracked = useItems().data.filter(i => row.variants.some(v => v.id === i.item_id))`.
- `[q, setQ] = useUrlState('q', '')` → `item = tracked.find(i => i.name === q) ?? tracked.sort(by rows desc)[0]`.
- `[season, setSeason] = useUrlState('season', '')` → default latest season by `start`.
- Layout (each block a `Card`), in this order: `ItemHeader`; if `item`: `PriceSection`, `NextHourRange` (only when season
  is the latest), `SeasonCompare` (Task 3), `RelatedUpdates` (Task 4), forecast summary (latest run for `item` from
  `useRuns()`: skill line + link `/forecast?run=<id>`); always: `<details>` "道具资料" with `QualityPrices`, `ListingStats`,
  `VariantTable`, `Params`; if `item`: `<details>` "挂单明细" with `ListingsTable`.
- Untracked: `ItemHeader` + `VariantTable` (its track buttons) + facts only.

`ItemHeader.tsx`: icon, `itemName(row)`, `typeLabel(row.item_type)`, quality `<select aria-label="品质">` (hidden for one
tracked quality), latest price / change from `useOverview()` group with `archetype`, `RefreshData` and one-click forecast
(move `quickForecast` from `Market.tsx` into a shared helper `pages/market/quickForecast.ts` and use it in both).

`PriceSection.tsx`: season `<select aria-label="赛季">`, resolution `<select aria-label="时间粒度">`, `useSeries`,
`PriceChart` with brush summary as on the Market page (move the summary helper next to `PriceChart` if it lives in Market).

`App.tsx`: `<Route path="/item/:archetype" element={<ItemPage />} />` and replace the catalog detail route with
`<Route path="/catalog/:archetype" element={<CatalogRedirect />} />` where
`function CatalogRedirect() { const { archetype } = useParams(); return <Navigate replace to={`/item/${encodeURIComponent(archetype!)}`} /> }`.
Keep `pages/catalog/ItemDetail.tsx` only as the module exporting `Params`, `ListingStats`, `VariantTable`; delete its
default page component and move its test cases for those pieces into `ItemPage.test.tsx` where they still apply.

`Market.tsx`: remove the `NextHourRange` block (it lives on the item page now).

i18n `item`: `{ quality: '品质', facts: '道具资料', listings: '挂单明细', forecast: '价格预测', latestRun: '最近一次预测：{{skill}}', noRun: '还没有预测，点上面的「一键预测」', fullForecast: '完整预测' }` (+ en).

- [ ] **Step 4: Run** — `npx vitest run src/pages/item` → 2 passed; full frontend suite + `tsc` → pass.

---

### Task 3: Season comparison

**Files:**
- Modify: `backend/services/series.py` (add `season_compare`)
- Modify: `backend/api/series.py` (`GET /series/{item}/seasons`)
- Create: `frontend/src/pages/item/SeasonCompare.tsx`
- Modify: `frontend/src/api/hooks.ts`, `frontend/src/api/types.ts`, i18n
- Test: `tests/backend/test_season_compare.py`, `frontend/src/pages/item/SeasonCompare.test.tsx`

**Interfaces:**
- Produces: `season_compare(repo, item, seasons: list[dict]) -> list[dict]`, one per season with data:
  `{"id","name","market_start","days": [{"day": int, "median": float}], "open","high","low","close","mean": float|None, "listings": int}`
  where `day 0` = the season's `market_start` (UTC date), daily medians of `price_per_unit` over sold listings.
  Hook `useSeasonCompare(item) -> { seasons: SeasonCompareRow[] }`.

- [ ] **Step 1: Failing backend test**

```python
# tests/backend/test_season_compare.py
import pandas as pd
from backend.services.series import season_compare


class Repo:
    def __init__(self, df): self.df = df
    def fetch_all(self, item, start, end):
        ts = pd.to_datetime(self.df.created_at, utc=True)
        m = ts >= start if end is None else (ts >= start) & (ts < end)
        return self.df[m]


def _rows(day0, days, price):
    return [{"price_per_unit": price + d, "quantity": 1, "has_sold": True, "price": price + d,
             "created_at": (pd.Timestamp(day0) + pd.Timedelta(days=d, hours=12)).isoformat()} for d in range(days)]


def test_seasons_line_up_by_day_since_trading_opened():
    df = pd.DataFrame(_rows("2026-01-10T00:00Z", 3, 10) + _rows("2026-03-01T00:00Z", 2, 20))
    seasons = [{"id": "a", "name": "A", "start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z", "market_open": "2026-01-10"},
               {"id": "b", "name": "B", "start": "2026-02-20T00:00:00Z", "end": None, "market_open": "2026-03-01"},
               {"id": "c", "name": "C", "start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"}]   # no data
    out = {s["id"]: s for s in season_compare(Repo(df), "X", seasons)}
    assert set(out) == {"a", "b"}
    assert out["a"]["days"] == [{"day": 0, "median": 10.0}, {"day": 1, "median": 11.0}, {"day": 2, "median": 12.0}]
    assert (out["a"]["open"], out["a"]["close"], out["a"]["high"], out["a"]["low"], out["a"]["listings"]) == (10.0, 12.0, 12.0, 10.0, 3)
    assert out["b"]["days"][0] == {"day": 0, "median": 20.0}
```

- [ ] **Step 2: Run, verify it fails** (`ImportError: season_compare`).

- [ ] **Step 3: Implement**

```python
# append to backend/services/series.py
def season_compare(repo, item: str, seasons: list[dict]) -> list[dict]:
    """Each season's daily median sale price, day 0 = the day trading opened, plus open/high/low/close/mean."""
    from src.season_calendar import market_start
    out = []
    for s in sorted(seasons, key=lambda x: x["start"]):
        start = pd.Timestamp(market_start(s))
        end = pd.Timestamp(s["end"]) if s.get("end") else None
        try:
            df = repo.fetch_all(item, start, end)
        except KeyError:
            continue
        sold = df[df["has_sold"].astype(bool)] if len(df) else df
        if sold.empty:
            continue
        ts = pd.to_datetime(sold["created_at"], utc=True)
        daily = sold["price_per_unit"].astype(float).groupby(ts.dt.floor("D")).median()
        day0 = start.floor("D")
        days = [{"day": int((d - day0) / pd.Timedelta(days=1)), "median": float(v)} for d, v in daily.items()]
        vals = daily.to_numpy()
        out.append({"id": s["id"], "name": s["name"], "market_start": start.isoformat(), "days": days,
                    "open": float(vals[0]), "close": float(vals[-1]), "high": float(vals.max()),
                    "low": float(vals.min()), "mean": float(vals.mean()), "listings": int(len(sold))})
    return out
```

```python
# backend/api/series.py — add
@router.get("/series/{item}/seasons")
def series_by_season(item: str, request: Request):
    from ..services.series import season_compare
    st = request.app.state
    return ok({"seasons": season_compare(st.repo, item_param(item), st.seasons.list())})
```

Hook: `export const useSeasonCompare = (item: string) => useQuery({ queryKey: ['series', 'seasons', item], enabled: !!item, queryFn: () => api.get<{ seasons: SeasonCompareRow[] }>(`/series/${item}/seasons`).then(r => r.data) })`.
Type: `export interface SeasonCompareRow { id: string; name: string; market_start: string; days: { day: number; median: number }[]; open: number; high: number; low: number; close: number; mean: number; listings: number }`.

`SeasonCompare.tsx`: ECharts line chart, `xAxis` value type named "赛季第几天", one series per season (`seasonLabel(name)`),
`data: days.map(d => [d.day, d.median])`; table below: 赛季 · 开盘 · 最高 · 最低 · 收盘 · 平均 · 挂单数 (`fmtNum`).
Empty: "还没有能对比的赛季数据（DarkerDB 的记录从 2026-07-31 开始）".

- [ ] **Step 4: Frontend test** (`SeasonCompare.test.tsx`, echarts mocked): renders one table row per season with
  open/close values; empty state text when `seasons: []`. Run → pass.

- [ ] **Step 5:** Mount `SeasonCompare` in `ItemPage` (between next-hour and updates). Full suites → pass.

---

### Task 4: Patch markers on the chart and the related-updates list

**Files:**
- Modify: `backend/api/announcements.py` (add before/after change to item mentions)
- Modify: `frontend/src/components/charts/PriceChart.tsx` (`marks` prop)
- Create: `frontend/src/pages/item/RelatedUpdates.tsx`
- Modify: `frontend/src/pages/item/PriceSection.tsx`, `ItemPage.tsx`, hooks, types, i18n
- Test: extend `tests/backend/test_announcements.py`; `frontend/src/components/charts/PriceChart.test.ts`; `frontend/src/pages/item/RelatedUpdates.test.tsx`

**Interfaces:**
- Consumes: `mentions()` (Task 1), `useAnnouncements()`, `useItemAnnouncements(archetype, item)`.
- Produces: `GET /api/announcements/item/{archetype}?item=<tracked name>` adds per mention `change3d: float | None` =
  median sale price of the 3 days after ÷ the 3 days before − 1 (`None` when either side has no sales).
  `buildPriceOption(points, highlightTs, L, marks?: { ts: string; label: string; strong: boolean }[])` — each mark
  snaps to the first point at or after `ts`; strong marks red `#dc2626`, others grey `#9ca3af` dashed.

- [ ] **Step 1: Failing tests**

```python
# add to tests/backend/test_announcements.py
def test_item_mentions_carry_the_price_change_around_the_update(client):
    client.app.state.announcements.merge([{"id": "9", "date": "2026-01-24T00:00:00Z", "title": "Hotfix #9", "url": "u",
                                            "text": "Test Ore drops doubled."}])
    d = client.get("/api/announcements/item/Test Ore", params={"item": "Test Ore"}).json()["data"]
    c = d["mentions"][0]["change3d"]
    assert c is not None and -0.5 < c < 0.5                     # fixture prices drift slowly
```

```ts
// frontend/src/components/charts/PriceChart.test.ts — add
it('marks game updates on the chart, snapped to the next point, strong ones in red', () => {
  const pts = ['2026-01-01T00:00:00Z', '2026-01-01T04:00:00Z', '2026-01-01T08:00:00Z'].map(ts => ({ ts, median_price: 1, min_price: 1, max_price: 1, p25: 1, p75: 1, listings: 1, sold: 1, volume: 1 }))
  const opt = buildPriceOption(pts, null, undefined, [{ ts: '2026-01-01T03:00:00Z', label: 'Hotfix #9', strong: true }, { ts: '2026-02-01T00:00:00Z', label: 'later', strong: false }])
  const median = opt.series.find(s => s.name === 'median') as { markLine: { data: { xAxis: string; lineStyle: { color: string } }[] } }
  expect(median.markLine.data).toEqual([expect.objectContaining({ xAxis: '2026-01-01T04:00:00Z', lineStyle: expect.objectContaining({ color: '#dc2626' }) })])
})
```

`RelatedUpdates.test.tsx`: given two mentions renders date, title, snippet with the item name in `<mark>`, a Steam link,
and change "+18%" / "–"; clicking a row calls `onPick(ts)`.

- [ ] **Step 2: Run, verify failures** (no `change3d`; `marks` ignored; component missing).

- [ ] **Step 3: Implement**
- Backend: in `item_mentions`, if `item` query given, `df = st.repo.fetch_all(item, date-3d, date+3d)` over sold rows;
  `before = median(created_at < date)`, `after = median(created_at >= date)`; `change3d = after/before - 1` when both exist.
- `buildPriceOption`: build `markData` from `marks` (snap with `points.findIndex(p => p.ts >= m.ts)`, drop when -1),
  merge with the existing highlight mark into the median series' `markLine.data`, each with its own `lineStyle` and
  `label: { formatter: m.label, position: 'insideEndTop', fontSize: 10 }`. `PriceChart` passes `marks` through.
- `PriceSection` gets `marks` from `useAnnouncements()` (all patches inside the series range, `strong` when the id is in
  this item's mentions) and a `focusTs` prop that sets `highlightTs`.
- `RelatedUpdates`: list as specified; footer note "只找补丁说明里直接点名的更新；名字常见的道具可能误匹配；片段为英文原文。"
  and, when `useAnnouncements().meta.error`, "公告暂时无法更新".
- Hooks: `useAnnouncements()` (returns `{ items, updatedAt, error }` from data + meta) and
  `useItemAnnouncements(archetype, item?)`.

- [ ] **Step 4: Run** the three test files, then full suites → pass.

---

### Task 5: Overview — 7-day sparkline, next-hour line, update badge, link to item page

**Files:**
- Modify: `backend/services/overview.py`, `src/forecast/next_hour.py` (`check` switch)
- Modify: `frontend/src/pages/dashboard/TypeCard.tsx`, `frontend/src/api/types.ts`, i18n
- Test: `tests/backend/test_overview.py` (extend), `frontend/src/pages/Dashboard.test.tsx` (extend)

**Interfaces:**
- Consumes: `next_hour_range(listings, now, check=False)`, `ann.mentions`.
- Produces: `SPARK_HOURS = 168` with the sparkline downsampled to 4-hour medians (42 points);
  per variant `next_hour: {p25, p75} | None`; per group `recent_update: {title, date} | None` (a mention within 14 days of
  the group's newest listing). `groupLink(g)` → `/item/<archetype>` (name-only items keep `/market?item=`).

- [ ] **Step 1: Failing tests**

```python
# tests/backend/test_overview.py — add
def test_overview_card_has_a_week_a_next_hour_range_and_recent_updates(client):
    client.app.state.announcements.merge([{"id": "9", "date": "2026-01-27T00:00:00Z", "title": "Hotfix #9", "url": "u",
                                            "text": "Test Ore drops doubled."}])
    g = client.get("/api/overview").json()["data"]["groups"][0]
    v = g["variants"][0]
    assert len(v["spark"]) == 42
    assert v["next_hour"] and v["next_hour"]["p25"] < v["next_hour"]["p75"]
    assert g["recent_update"]["title"] == "Hotfix #9"
```

```python
# tests/forecast/test_next_hour.py — add
def test_the_hit_rate_check_can_be_skipped():
    out = next_hour_range(_sales(), now=T0 + pd.Timedelta(hours=240), check=False)
    assert out["check"] == {"mid": None, "wide": None, "hours": 0}
```

Dashboard test: a card shows "下一小时：17–23" and "Hotfix #9" badge and links to `/item/id.item.ruby`.

- [ ] **Step 2: Run, verify failures.**

- [ ] **Step 3: Implement**
- `next_hour_range(..., check: bool = True)`: skip `_check` when False.
- Overview: fetch window `last - (168 + 24)h` (bounded by the season start); `summarize` keeps `latest`/`change24` from the
  hourly series and builds the sparkline from `hourly.resample("4h").median()` over the last 168 h, padded to 42;
  `next_hour` from `next_hour_range(df_last_3_days, check=False)` in a `try` (None on `InsufficientDataError`);
  `recent_update` from `ann.mentions(group English name, st.announcements.items())` filtered to the 14 days before the
  newest listing (or now).
- `TypeCard`: line `t('dashboard.nextHour', { lo, hi })` under the price when `next_hour`; a `Badge` with the update title
  (title attribute shows the date); link via the new `groupLink`.
- i18n `dashboard.nextHour: '下一小时：{{lo}}–{{hi}}'` (en `'Next hour: {{lo}}–{{hi}}'`).

- [ ] **Step 4: Run** tests → pass; full suites → pass.

---

### Task 6: Navigation, holdout page, links

**Files:**
- Modify: `frontend/src/components/layout/Shell.tsx`, `frontend/src/components/layout/Shell.test.tsx`
- Create: `frontend/src/pages/Holdout.tsx`; Modify: `frontend/src/pages/Forecast.tsx`, `frontend/src/App.tsx`
- Modify: catalog table row link → `/item/<archetype>`; i18n `nav.holdout: '赛季回测'`
- Test: Shell test order; `frontend/src/App.test.tsx` (new) for the `/catalog/:archetype` redirect

- [ ] **Step 1: Failing tests** — Shell: links in order
  `['概览', '价格走势', '价格预测', '赛季回测', '行情报告', '装备行情', '道具库', '数据采集', '定时采集', '本机数据库', '设置']`
  with group labels "② 查看与分析" before "① 获取数据". App: rendering `/catalog/id.item.ruby` lands on `/item/id.item.ruby`.
- [ ] **Step 2: Run, verify failures.**
- [ ] **Step 3: Implement** — reorder `GROUPS`; `Holdout.tsx` = `Card` with `SeasonHoldout` (moved out of `Forecast.tsx`);
  route `/holdout`; `CatalogTable` `go(archetype)` → `/item/…`.
- [ ] **Step 4: Run** full suites → pass. Then by hand in the browser (restart the backend with
  `curl -s -X POST -H "X-Darker-Market: 1" http://127.0.0.1:5001/api/setup/restart`): overview → 钴矿 card → item page;
  switch to 第 10 赛季; season comparison; related updates; `/catalog/id.item.cobalt_ores` redirects.
