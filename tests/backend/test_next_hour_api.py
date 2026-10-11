def test_next_hour_range_for_an_item(client):
    d = client.get("/api/next-hour/Test Ore", params={"season_id": "s2"}).json()
    assert d["ok"], d
    p = d["data"]["prices"]
    assert p["p10"] < p["p25"] < p["p50"] < p["p75"] < p["p90"]
    assert {"mid", "wide", "hours"} <= set(d["data"]["check"]) and d["data"]["stale_hours"] > 24   # fixture data is old


def test_next_hour_without_enough_sales_says_why(client):
    r = client.get("/api/next-hour/Test Ore", params={"season_id": "s1"})       # no listings in season 1
    assert r.status_code == 409 and "采集" in r.json()["error"]
    assert client.get("/api/next-hour/Nope Ore").status_code == 404


def test_an_earlier_season_uses_its_own_last_days(client):
    from backend.schemas import SeasonIn
    st = client.app.state
    s2 = next(s for s in st.seasons.list() if s["id"] == "s2")
    # season 2 now ends two days before the item's newest listing, which belongs to the next season
    st.seasons.update("s2", SeasonIn(name=s2["name"], start=s2["start"], end="2026-01-26T00:00:00+00:00", note=""))
    import sqlite3
    con = sqlite3.connect(st.settings.sqlite_path)     # a listing weeks later, outside season 2
    con.execute("INSERT INTO listings VALUES ('test_ore','late1','Test Ore',1,20.0,20.0,1,'2026-03-15T00:00:00Z','2026-03-15T00:00:00Z')")
    con.commit(); con.close()
    st.item_stats.invalidate()
    r = client.get("/api/next-hour/Test Ore", params={"season_id": "s2"}).json()
    assert r["ok"], r
    assert r["data"]["next_hour"] <= "2026-01-26T00:00:00+00:00"


def test_before_the_market_opens_it_says_when(client):
    import pandas as pd
    from backend.schemas import SeasonIn
    st = client.app.state
    s2 = next(s for s in st.seasons.list() if s["id"] == "s2")
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    st.seasons.update("s2", SeasonIn(name=s2["name"], start=s2["start"], end=None, note="", market_open=future))
    r = client.get("/api/next-hour/Test Ore", params={"season_id": "s2"})
    assert r.status_code == 409 and future in r.json()["error"]
