"""Dashboard overview: one group per item type, each tracked quality with its latest price, 24h change and sparkline."""
import pandas as pd

from tests.backend.test_round5 import GEMS_AND_POTIONS

H = {"X-Darker-Market": "1"}


def test_overview_groups_qualities_of_one_type(client, settings):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    CatalogStore(settings.catalog_path).save_items(GEMS_AND_POTIONS, "S11")
    st = client.app.state
    st.items.track("Ruby (Cracked) (poor)", "id.item.ruby_1001", "misc")
    st.items.track("Ruby (Flawed) (common)", "id.item.ruby_2001", "misc")
    d = client.get("/api/overview").json()["data"]
    groups = {g["key"]: g for g in d["groups"]}
    ruby = groups["id.item.ruby"]
    assert ruby["name_zh"] == "红宝石" and [v["label"] for v in ruby["variants"]] == ["裂开", "瑕疵"]
    assert all(v["rows"] == 0 and v["latest"] is None for v in ruby["variants"])
    # a legacy item without a catalog id is its own group
    test_ore = groups["Test Ore"]
    v = test_ore["variants"][0]
    assert v["rows"] > 0 and v["latest"] is not None and len(v["spark"]) == 42
    assert v["change24"] is not None                         # fixture has 200 hours of data
    assert 0 < v["season_rows"] <= v["rows"] and test_ore["season_rows"] == v["season_rows"]   # since the latest season start


def test_change_is_against_the_price_a_day_before_the_latest():
    from backend.services.overview import summarize
    idx = pd.date_range("2026-10-01", periods=30, freq="h", tz="UTC", name="ts")
    hourly = pd.DataFrame({"median_price": [100.0] * 5 + [None] * 24 + [150.0]}, index=idx)
    s = summarize(hourly)
    assert s["latest"] == 150.0 and s["change24"] == 0.5 and len(s["spark"]) == 42
    assert summarize(hourly.iloc[-3:])["change24"] is None   # under a day of data


def test_overview_card_has_a_week_a_next_hour_range_and_recent_updates(client):
    client.app.state.announcements.merge([{"id": "9", "date": "2026-01-27T00:00:00Z", "title": "Hotfix #9", "url": "u",
                                            "text": "Test Ore drops doubled."}])
    g = client.get("/api/overview").json()["data"]["groups"][0]
    v = g["variants"][0]
    assert len(v["spark"]) == 42
    assert v["next_hour"] and v["next_hour"]["p25"] < v["next_hour"]["p75"]
    assert g["recent_update"]["title"] == "Hotfix #9"


def test_a_season_without_data_shows_no_price_until_that_season_is_chosen(client):
    from backend.schemas import SeasonIn
    st = client.app.state
    s2 = st.seasons.get("s2")
    # season 2 (holding the fixture's listings) ended; season 3 started later and has none yet
    st.seasons.update("s2", SeasonIn(name=s2["name"], start=s2["start"], end="2026-02-01T00:00:00+00:00", note=""))
    st.seasons.add(SeasonIn(name="Season 3", start="2026-03-01T00:00:00+00:00", end=None, note=""))
    d = client.get("/api/overview").json()["data"]
    g = d["groups"][0]
    v = g["variants"][0]
    assert d["season"]["name"] == "Season 3"                           # the current season by default
    assert g["season_rows"] == 0 and v["latest"] is None and v["next_hour"] is None and v["rows"] > 0
    assert "past" not in v                                              # no earlier season's price stands in
    # the season that holds the listings can be chosen explicitly
    d = client.get("/api/overview", params={"season_id": "s2"}).json()["data"]
    v = d["groups"][0]["variants"][0]
    assert d["season"]["id"] == "s2" and v["season_rows"] == v["rows"] > 0 and v["latest"] is not None
    assert client.get("/api/overview", params={"season_id": "nope"}).status_code == 404


def test_an_ended_season_stops_at_its_end(client):
    """Listings after a season's end (the next season's) do not leak into its cards."""
    from backend.schemas import SeasonIn
    st = client.app.state
    s2 = st.seasons.get("s2")
    cut = (pd.Timestamp(s2["start"]) + pd.Timedelta(hours=100)).isoformat()
    st.seasons.update("s2", SeasonIn(name=s2["name"], start=s2["start"], end=cut, note=""))
    st.seasons.add(SeasonIn(name="Season 3", start=cut, end=None, note=""))
    older = client.get("/api/overview", params={"season_id": "s2"}).json()["data"]["groups"][0]["variants"][0]
    newer = client.get("/api/overview").json()["data"]["groups"][0]["variants"][0]
    assert 0 < older["season_rows"] < older["rows"] and 0 < newer["season_rows"] < newer["rows"]
    assert older["season_rows"] + newer["season_rows"] == older["rows"]
    assert pd.Timestamp(older["latest_ts"]) < pd.Timestamp(cut) <= pd.Timestamp(newer["latest_ts"])


def test_a_cards_next_hour_range_says_how_old_its_data_is(client):
    v = client.get("/api/overview").json()["data"]["groups"][0]["variants"][0]
    assert v["next_hour"]["stale_hours"] > 24                        # the fixture's listings are from January


def test_the_latest_price_says_which_hour_and_how_many_listings():
    from backend.services.overview import summarize
    idx = pd.date_range("2026-10-09", periods=3, freq="h", tz="UTC", name="ts")
    hourly = pd.DataFrame({"median_price": [129.5, 104.165, 148.0], "listings": [37, 52, 51]}, index=idx)
    s = summarize(hourly)
    assert s["latest"] == 148.0 and s["latest_ts"] == "2026-10-09T02:00:00+00:00" and s["latest_listings"] == 51


def test_all_items_mode_shows_each_items_newest_trading_season(client):
    """The overview lists every item at its newest season that has trading data, labelled with that season."""
    from backend.schemas import SeasonIn
    st = client.app.state
    s2 = st.seasons.get("s2")
    st.seasons.update("s2", SeasonIn(name=s2["name"], start=s2["start"], end="2026-02-01T00:00:00+00:00", note=""))
    st.seasons.add(SeasonIn(name="Season 3", start="2026-03-01T00:00:00+00:00", end=None, note=""))
    d = client.get("/api/overview", params={"all": True}).json()["data"]
    v = d["groups"][0]["variants"][0]
    assert v["latest"] is not None and v["season_id"] == "s2" and v["season_name"] == s2["name"]
    assert v["season_rows"] == v["rows"] > 0
    # the default (one season) is unchanged: the current season, no price from another
    assert client.get("/api/overview").json()["data"]["groups"][0]["variants"][0]["latest"] is None


def test_all_items_mode_ignores_records_from_before_a_market_opened(client):
    s = client.app.state.seasons.get("s2")
    client.put("/api/seasons/s2", json={**s, "market_open": "2026-02-01"})   # the fixture's hours all precede it
    v = client.get("/api/overview", params={"all": True}).json()["data"]["groups"][0]["variants"][0]
    assert v["latest"] is None and v["season_id"] is None
