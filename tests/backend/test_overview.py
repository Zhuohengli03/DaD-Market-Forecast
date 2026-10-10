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
    assert v["rows"] > 0 and v["latest"] is not None and len(v["spark"]) == 48
    assert v["change24"] is not None                         # fixture has 200 hours of data
    assert 0 < v["season_rows"] <= v["rows"] and test_ore["season_rows"] == v["season_rows"]   # since the latest season start


def test_change_is_against_the_price_a_day_before_the_latest():
    from backend.services.overview import summarize
    idx = pd.date_range("2026-10-01", periods=30, freq="h", tz="UTC", name="ts")
    hourly = pd.DataFrame({"median_price": [100.0] * 5 + [None] * 24 + [150.0]}, index=idx)
    s = summarize(hourly)
    assert s["latest"] == 150.0 and s["change24"] == 0.5 and len(s["spark"]) == 48
    assert summarize(hourly.iloc[-3:])["change24"] is None   # under a day of data
