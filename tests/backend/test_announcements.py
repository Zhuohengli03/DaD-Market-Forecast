"""Steam patch notes: keep patches only, store them locally, find the ones that name an item."""
from backend.services.announcements import AnnouncementStore, fetch_steam, is_patch, mentions, plain_text

H = {"X-Darker-Market": "1"}


def _post(gid, date, title, contents):
    return {"gid": gid, "date": date, "title": title, "url": f"https://steam/{gid}", "contents": contents,
            "feedname": "steam_community_announcements"}


def test_only_patches_count():
    assert is_patch("Hotfix #126") and is_patch("Early Access Season #11") and is_patch("Patch Notes 0.17")
    assert not is_patch("EAS10 Warlord Tournament – Final Results") and not is_patch("Tavern Table #5 is now LIVE!")
    assert not is_patch("Steam Autumn Sale | 30% Off Legendary Status, 50% Off EAS10 Seasonal Pack")
    assert not is_patch("Early Access Season #9 Twitch Drops Have Arrived!") and not is_patch("[Official Trailer] Dark and Darker: Season 5")
    assert is_patch("Currency Adjustment & Revaluation for EA Season #8") and is_patch("Semi-Wipe Announcement in Preparation for Early Access Season #6")


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
    s.merge([{"id": "3", "date": "2026-03-01T00:00:00Z", "title": "Steam Autumn Sale | 30% Off", "url": "u", "text": "t"}])
    assert [a["id"] for a in s.items()] == ["2", "1"]          # stored before the filter knew it: still left out
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


def test_item_mentions_carry_the_price_change_around_the_update(client):
    client.app.state.announcements.merge([{"id": "9", "date": "2026-01-24T00:00:00Z", "title": "Hotfix #9", "url": "u",
                                            "text": "Test Ore drops doubled."}])
    d = client.get("/api/announcements/item/Test Ore", params={"item": "Test Ore"}).json()["data"]
    c = d["mentions"][0]["change3d"]
    assert c is not None and -0.5 < c < 0.5                     # fixture prices drift slowly


def test_paging_stops_when_steam_keeps_returning_the_same_page():
    same = {"appnews": {"newsitems": [_post("5", 1791000000, "Hotfix #5", "x")]}}
    calls = []

    def get(url, params):
        calls.append(1)
        if len(calls) > 500:
            raise AssertionError("still paging")
        return same
    out = fetch_steam(get, since_ts=1)
    assert [a["id"] for a in out] == ["5"] and len(calls) <= 3


def test_item_matching_is_remembered_until_the_patches_change(tmp_path, monkeypatch):
    from backend.services import announcements as ann
    s = AnnouncementStore(tmp_path / "a.json")
    s.merge([{"id": "1", "date": "2026-01-01T00:00:00Z", "title": "Hotfix #1", "url": "u", "text": "Gold Ore up."}])
    calls = []
    real = ann.mentions
    monkeypatch.setattr(ann, "mentions", lambda name, items: (calls.append(name), real(name, items))[1])
    assert ann.cached_mentions("Gold Ore", s)[0]["id"] == "1"
    ann.cached_mentions("Gold Ore", s)
    assert calls == ["Gold Ore"]                       # second time from memory
    s.merge([{"id": "2", "date": "2026-02-01T00:00:00Z", "title": "Hotfix #2", "url": "u", "text": "Gold Ore down."}])
    assert [m["id"] for m in ann.cached_mentions("Gold Ore", s)] == ["2", "1"] and len(calls) == 2
