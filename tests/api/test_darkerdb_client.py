import pytest

from src.api.darkerdb_client import DarkerDBClient, DarkerDBError, normalize_listing


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.responses.pop(0)


def _env(body, nxt=None, code=200):
    return {"code": code, "status": "OK", "body": body, "pagination": {"next": nxt, "limit": 50}}


def test_get_sends_key_header_and_unwraps_body():
    s = _Session([_Resp(200, _env({"status": "OK"}))])
    c = DarkerDBClient("k", session=s, min_interval=0)
    assert c.get("/health-check")["body"] == {"status": "OK"}
    url, params, headers = s.calls[0]
    assert url == "https://api.darkerdb.com/v2/health-check" and headers["X-API-Key"] == "k"


def test_error_envelope_raises_with_messages():
    s = _Session([_Resp(401, {"code": 401, "status": "Unauthorized",
                              "flash": {"errors": [{"message": "A valid DarkerDB API key is required", "code": "auth:unauthenticated"}]}})])
    c = DarkerDBClient("bad", session=s, min_interval=0)
    with pytest.raises(DarkerDBError) as e:
        c.get("/market")
    assert e.value.code == 401 and "valid DarkerDB API key" in str(e.value)


def test_429_honours_retry_after_then_succeeds(monkeypatch):
    sleeps = []
    s = _Session([_Resp(429, {"code": 429, "status": "Too Many Requests"}, {"Retry-After": "2"}),
                  _Resp(200, _env([]))])
    c = DarkerDBClient("k", session=s, min_interval=0, sleep=sleeps.append)
    assert c.get("/market")["body"] == []
    assert sleeps == [2.0] and len(s.calls) == 2


def test_iter_market_follows_cursor_until_null():
    s = _Session([_Resp(200, _env([{"id": "1"}, {"id": "2"}], nxt="c2")),
                  _Resp(200, _env([{"id": "3"}], nxt=None))])
    c = DarkerDBClient("k", session=s, min_interval=0)
    pages = list(c.iter_market("id.item.iron_ore_1001", since="2026-01-01", limit=50))
    assert [r["id"] for p in pages for r in p] == ["1", "2", "3"]
    assert s.calls[0][1] == {"item_id": "id.item.iron_ore_1001", "from": "2026-01-01", "limit": 50, "sort": "created_at:desc"}
    assert s.calls[1][1]["cursor"] == "c2"


def test_iter_market_respects_max_pages():
    s = _Session([_Resp(200, _env([{"id": "1"}], nxt="c2")), _Resp(200, _env([{"id": "2"}], nxt="c3"))])
    c = DarkerDBClient("k", session=s, min_interval=0)
    assert len(list(c.iter_market("x", max_pages=2))) == 2


def test_find_item_id_prefers_exact_name():
    s = _Session([_Resp(200, _env([{"id": "id.item.iron_ore_ingot_1001", "name": "Iron Ore Ingot"},
                                   {"id": "id.item.iron_ore_1001", "name": "Iron Ore"}]))])
    c = DarkerDBClient("k", session=s, min_interval=0)
    assert c.find_item_id("iron ore") == "id.item.iron_ore_1001"
    assert s.calls[0][1]["name"] == "iron ore"


def test_find_item_id_raises_when_absent():
    s = _Session([_Resp(200, _env([]))])
    with pytest.raises(DarkerDBError):
        DarkerDBClient("k", session=s, min_interval=0).find_item_id("Nope")


def test_normalize_listing_flat_and_nested():
    flat = {"id": 638093334476250124, "item": "Iron Ore", "quantity": 6, "price": 510, "price_per_unit": 85,
            "has_sold": True, "created_at": "2026-04-04T14:17:24Z"}
    n = normalize_listing(flat, "Iron Ore")
    assert n == {"id": "638093334476250124", "item": "Iron Ore", "quantity": 6, "price_per_unit": 85.0,
                 "price": 510.0, "has_sold": True, "created_at": "2026-04-04T14:17:24Z"}
    nested = {"id": "7", "item": {"id": "id.item.iron_ore_1001", "name": "Iron Ore"}, "quantity": 2,
              "price": 10, "listing_state": "sold", "created_at": "2026-04-04T14:17:24Z"}
    n2 = normalize_listing(nested, "fallback")
    assert n2["item"] == "Iron Ore" and n2["has_sold"] is True and n2["price_per_unit"] == 5.0


def test_normalize_listing_missing_fields_raise():
    with pytest.raises(KeyError):
        normalize_listing({"id": "1"}, "X")


def test_normalize_listing_canonicalizes_created_at():
    base = {"id": "1", "item": "Iron Ore", "quantity": 1, "price": 1, "price_per_unit": 1}
    for raw in ("2026-04-04T14:17:24Z", "2026-04-04T14:17:24+00:00", "2026-04-04T14:17:24.123Z", 1775312244):
        assert normalize_listing({**base, "created_at": raw}, "x")["created_at"] == "2026-04-04T14:17:24Z"


def test_iter_pages_generic_and_item_id_extraction():
    from src.api.darkerdb_client import listing_item_ref
    s = _Session([_Resp(200, _env([{"id": "a"}], nxt="n")), _Resp(200, _env([{"id": "b"}]))])
    c = DarkerDBClient("k", session=s, min_interval=0)
    assert [r["id"] for p in c.iter_pages("/items", {"limit": 200}) for r in p] == ["a", "b"]
    assert s.calls[1][1] == {"limit": 200, "cursor": "n"}
    assert listing_item_ref({"item_id": "id.item.x", "item": "X"}) == ("id.item.x", "X")
    assert listing_item_ref({"item": {"id": "id.item.y", "name": "Y"}}) == ("id.item.y", "Y")
    assert listing_item_ref({"item": "Z"}) == (None, "Z")


class _HResp(_Resp):
    def __init__(self, status, body, limit, remaining):
        super().__init__(status, body, {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(remaining)})


def test_adaptive_pacing_bursts_while_quota_remains_and_slows_near_the_limit():
    sleeps = []
    s = _Session([_HResp(200, _env([]), 60, 40), _HResp(200, _env([]), 60, 30), _HResp(200, _env([]), 60, 3), _HResp(200, _env([]), 60, 2)])
    c = DarkerDBClient("k", session=s, sleep=sleeps.append)
    for _ in range(4):
        c.get("/market")
    # no waiting while plenty of quota is left; once nearly exhausted, pace at 60s / limit
    assert sleeps and all(abs(x - 1.0) < 0.05 for x in sleeps) and len(sleeps) == 1


def test_pacing_follows_a_higher_plan_limit():
    sleeps = []
    s = _Session([_HResp(200, _env([]), 300, 2), _HResp(200, _env([]), 300, 1)])
    c = DarkerDBClient("k", session=s, sleep=sleeps.append)
    c.get("/market"); c.get("/market")
    assert sleeps and abs(sleeps[0] - 0.2) < 0.05          # 300 requests/min allowed -> 0.2s spacing


def test_collector_skips_the_lookup_when_the_item_id_is_known(tmp_path, monkeypatch):
    import src.api.collect as col
    calls = []

    class FakeClient:
        def __init__(self, key):
            pass

        def find_item_id(self, name):
            calls.append(name)
            return "id.item.x"

        def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None):
            yield []

    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", FakeClient)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    col.main(["--item", "Iron Ore", "--item-id", "id.item.iron_ores_2001", "--data-dir", str(tmp_path)])
    assert calls == []


def test_client_records_credits_left_for_the_app(tmp_path):
    import json
    f = tmp_path / "quota.json"
    s = _Session([_Resp(200, _env([]), {"X-Credits-Remaining": "7232", "X-Credits-Cost": "1", "X-Ratelimit-Limit": "60", "X-Ratelimit-Remaining": "59"})])
    DarkerDBClient("k", session=s, min_interval=0, quota_file=f).get("/patches")
    d = json.loads(f.read_text())
    assert d["credits_remaining"] == 7232 and d["rate_limit"] == 60 and d["updated_at"]


def test_quota_file_defaults_to_the_data_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DARKER_DATA_DIR", str(tmp_path))
    s = _Session([_Resp(200, _env([]), {"X-Credits-Remaining": "5"})])
    DarkerDBClient("k", session=s, min_interval=0).get("/patches")
    assert (tmp_path / "quota.json").exists()


def test_no_credit_header_leaves_the_file_alone(tmp_path):
    f = tmp_path / "quota.json"
    s = _Session([_Resp(200, _env([]), {})])
    DarkerDBClient("k", session=s, min_interval=0, quota_file=f).get("/patches")
    assert not f.exists()
