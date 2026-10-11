"""Market reports: the program computes every number, the language model only writes them up (via Alibaba
Cloud Bailian), and every number in the text is checked against the facts."""
import json
import time

import pandas as pd
import pytest

from backend.services.llm import BailianClient, LLMError, base_url
from backend.services.report import build_facts, parse_sections, unmatched_numbers

H = {"X-Darker-Market": "1"}


def _wait(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ---- facts --------------------------------------------------------------------------------------------
def test_facts_cover_the_last_week_of_the_season_with_the_week_before(client):
    st = client.app.state
    f = build_facts(st, ["Test Ore"], days=7)
    assert f["period"]["days"] == 7 and f["season"]["name"] == "Season 2"
    end = pd.Timestamp(f["period"]["end"])
    newest = pd.Timestamp(next(i for i in st.item_stats.get() if i.name == "Test Ore").last_ts)
    assert abs((end - newest).total_seconds()) < 3600           # ends at the newest data, not "now"
    it = f["items"][0]
    assert it["name"] == "Test Ore" and it["listings"] > 0 and it["median"] > 0
    assert it["prev_median"] is not None and it["change_pct"] == pytest.approx(it["median"] / it["prev_median"] * 100 - 100, abs=0.11)
    assert it["low"] <= it["median"] <= it["high"]
    assert f["movers"]["up"] or f["movers"]["down"]


def test_facts_skip_items_without_data_in_the_period(client):
    st = client.app.state
    st.items.track("Iron Ore", "id.item.iron_ores_2001", "misc")
    f = build_facts(st, ["Test Ore", "Iron Ore"], days=7)
    assert [i["name"] for i in f["items"]] == ["Test Ore"] and f["skipped"] == ["Iron Ore"]


# ---- checking the model's numbers -------------------------------------------------------------------------
FACTS = {"items": [{"name_zh": "金矿", "median": 216.67, "change_pct": 12.3, "listings": 292, "low": 104.17, "high": 235.67}],
         "period": {"start": "2026-10-02T00:00:00+00:00", "end": "2026-10-09T00:00:00+00:00", "days": 7}}


def test_numbers_in_the_text_must_come_from_the_facts():
    ok = "金矿本周中位价 216.67，比上周涨 12.3%，共 292 条挂单，区间 104.17 到 235.67（2026-10-09）。前 3 名。"
    assert unmatched_numbers(ok, FACTS) == []
    bad = "金矿本周中位价 230.5，涨了 15%。"
    assert unmatched_numbers(bad, FACTS) == ["230.5", "15%"]
    assert unmatched_numbers("约 217，涨 12%", FACTS) == []      # rounding of real numbers is fine


def test_reply_splits_into_public_summary_and_subscriber_body():
    text = "## 公开摘要\n本周金矿上涨。\n\n## 订阅者正文\n### 重点道具\n金矿……"
    assert parse_sections(text) == {"public": "本周金矿上涨。", "body": "### 重点道具\n金矿……"}
    assert parse_sections("没有标题的回复") == {"public": "", "body": "没有标题的回复"}


# ---- Bailian client --------------------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        return self.resp


def test_base_url_per_region():
    assert base_url("cn", "") == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert base_url("cn", "ws123") == "https://ws123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert base_url("intl", "") == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert base_url("us", "") == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"


def test_client_sends_chat_and_reads_usage():
    s = _Session(_Resp(200, {"choices": [{"message": {"content": "## 公开摘要\n好"}}], "usage": {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200}}))
    out = BailianClient("sk-x", "qwen-plus", region="cn", session=s).chat([{"role": "user", "content": "hi"}])
    url, headers, body = s.calls[0]
    assert url.endswith("/compatible-mode/v1/chat/completions") and headers["Authorization"] == "Bearer sk-x"
    assert body["model"] == "qwen-plus" and body["messages"][0]["content"] == "hi"
    assert out == {"text": "## 公开摘要\n好", "prompt_tokens": 900, "completion_tokens": 300}


def test_client_explains_a_rejected_key():
    s = _Session(_Resp(401, {"error": {"message": "Incorrect API key"}}))
    with pytest.raises(LLMError) as e:
        BailianClient("bad", "qwen-plus", session=s).chat([{"role": "user", "content": "hi"}])
    assert "API key" in str(e.value) and "地域" in str(e.value)


# ---- settings and report endpoints -----------------------------------------------------------------------
def test_llm_settings_in_the_app(client, settings):
    d = client.get("/api/setup/llm").json()["data"]
    assert d["configured"] is False and d["model"] == "qwen-plus" and d["region"] == "cn"
    r = client.put("/api/setup/llm", json={"api_key": "sk-abcdef1234", "model": "qwen-max", "region": "intl", "workspace": "",
                                           "price_in": 2.4, "price_out": 9.6}, headers=H)
    d = r.json()["data"]
    assert r.status_code == 200 and d["configured"] and d["key_hint"] == "…1234" and d["model"] == "qwen-max"
    assert "sk-abcdef1234" not in json.dumps(d)                    # the key never comes back
    assert "DASHSCOPE_API_KEY=sk-abcdef1234" in settings.env_path.read_text()
    # saving without a key keeps the stored one
    client.put("/api/setup/llm", json={"model": "qwen-plus", "region": "cn", "workspace": ""}, headers=H)
    assert client.get("/api/setup/llm").json()["data"]["configured"] is True


class _FakeLLM:
    def __init__(self, text):
        self.text, self.messages = text, None

    def chat(self, messages):
        self.messages = messages
        return {"text": self.text, "prompt_tokens": 1000, "completion_tokens": 400}


def test_generate_a_report_end_to_end(client):
    st = client.app.state
    client.put("/api/setup/llm", json={"api_key": "sk-test9999", "model": "qwen-plus", "region": "cn", "workspace": "",
                                       "price_in": 0.8, "price_out": 2.0}, headers=H)
    est = client.post("/api/reports/estimate", json={"names": ["Test Ore"], "days": 7}, headers=H).json()["data"]
    assert est["items"] == 1 and est["prompt_tokens"] > 0 and est["cost"] is not None
    fake = _FakeLLM("## 公开摘要\nTest Ore 本周中位价 999.99。\n\n## 订阅者正文\n分析……")
    st.llm_factory = lambda cfg: fake
    r = client.post("/api/reports", json={"names": ["Test Ore"], "days": 7}, headers=H)
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded"
    rid = job["result_ref"]
    rep = client.get(f"/api/reports/{rid}").json()["data"]
    assert rep["public"].startswith("Test Ore 本周") and rep["body"] == "分析……"
    assert rep["unmatched"] == ["999.99"]                           # flagged: not in the facts
    assert rep["usage"] == {"prompt_tokens": 1000, "completion_tokens": 400} and rep["cost"] == pytest.approx(0.0016)
    assert "只能使用" in fake.messages[0]["content"] and '"Test Ore"' in fake.messages[1]["content"]
    assert client.get("/api/reports").json()["data"][0]["id"] == rid
    client.put(f"/api/reports/{rid}", json={"public": "改过的摘要", "body": "改过的正文"}, headers=H)
    assert client.get(f"/api/reports/{rid}").json()["data"]["public"] == "改过的摘要"
    assert client.delete(f"/api/reports/{rid}", headers=H).status_code == 200
    assert client.get(f"/api/reports/{rid}").status_code == 404


def test_report_needs_a_key(client):
    r = client.post("/api/reports", json={"names": ["Test Ore"], "days": 7}, headers=H)
    assert r.status_code == 409 and "百炼" in r.json()["error"]


def test_report_scope_falls_back_to_all_items_with_data_when_none_are_ticked(client):
    client.put("/api/items/scheduled", json={"names": ["Test Ore"], "scheduled": False}, headers=H)
    est = client.post("/api/reports/estimate", json={"days": 7}, headers=H).json()["data"]
    assert est["items"] == 1 and est["scope"] == "all"
    client.put("/api/items/scheduled", json={"names": ["Test Ore"], "scheduled": True}, headers=H)
    assert client.post("/api/reports/estimate", json={"days": 7}, headers=H).json()["data"]["scope"] == "ticked"


def test_report_skips_records_from_before_the_market_opened(client):
    from backend.services.report import build_facts
    st = client.app.state
    s = st.seasons.get("s2")
    client.put("/api/seasons/s2", json={**s, "market_open": "2026-01-25"}, headers={"X-Darker-Market": "1"})
    f = build_facts(st, ["Test Ore"], days=60)
    assert f["period"]["start"] >= "2026-01-25T00:00:00+00:00" and f["season"]["id"] == "s2"


def test_before_this_season_trades_the_report_covers_the_last_season_that_did(client):
    from backend.schemas import SeasonIn
    from backend.services.report import build_facts
    st = client.app.state
    s2 = st.seasons.get("s2")
    st.seasons.update("s2", SeasonIn(name=s2["name"], start=s2["start"], end="2026-02-01T00:00:00+00:00", note=""))
    st.seasons.add(SeasonIn(name="Season 3", start="2026-03-01T00:00:00+00:00", end=None, note=""))
    f = build_facts(st, ["Test Ore"], days=7)
    assert f["season"]["id"] == "s2" and f["items"] and f["period"]["end"] < "2026-02-01"
