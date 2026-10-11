"""Weekly market reports. The program computes every number from local data; the language model only writes
them up in Chinese. Every number in the reply is checked against the facts and anything else is flagged."""
from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.season_calendar import market_start

from src.forecast.aggregate import hourly_series
from ..data.catalog import variant_name_zh
from .envfile import atomic_write

MOVERS = 3
EXPECTED_REPLY_TOKENS = 1500
CHARS_PER_TOKEN = 1.5            # rough for Chinese text and JSON
COMMON_NUMBERS = {24, 48, 72, 168, 100}   # hours/percent words a writer may use without them being "data"

SYSTEM_PROMPT = """你是《暗黑地牢》（Dark and Darker）交易行的行情分析师，为付费订阅者写每周行情报告。
规则：
1. 只能使用用户提供的事实数据里的数字，不得编造、推算或补充任何数字；可以四舍五入，但不要换算出新的数。
2. 预测部分：如果某个道具的预测没有跑赢基准（beats_baseline 为 false），必须明确写“价格接近随机波动，没有可靠的方向判断”，不要硬给方向。
3. 这是游戏内交易行，不是金融投资；不要承诺收益，结尾加一句简短的风险提示。
4. 用简体中文，语气专业但易懂，道具用中文名（name_zh 为空时用 name）。
5. 严格按下面的格式输出，只用这两个二级标题：
## 公开摘要
（3~5 句话，概括本周整体行情和涨跌最明显的道具，所有人可见，用来吸引订阅）
## 订阅者正文
（分小节：本周涨跌榜、重点道具分析、预测结论、操作建议与风险提示；可以用列表和小标题）"""


def _num(v, digits=2):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits)


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t if t.tzinfo else t.tz_localize("UTC")


def _window_stats(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    hourly = hourly_series(df)
    prices = hourly["median_price"].dropna()
    if prices.empty:
        return None
    mean = float(prices.mean())
    return {"median": _num(prices.median()), "first": _num(prices.iloc[0]), "last": _num(prices.iloc[-1]),
            "low": _num(prices.min()), "high": _num(prices.max()),
            "volatility_pct": _num(float(prices.std() / mean * 100) if len(prices) > 1 and mean else 0.0, 1),
            "hours": int(len(prices)), "listings": int(len(df)),
            "sold": int(pd.to_numeric(df["has_sold"], errors="coerce").fillna(0).astype(bool).sum()) if "has_sold" in df else 0}


def _forecast(st, name: str, season_id: str) -> dict | None:
    """The newest forecast for this item in this season, reduced to its conclusion."""
    runs = [r for r in st.runs.list() if r.get("item") == name and r.get("season_id") == season_id]
    if not runs:
        return None
    run = st.runs.get(runs[0]["run_id"])
    tail = [p["median_price"] for p in run.get("series_tail") or [] if p.get("median_price") is not None]
    end = (run.get("forecast") or [None])[-1]
    if not end or not tail:
        return None
    return {"horizon_hours": run["request"]["horizon_hours"], "method": run["selected"]["name"],
            "beats_baseline": bool(run.get("beats_baseline")), "skill_pct": _num((run.get("skill_vs_naive") or 0) * 100, 1),
            "from_price": _num(tail[-1]), "predicted": _num(end["pred"]), "low": _num(end["lower"]), "high": _num(end["upper"]),
            "change_pct": _num((end["pred"] / tail[-1] - 1) * 100, 1) if tail[-1] else None, "made_at": run.get("created_at")}


def build_facts(st, names: list[str], days: int = 7, season_id: str | None = None) -> dict:
    """Everything the report may say, computed here. The period ends at the newest data in the season."""
    from .collection import pick_season
    from .overview import _trading_windows
    if season_id:
        season = pick_season(st, season_id)
        windows = [(season, _ts(market_start(season)), _ts(season["end"]) if season.get("end") else None)]
    else:
        # the newest season with trading data: before a new season's market opens, the last one that traded
        windows = _trading_windows(st)
    if not windows:
        raise ValueError("还没有赛季信息，请先在设置里导入赛季 / no seasons yet")
    for season, s_start, s_end in windows:      # from the market opening: solo weeks before it are no market
        frames = {}
        for name in names:
            try:
                df = st.repo.fetch_all(name, s_start, s_end)
            except KeyError:
                continue
            if not df.empty:
                frames[name] = df.assign(created_at=pd.to_datetime(df["created_at"], utc=True))
        if frames:
            break
    newest = max((df["created_at"].max() for df in frames.values()), default=None)
    end = newest if newest is not None else (s_end or pd.Timestamp.now(tz="UTC"))
    start = max(end - pd.Timedelta(days=days), s_start)
    prev_start = max(start - (end - start), s_start)
    items, skipped = [], [n for n in names if n not in frames]
    for name, df in frames.items():
        now = _window_stats(df[(df["created_at"] >= start) & (df["created_at"] <= end)])
        if now is None:
            skipped.append(name)
            continue
        prev = _window_stats(df[(df["created_at"] >= prev_start) & (df["created_at"] < start)]) if prev_start < start else None
        entry = st.items.get(name)
        found = st.catalog.variant(entry["item_id"]) if entry.get("item_id") else None
        change = _num((now["median"] / prev["median"] - 1) * 100, 1) if prev and prev["median"] else None
        items.append({"name": name, "name_zh": variant_name_zh(*found) if found else None, **now,
                      "prev_median": prev["median"] if prev else None, "change_pct": change,
                      "forecast": _forecast(st, name, season["id"])})
    ranked = [i for i in items if i["change_pct"] is not None]
    up = sorted((i for i in ranked if i["change_pct"] > 0), key=lambda i: -i["change_pct"])[:MOVERS]
    down = sorted((i for i in ranked if i["change_pct"] < 0), key=lambda i: i["change_pct"])[:MOVERS]
    label = lambda i: {"name": i["name_zh"] or i["name"], "change_pct": i["change_pct"]}
    return {"season": {"id": season["id"], "name": season["name"]},
            "period": {"start": start.isoformat(), "end": end.isoformat(), "days": days,
                       "compared_with": {"start": prev_start.isoformat(), "end": start.isoformat()}},
            "items": items, "skipped": skipped, "movers": {"up": [label(i) for i in up], "down": [label(i) for i in down]}}


def messages(facts: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "以下是本期的事实数据（JSON），请据此写报告：\n" + json.dumps(facts, ensure_ascii=False, indent=1)}]


def estimate_tokens(facts: dict) -> int:
    return math.ceil(sum(len(m["content"]) for m in messages(facts)) / CHARS_PER_TOKEN)


def parse_sections(text: str) -> dict:
    """Split the reply at the two required headings; anything unexpected becomes the body."""
    m = re.search(r"^#+\s*公开摘要\s*$(.*?)^#+\s*订阅者正文\s*$(.*)", text, re.S | re.M)
    if not m:
        return {"public": "", "body": text.strip()}
    return {"public": m.group(1).strip(), "body": m.group(2).strip()}


def _fact_numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)) and not (isinstance(obj, float) and math.isnan(obj)):
        out += [float(obj), abs(float(obj))]
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _fact_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _fact_numbers(v)
    return out


_DATES = re.compile(r"\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}日?|\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")


def unmatched_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in the text that are not (a rounding of) a number in the facts; dates and small counts are skipped."""
    known = _fact_numbers(facts) + [float(n) for n in COMMON_NUMBERS]
    bad = []
    for m in _NUMBER.finditer(_DATES.sub(" ", text.replace(",", ""))):
        raw, pct = m.group(1), m.group(2)
        v = float(raw)
        if not pct and v <= 10 and "." not in raw:
            continue                                   # "前 3 名", "7 天"
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(v - k) <= tol for k in known):
            bad.append(raw + pct)
    return bad


def cost(prompt_tokens: int, completion_tokens: int, price_in, price_out) -> float | None:
    """Yuan, from prices per million tokens; None when the prices are not set."""
    if price_in is None or price_out is None:
        return None
    return round((prompt_tokens * float(price_in) + completion_tokens * float(price_out)) / 1e6, 6)


class ReportStore:
    def __init__(self, directory: Path):
        self._dir = Path(directory)

    def _path(self, rid: str) -> Path:
        if not re.fullmatch(r"[0-9A-Za-z-]+", rid):
            raise KeyError(rid)
        return self._dir / f"{rid}.json"

    def save(self, report: dict) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        rid = report.get("id") or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        report = {**report, "id": rid}
        atomic_write(self._path(rid), json.dumps(report, ensure_ascii=False, indent=1))
        return rid

    def get(self, rid: str) -> dict:
        p = self._path(rid)
        if not p.exists():
            raise KeyError(rid)
        return json.loads(p.read_text(encoding="utf-8"))

    def delete(self, rid: str) -> None:
        p = self._path(rid)
        if not p.exists():
            raise KeyError(rid)
        p.unlink()

    def list(self) -> list[dict]:
        out = []
        for p in self._dir.glob("*.json") if self._dir.exists() else []:
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({k: r.get(k) for k in ("id", "created_at", "period", "season", "model", "usage", "cost")} | {"items": len(r.get("facts", {}).get("items", [])), "unmatched": len(r.get("unmatched") or [])})
        return sorted(out, key=lambda r: r["created_at"] or "", reverse=True)
