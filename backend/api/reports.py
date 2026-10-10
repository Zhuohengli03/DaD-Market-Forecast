"""AI market reports through Alibaba Cloud Bailian, configured in Settings."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..schemas import ok
from ..services import report as rep
from ..services.envfile import upsert_env
from ..services.llm import DEFAULT_MODEL, BailianClient, LLMError, base_url
from .setup import key_hint

router = APIRouter(tags=["reports"])
MAX_ITEMS = 60


def _cfg(st) -> dict:
    p = st.app_settings.get()
    return {"api_key": st.settings.dashscope_api_key, "model": p["llm_model"] or DEFAULT_MODEL, "region": p["llm_region"],
            "workspace": p["llm_workspace"], "price_in": p["llm_price_in"], "price_out": p["llm_price_out"]}


def _client(st):
    cfg = _cfg(st)
    if not cfg["api_key"]:
        raise HTTPException(409, "还没有填写阿里云百炼 API key，请先到「设置 → AI 分析」填写 / set your Bailian API key first")
    factory = getattr(st, "llm_factory", None)
    return factory(cfg) if factory else BailianClient(cfg["api_key"], cfg["model"], cfg["region"], cfg["workspace"])


def _public(st) -> dict:
    cfg = _cfg(st)
    return {"configured": bool(cfg["api_key"]), "key_hint": key_hint(cfg["api_key"]), "model": cfg["model"], "region": cfg["region"],
            "workspace": cfg["workspace"], "price_in": cfg["price_in"], "price_out": cfg["price_out"],
            "base_url": base_url(cfg["region"], cfg["workspace"])}


@router.get("/setup/llm")
def get_llm(request: Request):
    return ok(_public(request.app.state))


class LLMIn(BaseModel):
    api_key: str | None = None            # empty: keep the saved key
    model: str = Field(DEFAULT_MODEL, min_length=1, max_length=80, pattern=r"^[\w.\-]+$")
    region: Literal["cn", "intl", "us"] = "cn"
    workspace: str = Field("", max_length=80, pattern=r"^[\w-]*$")
    price_in: float | None = Field(None, ge=0, le=10000)
    price_out: float | None = Field(None, ge=0, le=10000)


@router.put("/setup/llm")
def put_llm(body: LLMIn, request: Request):
    st = request.app.state
    key = (body.api_key or "").strip()
    if key:
        upsert_env(st.settings.env_path, "DASHSCOPE_API_KEY", key)
        st.settings.dashscope_api_key = key
    st.app_settings.update(llm_model=body.model, llm_region=body.region, llm_workspace=body.workspace,
                           llm_price_in=body.price_in, llm_price_out=body.price_out)
    return ok(_public(st))


@router.post("/setup/llm/test")
def test_llm(request: Request):
    """A tiny request to check key, region and model (a few tokens)."""
    st = request.app.state
    try:
        out = _client(st).chat([{"role": "user", "content": "只回复：OK"}])
    except LLMError as exc:
        raise HTTPException(502, str(exc))
    return ok({"reply": out["text"][:40], "tokens": out["prompt_tokens"] + out["completion_tokens"]})


class ReportIn(BaseModel):
    names: list[str] | None = Field(None, max_length=MAX_ITEMS * 10)   # default: items ticked for scheduled collection
    days: int = Field(7, ge=1, le=60)
    season_id: str | None = None


def _names(st, body: ReportIn) -> tuple[list[str], str]:
    """Explicit names; else the items ticked for scheduled collection; else (none ticked) every item with data."""
    if body.names:
        return list(dict.fromkeys(body.names)), "chosen"
    stats = {i.name: i.rows for i in st.item_stats.get()}
    with_data = [it for it in st.items.list() if stats.get(it["name"])]
    ticked = [it["name"] for it in with_data if it.get("scheduled") is not False]
    return (ticked, "ticked") if ticked else ([it["name"] for it in with_data], "all")


def _facts(st, body: ReportIn) -> dict:
    names, scope = _names(st, body)
    if not names:
        raise HTTPException(422, "没有可以写进报告的道具（需要已追踪、有数据的道具）/ no items with data")
    try:
        facts = rep.build_facts(st, names, body.days, body.season_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc).strip("'"))
    # keep the prompt (and its price) bounded: the most-listed items first
    facts["items"] = sorted(facts["items"], key=lambda i: -i["listings"])[:MAX_ITEMS]
    if not facts["items"]:
        raise HTTPException(422, "所选时间里这些道具都没有挂单 / no listings in that period")
    facts["scope"] = scope
    return facts


@router.post("/reports/estimate")
def estimate(body: ReportIn, request: Request):
    st = request.app.state
    facts = _facts(st, body)
    cfg = _cfg(st)
    pt = rep.estimate_tokens(facts)
    return ok({"items": len(facts["items"]), "skipped": len(facts["skipped"]), "period": facts["period"], "scope": facts["scope"],
               "prompt_tokens": pt, "completion_tokens": rep.EXPECTED_REPLY_TOKENS,
               "cost": rep.cost(pt, rep.EXPECTED_REPLY_TOKENS, cfg["price_in"], cfg["price_out"])})


@router.post("/reports", status_code=202)
def create(body: ReportIn, request: Request):
    st = request.app.state
    client = _client(st)
    facts = _facts(st, body)
    if st.jobs.find_active("report"):
        raise HTTPException(409, "已有报告正在生成 / a report is already being written")
    cfg = _cfg(st)
    job = st.jobs.create("report", "生成行情报告 / write market report")

    def run(j):
        st.jobs.progress(j, "writing", None, None, items=len(facts["items"]))
        try:
            out = client.chat(rep.messages(facts))
        except LLMError as exc:
            raise RuntimeError(str(exc))
        parts = rep.parse_sections(out["text"])
        usage = {"prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"]}
        from datetime import datetime, timezone
        rid = st.reports.save({"created_at": datetime.now(timezone.utc).isoformat(), "model": cfg["model"],
                               "period": facts["period"], "season": facts["season"], "facts": facts, **parts,
                               "unmatched": rep.unmatched_numbers(parts["public"] + "\n" + parts["body"], facts),
                               "usage": usage, "cost": rep.cost(usage["prompt_tokens"], usage["completion_tokens"], cfg["price_in"], cfg["price_out"])})
        st.jobs.progress(j, "done", None, None, report_id=rid)
        return rid
    st.jobs.run_in_thread(job, run)
    return ok({"job_id": job.id})


@router.get("/reports")
def list_reports(request: Request):
    return ok(request.app.state.reports.list())


def _get(st, rid: str) -> dict:
    try:
        return st.reports.get(rid)
    except KeyError:
        raise HTTPException(404, "没有这份报告 / report not found")


@router.get("/reports/{rid}")
def get_report(rid: str, request: Request):
    return ok(_get(request.app.state, rid))


class EditIn(BaseModel):
    public: str = Field(max_length=20000)
    body: str = Field(max_length=100000)


@router.put("/reports/{rid}")
def edit_report(rid: str, body: EditIn, request: Request):
    st = request.app.state
    r = _get(st, rid)
    r.update(public=body.public, body=body.body, unmatched=rep.unmatched_numbers(body.public + "\n" + body.body, r["facts"]), edited=True)
    st.reports.save(r)
    return ok(r)


@router.delete("/reports/{rid}")
def delete_report(rid: str, request: Request):
    st = request.app.state
    _get(st, rid)
    st.reports.delete(rid)
    return ok({"deleted": rid})
