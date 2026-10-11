# 赛季日期与交易开放日期

DarkerDB `/v2/patches?kind=season` 当前只给出 `released_at` 日期，例如第 11 赛季为 `2026-10-09`。这不是精确的维护结束或交易开放时刻。导入时保留数据源提供的时间精度：日期按 UTC 零点近似；如果提供完整时间戳，则保留具体时刻。

赛季记录的 `start` / `end` 表示赛季区间，另有可选的 `market_open`（YYYY-MM-DD）表示交易行公告开放日期。行情、挂单列表、本赛季统计、采集和预测统一使用 `max(start, market_open 的 UTC 零点)` 作为下界，结束时间仍是该赛季的 `end`（不包含）。显式请求日期只能缩小选中赛季的范围；不选赛季的自定义查询仍能查看全部原始记录。

## 历史日历（核对日期：2026-10-10）

| 赛季 | 补丁发布日期 / 当前赛季下界 | 交易开放日 | 依据与精度 |
| --- | --- | --- | --- |
| 4 | 2024-09-05 | 未核实 | [补丁记录](https://darkerdb.com/database/patches/early-access-season-4)未提供延迟开放日期，保留原筛选边界 |
| 5 | 2024-12-18 | 未核实 | [补丁记录](https://darkerdb.com/database/patches/early-access-season-5)未提供延迟开放日期，保留原筛选边界 |
| 6 | 2025-04-10 | 2025-04-24（推算） | [赛季公告](https://darkanddarker.com/news/109)、[公告镜像](https://darkerdb.com/database/patches/early-access-season-6)：明确前两周关闭交易；由发布日期加 14 天得到，未核实精确开放时刻 |
| 7 | 2025-09-24 | 2025-10-07 | [赛季公告](https://darkanddarker.com/news/145)、[公告镜像](https://darkerdb.com/database/patches/early-access-season-7)：直接指定 10 月 7 日 |
| 8 | 2026-01-29 | 2026-02-20 | [赛季公告](https://www.darkanddarker.com/news/168)：直接指定 2 月 20 日 |
| 9 | 2026-04-28 | 2026-05-14 | [赛季公告](https://darkanddarker.com/news/185)：直接指定 5 月 14 日 |
| 10 | 2026-07-17 | 2026-07-30 | [Hotfix #121](https://darkanddarker.com/news/206)、[补丁日期及公告镜像](https://darkerdb.com/database/patches/hotfix-121)：该补丁宣布开放交易行，优先于开季预告里冲突的 7 月 30 / 31 日 |
| 11 | 2026-10-09 | 2026-10-23（计划） | IRONMACE **Early Access Season #11 / Game Updates** [Steam 公告](https://steamcommunity.com/app/2016590/allnews/?l=indonesian) |

以上没有确认精确开服时刻，因此 UTC 零点是日级筛选约定，不是已确认的开服时间。保留 DarkerDB 的赛季发布日期，避免将公告发布日、当地日期或交易开放日直接替换为清档时间。未列出的赛季不推算固定的 SSF 周数。

已导入的第 6–11 赛季通过 `src/season_calendar.py` 自动补充交易开放日期，仅匹配已知 ID 和发布日期。设置页的手工日期（包括明确清空）优先，重新导入补丁日期也会保留。第 6 赛季中途还有经济重置，本表遵循数据源的编号赛季，暂未将其拆成独立经济周期。

不修改原始挂单，也不把边界不明确的记录强行分配到上一赛季。历史预测文件不会重写，修正只影响后续预测。价格图坐标使用 UTC（Z），与设置页和筛选日期一致。

本次排查：金矿截图的 10 月 8 日 20:00 是 `2026-10-09T00:00:00Z` 在美东夏令时的显示。DarkerDB 最新金矿记录为 `2026-10-09T11:31:30Z`，接口 `pagination.freshness.status` 为 `stale`；这些记录不能仅凭补丁发布日期认定为第 11 赛季交易数据。
