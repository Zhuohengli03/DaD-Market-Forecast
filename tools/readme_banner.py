"""Draw the README animation: a price line draws itself with volume underneath, a "now" marker appears, then a
dashed forecast with its uncertainty band. Gold tones to match the cover. Pure SVG + CSS (works in a README <img>).

    python tools/readme_banner.py   ->  docs/assets/market-chart.svg, docs/assets/market-chart-dark.svg
"""
from __future__ import annotations

import math
from pathlib import Path

W, H = 760, 200
LEFT, RIGHT, TOP, BOTTOM = 16, 16, 18, 46       # chart area margins; volume bars live in the bottom margin
HISTORY, FUTURE = 60, 20                         # points of history and of forecast
LOOP = 9.0

THEMES = {
    "market-chart.svg": {"line": "#a16207", "area": "#ca8a04", "band": "#ca8a04", "grid": "#e5e7eb",
                         "vol": "#d6d3d1", "now": "#78716c"},
    "market-chart-dark.svg": {"line": "#e3b341", "area": "#e3b341", "band": "#e3b341", "grid": "#21262d",
                              "vol": "#30363d", "now": "#8b949e"},
}


def series() -> list[float]:
    """A believable hourly price: daily swing, noise and a slow climb (deterministic)."""
    return [100 + 9 * math.sin(i / 6.5) + 4 * math.sin(i / 2.3 + 1) + 2.5 * math.sin(i * 1.7) + 0.35 * i
            for i in range(HISTORY + FUTURE)]


def scale(values):
    lo, hi = min(values) - 6, max(values) + 8
    xs = lambda i: LEFT + (W - LEFT - RIGHT) * i / (HISTORY + FUTURE - 1)
    ys = lambda v: TOP + (H - TOP - BOTTOM) * (1 - (v - lo) / (hi - lo))
    return xs, ys


def length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build(t: dict) -> str:
    v = series()
    last = v[HISTORY - 1]
    edges = [last + 0.12 * k + 2.2 * math.sin(k / 3.5) + sgn * 0.75 * k for k in range(FUTURE + 1) for sgn in (1, -1)]
    xs, ys = scale(v[:HISTORY] + edges)
    hist = [(xs(i), ys(v[i])) for i in range(HISTORY)]
    # forecast: a smoothed continuation, its band widening further ahead
    last = v[HISTORY - 1]
    # forecasts are smooth: a gentle drift with a soft daily wave, and a band that widens further ahead
    mid = [last + 0.12 * k + 2.2 * math.sin(k / 3.5) for k in range(FUTURE + 1)]
    fc = [(xs(HISTORY - 1 + k), ys(m)) for k, m in enumerate(mid)]
    up = [(xs(HISTORY - 1 + k), ys(m + 0.75 * k)) for k, m in enumerate(mid)]
    dn = [(xs(HISTORY - 1 + k), ys(m - 0.75 * k)) for k, m in enumerate(mid)]
    poly = lambda pts: " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    base = H - BOTTOM
    hl, fl = length(hist), length(fc)
    nx = hist[-1][0]
    # timeline (share of the loop)
    draw_end, now_at, fc_start, fc_end, fade = 0.36, 0.40, 0.44, 0.62, 0.90
    css = [
        f"@keyframes draw{{0%{{stroke-dashoffset:{hl:.0f}}}{pct(draw_end)},100%{{stroke-dashoffset:0}}}}",
        f".hist{{stroke-dasharray:{hl:.0f};animation:draw {LOOP}s ease-in-out infinite,fade {LOOP}s linear infinite}}",
        f"@keyframes area{{0%,{pct(draw_end - 0.05)}{{opacity:0}}{pct(draw_end + 0.04)},{pct(fade)}{{opacity:.14}}100%{{opacity:0}}}}",
        f".area{{animation:area {LOOP}s linear infinite}}",
        f"@keyframes now{{0%,{pct(now_at)}{{opacity:0}}{pct(now_at + 0.03)},{pct(fade)}{{opacity:1}}100%{{opacity:0}}}}",
        f".now{{animation:now {LOOP}s linear infinite}}",
        f"@keyframes pulse{{0%,100%{{r:3.5}}50%{{r:6}}}}.dot{{animation:now {LOOP}s linear infinite,pulse 1.4s ease-in-out infinite}}",
        f"@keyframes fdraw{{0%,{pct(fc_start)}{{stroke-dashoffset:{fl:.0f}}}{pct(fc_end)},100%{{stroke-dashoffset:0}}}}",
        f".fc{{stroke-dasharray:{fl:.0f};animation:fdraw {LOOP}s ease-out infinite,fade {LOOP}s linear infinite}}",
        f"@keyframes band{{0%,{pct(fc_start + 0.04)}{{opacity:0}}{pct(fc_end)},{pct(fade)}{{opacity:.18}}100%{{opacity:0}}}}",
        f".band{{animation:band {LOOP}s linear infinite}}",
        f"@keyframes fade{{0%,{pct(fade)}{{opacity:1}}100%{{opacity:0}}}}",
    ]
    body = [f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>'
            for y in (TOP + (base - TOP) * k / 3 for k in range(4))]
    # volume: one bar per hour, rising in step with the line
    for i in range(HISTORY):
        hgt = 6 + 16 * (0.5 + 0.5 * math.sin(i * 1.3) * math.cos(i / 4.0)) ** 2
        on = draw_end * i / HISTORY
        name = f"v{i}"
        css.append(f"@keyframes {name}{{0%,{pct(on)}{{transform:scaleY(0)}}{pct(on + 0.03)},{pct(fade)}{{transform:scaleY(1)}}100%{{transform:scaleY(0)}}}}"
                   f".{name}{{animation:{name} {LOOP}s ease-out infinite;transform-box:fill-box;transform-origin:bottom}}")
        body.append(f'<rect class="{name}" x="{xs(i) - 3:.1f}" y="{H - 6 - hgt:.1f}" width="6" height="{hgt:.1f}" rx="1.5" fill="{t["vol"]}"/>')
    body += [
        f'<polygon class="area" points="{poly(hist)} {hist[-1][0]:.1f},{base} {hist[0][0]:.1f},{base}" fill="{t["area"]}" opacity="0"/>',
        f'<polyline class="hist" points="{poly(hist)}" fill="none" stroke="{t["line"]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>',
        f'<polygon class="band" points="{poly(up)} {poly(list(reversed(dn)))}" fill="{t["band"]}" opacity="0"/>',
        # the dashed forecast is revealed by a solid line drawing itself inside a mask (dashes and drawing both need dasharray)
        f'<mask id="reveal" maskUnits="userSpaceOnUse"><polyline class="fc" points="{poly(fc)}" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/></mask>',
        f'<polyline points="{poly(fc)}" fill="none" stroke="{t["line"]}" stroke-width="2.5" stroke-dasharray="6 5" stroke-linecap="round" mask="url(#reveal)"/>',
        f'<line class="now" x1="{nx:.1f}" x2="{nx:.1f}" y1="{TOP - 6}" y2="{base}" stroke="{t["now"]}" stroke-width="1.2" stroke-dasharray="3 4" opacity="0"/>',
        f'<circle class="dot" cx="{nx:.1f}" cy="{hist[-1][1]:.1f}" r="3.5" fill="{t["line"]}" opacity="0"/>',
    ]
    title = "Darker Market: the price line draws itself, then the forecast and its range appear"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{title}">'
            f"<title>{title}</title><style>{''.join(css)}</style>{''.join(body)}</svg>\n")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (out / name).write_text(build(theme), encoding="utf-8")
        print(out / name, f"{(out / name).stat().st_size // 1024} KB")
