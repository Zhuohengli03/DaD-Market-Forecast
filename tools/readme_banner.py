"""Draw the animated README banner in a pixel-grid style: price history lights up column by column as if it
were being collected, then a "now" line appears and the forecast fades in with its uncertainty band.
Cells use Dark and Darker rarity colours. Pure SVG + CSS animation (works inside a README <img>).

    python tools/readme_banner.py   ->  docs/assets/market-banner.svg, docs/assets/market-banner-dark.svg
"""
from __future__ import annotations

import math
from pathlib import Path

COLS, ROWS = 44, 7
HISTORY = 32                       # columns of history; the rest is forecast
CELL, GAP, PAD = 11, 4, 6
PITCH = CELL + GAP
LOOP_S = 9.0                       # one full cycle
T_HISTORY = (0.04, 0.50)           # history fills in during this share of the loop
T_NOW = 0.53                       # the "now" divider appears
T_FORECAST = (0.56, 0.72)          # forecast fades in
T_FADE = (0.90, 0.98)              # everything fades before the loop restarts

THEMES = {
    "market-banner.svg": {
        "empty": "#ebedf0", "now": "#57606a",
        # poor, common, uncommon, rare, epic, legendary, unique (bottom row to top)
        "rarity": ["#8c959f", "#c4ccd4", "#2da44e", "#0969da", "#8250df", "#e36209", "#bf8700"],
    },
    "market-banner-dark.svg": {
        "empty": "#161b22", "now": "#8b949e",
        "rarity": ["#6e7681", "#c9d1d9", "#3fb950", "#58a6ff", "#bc8cff", "#f0883e", "#e3b341"],
    },
}


def series() -> list[float]:
    """A made-up but believable price: daily swing, noise and a slow climb, on a 1..7 scale."""
    return [max(1.0, min(ROWS, 3.3 + 1.8 * math.sin(c / 4.0) + 0.8 * math.sin(c / 1.5 + 1) + 0.04 * c)) for c in range(COLS)]


def forecast_band(c: int, mid: float) -> tuple[int, int, int]:
    """Median level and low/high levels for a forecast column; the band widens further ahead."""
    spread = 0.6 + 0.22 * (c - HISTORY)
    lo, hi = max(1, round(mid - spread)), min(ROWS, round(mid + spread))
    return lo, round(mid), hi


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def cell_anim(name: str, on: float, opacity: float) -> str:
    """Hidden, pop in at `on`, hold, fade out near the end of the loop."""
    return (f"@keyframes {name}{{0%,{pct(on)}{{opacity:0;transform:scale(.4)}}"
            f"{pct(on + 0.025)}{{opacity:{opacity};transform:scale(1)}}"
            f"{pct(T_FADE[0])}{{opacity:{opacity};transform:scale(1)}}{pct(T_FADE[1])},100%{{opacity:0;transform:scale(1)}}}}"
            f".{name}{{animation:{name} {LOOP_S}s ease-out infinite;transform-box:fill-box;transform-origin:center}}")


def build(theme: dict) -> str:
    prices = series()
    w, h = PAD * 2 + COLS * PITCH - GAP, PAD * 2 + ROWS * PITCH - GAP
    css, back, front = [], [], []
    for c in range(COLS):
        for r in range(ROWS):
            back.append(f'<rect x="{PAD + c * PITCH}" y="{PAD + r * PITCH}" width="{CELL}" height="{CELL}" rx="2.5" fill="{theme["empty"]}"/>')
    for c in range(COLS):
        x = PAD + c * PITCH
        if c < HISTORY:
            top = round(prices[c])
            for level in range(top):
                # collected column by column, each one stacking up from the bottom
                on = T_HISTORY[0] + (T_HISTORY[1] - T_HISTORY[0]) * (c + level / ROWS) / HISTORY
                name = f"h{c}_{level}"
                css.append(cell_anim(name, on, 1))
                y = PAD + (ROWS - 1 - level) * PITCH
                front.append(f'<rect class="{name}" x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2.5" fill="{theme["rarity"][level]}"/>')
        else:
            lo, mid, hi = forecast_band(c, prices[c])
            on = T_FORECAST[0] + (T_FORECAST[1] - T_FORECAST[0]) * (c - HISTORY) / (COLS - HISTORY)
            for level in range(lo - 1, hi):
                name = f"f{c}_{level}"
                median = level == mid - 1
                css.append(cell_anim(name, on, 1 if median else 0.35))
                y = PAD + (ROWS - 1 - level) * PITCH
                color = theme["rarity"][level]
                if median:   # the forecast itself: an outlined cell so it reads as "not yet real"
                    front.append(f'<rect class="{name}" x="{x + 1}" y="{y + 1}" width="{CELL - 2}" height="{CELL - 2}" rx="2" fill="none" stroke="{color}" stroke-width="2"/>')
                else:
                    front.append(f'<rect class="{name}" x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2.5" fill="{color}"/>')
    nx = PAD + HISTORY * PITCH - GAP / 2
    css.append(f"@keyframes now{{0%,{pct(T_NOW)}{{opacity:0}}{pct(T_NOW + 0.02)},{pct(T_FADE[0])}{{opacity:1}}{pct(T_FADE[1])},100%{{opacity:0}}}}"
               f".now{{animation:now {LOOP_S}s linear infinite}}")
    front.append(f'<line class="now" x1="{nx}" y1="{PAD - 3}" x2="{nx}" y2="{h - PAD + 3}" stroke="{theme["now"]}" stroke-width="1.5" stroke-dasharray="3 3"/>')
    title = "Darker Market: price history in Dark and Darker rarity colours fills in, then the forecast appears"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">'
            f"<title>{title}</title><style>{''.join(css)}</style>{''.join(back)}{''.join(front)}</svg>\n")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (out / name).write_text(build(theme), encoding="utf-8")
        print(out / name, f"{(out / name).stat().st_size // 1024} KB")
