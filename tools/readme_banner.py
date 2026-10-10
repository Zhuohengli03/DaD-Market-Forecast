"""Draw the animated README banner: a price chart made of grid cells in Dark and Darker rarity colours,
eaten by a gold snake. Pure SVG + CSS animation (works inside a README <img>), light and dark versions.

    python tools/readme_banner.py   ->  docs/assets/market-snake.svg, docs/assets/market-snake-dark.svg
"""
from __future__ import annotations

import math
from pathlib import Path

COLS, ROWS = 44, 7
CELL, GAP, PAD = 11, 4, 6
STEP_S = 0.055                     # seconds per cell the snake moves
SNAKE = 7                          # body segments
PITCH = CELL + GAP

THEMES = {
    "market-snake.svg": {
        "empty": "#ebedf0",
        # poor, common, uncommon, rare, epic, legendary, unique (bottom row to top)
        "rarity": ["#8c959f", "#c4ccd4", "#2da44e", "#0969da", "#8250df", "#e36209", "#bf8700"],
        "snake": ("#b45309", "#f59e0b"),
    },
    "market-snake-dark.svg": {
        "empty": "#161b22",
        "rarity": ["#6e7681", "#c9d1d9", "#3fb950", "#58a6ff", "#bc8cff", "#f0883e", "#e3b341"],
        "snake": ("#fbbf24", "#d97706"),
    },
}


def price_heights() -> list[int]:
    """A made-up but believable hourly price: daily swing, noise and a slow climb, as column heights 1..7."""
    out = []
    for c in range(COLS):
        v = 3.4 + 1.9 * math.sin(c / 4.2) + 0.9 * math.sin(c / 1.6 + 1) + 0.045 * c
        out.append(max(1, min(ROWS, round(v))))
    return out


def route() -> list[tuple[int, int]]:
    """Lead-in from the left, then column by column up and down (boustrophedon), then out to the right."""
    path = [(-SNAKE - 1 + i, 0) for i in range(SNAKE + 1)]
    for c in range(COLS):
        rows = range(ROWS) if c % 2 == 0 else range(ROWS - 1, -1, -1)
        path += [(c, r) for r in rows]
    last = path[-1][1]
    path += [(COLS + i, last) for i in range(SNAKE + 2)]
    return path


def xy(col: int, row: int) -> tuple[float, float]:
    return PAD + col * PITCH, PAD + row * PITCH


def build(theme: dict) -> str:
    heights = price_heights()
    path = route()
    n = len(path)
    total = n * STEP_S
    pct = lambda i: 100 * i / (n - 1)
    w, h = PAD * 2 + COLS * PITCH - GAP, PAD * 2 + ROWS * PITCH - GAP
    css, cells = [], []
    order = {p: i for i, p in enumerate(path)}
    for c in range(COLS):
        for r in range(ROWS):
            x, y = xy(c, r)
            level = ROWS - 1 - r                      # 0 = bottom row
            food = level < heights[c]
            if not food:
                cells.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2.5" fill="{theme["empty"]}"/>')
                continue
            color = theme["rarity"][level]
            eaten = pct(order[(c, r)])
            name = f"c{c}_{r}"
            # the cell keeps its colour until the head arrives, stays eaten, and grows back as the loop restarts
            css.append(f"@keyframes {name}{{0%,{max(eaten - 0.01, 0):.3f}%{{fill:{color}}}{eaten:.3f}%,97%{{fill:{theme['empty']}}}100%{{fill:{color}}}}}")
            cells.append(f'<rect class="{name}" x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2.5" fill="{color}"/>')
            css.append(f".{name}{{animation:{name} {total:.2f}s linear infinite}}")
    frames = "".join(f"{pct(i):.3f}%{{transform:translate({xy(c, r)[0]}px,{xy(c, r)[1]}px)}}" for i, (c, r) in enumerate(path))
    css.append(f"@keyframes snake{{{frames}}}")
    head, tail = theme["snake"]
    body = []
    for k in range(SNAKE):
        t = k / (SNAKE - 1)
        size = CELL + 2 - 4 * t                     # head a little bigger, tail thinner
        off = (CELL - size) / 2
        color = mix(head, tail, t)
        body.append(f'<rect class="s{k}" x="{off:.2f}" y="{off:.2f}" width="{size:.2f}" height="{size:.2f}" rx="{size / 2.6:.2f}" fill="{color}"/>')
        # each segment follows the same path k steps behind the head (one full shorthand, so nothing resets the delay)
        css.append(f".s{k}{{transform:translate(-100px,0);animation:snake {total:.2f}s linear {k * STEP_S:.3f}s infinite both}}")
    title = "Darker Market: a price chart in Dark and Darker rarity colours, eaten by a snake"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">'
            f"<title>{title}</title><style>{''.join(css)}</style>{''.join(cells)}{''.join(reversed(body))}</svg>\n")


def mix(a: str, b: str, t: float) -> str:
    pa, pb = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (out / name).write_text(build(theme), encoding="utf-8")
        print(out / name, f"{(out / name).stat().st_size // 1024} KB")
