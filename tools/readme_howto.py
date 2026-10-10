"""Three small pixel-grid animations that explain the app in the README, in light and dark versions:

  how-collect   listings stream from DarkerDB into the local database, which fills up
  how-filter    equipment listings with random stats; a stat range slides and non-matching listings dim
  how-forecast  backtest race: the baseline's error bar against a model's; the model is used only if clearly shorter

    python tools/readme_howto.py  ->  docs/assets/how-*.svg
"""
from __future__ import annotations

import random
from pathlib import Path

CELL, GAP = 11, 4
P = CELL + GAP
LOOP = 6.0

THEMES = {
    "": {"empty": "#ebedf0", "ink": "#57606a", "muted": "#d0d7de", "good": "#2da44e", "base": "#8c959f",
         "rarity": ["#8c959f", "#c4ccd4", "#2da44e", "#0969da", "#8250df", "#e36209", "#bf8700"]},
    "-dark": {"empty": "#161b22", "ink": "#8b949e", "muted": "#30363d", "good": "#3fb950", "base": "#6e7681",
              "rarity": ["#6e7681", "#c9d1d9", "#3fb950", "#58a6ff", "#bc8cff", "#f0883e", "#e3b341"]},
}


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def rect(x, y, fill, cls="", extra=""):
    c = f' class="{cls}"' if cls else ""
    return f'<rect{c} x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2.5" fill="{fill}"{extra}/>'


def svg(w, h, title, css, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">'
            f"<title>{title}</title><style>{''.join(css)}</style>{''.join(body)}</svg>\n")


def collect(t) -> str:
    """Source block on the left, cells travel along a track, the database on the right fills bottom-up."""
    rnd = random.Random(7)
    src_cols, src_rows, track, db_cols, db_rows = 3, 5, 6, 5, 5
    w = (src_cols + track + db_cols) * P + 6
    h = db_rows * P + 6
    css, body = [], []
    for c in range(src_cols):                 # DarkerDB: a shimmering block of listings
        for r in range(src_rows):
            name = f"s{c}_{r}"
            d = rnd.uniform(0, LOOP)
            css.append(f".{name}{{animation:shim 1.6s ease-in-out {-d:.2f}s infinite}}")
            body.append(rect(3 + c * P, 3 + r * P, t["rarity"][rnd.randrange(2, 7)], name))
    css.append("@keyframes shim{0%,100%{opacity:1}50%{opacity:.45}}")
    for c in range(track):                    # dotted track
        body.append(f'<rect x="{3 + (src_cols + c) * P + 4}" y="{3 + 2 * P + 4}" width="3" height="3" rx="1.5" fill="{t["muted"]}"/>')
    n = db_cols * db_rows
    for i in range(n):                        # database slots, filled one by one (bottom row first)
        r, c = db_rows - 1 - i // db_cols, i % db_cols
        x, y = 3 + (src_cols + track + c) * P, 3 + r * P
        body.append(rect(x, y, t["empty"]))
        on = 0.05 + 0.8 * i / n
        name = f"d{i}"
        css.append(f"@keyframes {name}{{0%,{pct(on)}{{opacity:0}}{pct(on + 0.02)},92%{{opacity:1}}100%{{opacity:0}}}}"
                   f".{name}{{animation:{name} {LOOP}s linear infinite}}")
        body.append(rect(x, y, t["rarity"][2 + (i * 3) % 5], name))
    # one travelling cell per arrival, sliding along the track
    x0, x1 = 3 + src_cols * P, 3 + (src_cols + track - 1) * P
    for k in range(5):
        name = f"m{k}"
        css.append(f".{name}{{animation:move {LOOP / 5:.2f}s linear {-k * LOOP / 25:.2f}s infinite}}")
        body.append(rect(0, 3 + 2 * P, t["rarity"][3 + k % 4], name))
    css.append(f"@keyframes move{{0%{{transform:translateX({x0}px);opacity:0}}15%{{opacity:1}}85%{{opacity:1}}100%{{transform:translateX({x1}px);opacity:0}}}}")
    return svg(w, h, "Listings flow from DarkerDB into your local database", css, body)


def filt(t) -> str:
    """A grid of equipment listings with one stat value each; a highlighted range slides along a scale and
    listings outside it dim."""
    rnd = random.Random(3)
    cols, rows = 14, 4
    w, h = cols * P + 6, (rows + 2) * P + 6
    css, body = [], []
    values = [[rnd.random() for _ in range(cols)] for _ in range(rows)]
    # range window: [a, a + 0.35], with a moving 0 -> 0.65 -> 0
    lo_keys = [0.0, 0.3, 0.65, 0.3, 0.0]
    for r in range(rows):
        for c in range(cols):
            v = values[r][c]
            inside = [lo <= v <= lo + 0.35 for lo in lo_keys]
            frames = "".join(f"{pct(i / 4)}{{opacity:{1 if ins else .15}}}" for i, ins in enumerate(inside))
            name = f"e{r}_{c}"
            css.append(f"@keyframes {name}{{{frames}}}.{name}{{animation:{name} {LOOP}s ease-in-out infinite}}")
            body.append(rect(3 + c * P, 3 + r * P, t["rarity"][3 + int(v * 3.99)], name))
    sy = 3 + (rows + 1) * P + 3                # the stat scale and its sliding window
    body.append(f'<rect x="3" y="{sy}" width="{cols * P - GAP}" height="5" rx="2.5" fill="{t["muted"]}"/>')
    span = (cols * P - GAP) * 0.35
    xs = [3 + (cols * P - GAP - 0) * lo for lo in lo_keys]
    frames = "".join(f"{pct(i / 4)}{{transform:translateX({x:.1f}px)}}" for i, x in enumerate(xs))
    css.append(f"@keyframes win{{{frames}}}.win{{animation:win {LOOP}s ease-in-out infinite}}")
    body.append(f'<rect class="win" x="0" y="{sy - 3}" width="{span:.1f}" height="11" rx="5.5" fill="{t["good"]}" opacity=".85"/>')
    return svg(w, h, "Add a stat range: listings outside it fade out", css, body)


def forecast(t) -> str:
    """Two rows of cells grow like error bars: the baseline first, then a model with a shorter bar;
    the shorter one gets a tick, meaning the model is used."""
    cols = 14
    w, h = cols * P + 6 + 2 * P, 2 * P + 6 + 4
    css, body = [], []
    for r, (length, color, start) in enumerate(((12, t["base"], 0.05), (8, t["rarity"][3], 0.35))):
        y = 3 + r * (P + 4)
        for c in range(cols):
            body.append(rect(3 + c * P, y, t["empty"]))
        for c in range(length):
            on = start + 0.25 * c / length
            name = f"b{r}_{c}"
            css.append(f"@keyframes {name}{{0%,{pct(on)}{{opacity:0}}{pct(on + 0.015)},90%{{opacity:1}}100%{{opacity:0}}}}"
                       f".{name}{{animation:{name} {LOOP}s linear infinite}}")
            body.append(rect(3 + c * P, y, color, name))
    tx, ty = 3 + cols * P + 4, 3 + (P + 4) + 1      # tick next to the model's (shorter = better) bar
    css.append(f"@keyframes tick{{0%,68%{{opacity:0;transform:scale(.4)}}72%,90%{{opacity:1;transform:scale(1)}}100%{{opacity:0}}}}"
               ".tick{animation:tick 6s ease-out infinite;transform-box:fill-box;transform-origin:center}")
    body.append(f'<path class="tick" d="M{tx} {ty + 5} l4 4 l8 -9" fill="none" stroke="{t["good"]}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
    return svg(w, h, "Backtest: a model is used only when its error is clearly smaller than the baseline", css, body)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "assets"
    for suffix, theme in THEMES.items():
        for name, fn in (("how-collect", collect), ("how-filter", filt), ("how-forecast", forecast)):
            path = out / f"{name}{suffix}.svg"
            path.write_text(fn(theme), encoding="utf-8")
            print(path.name, f"{path.stat().st_size // 1024} KB")
