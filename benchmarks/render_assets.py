"""Render benchmark preview SVGs from data — a drift check, not the published art.

The published benchmark assets in ``assets/`` were redesigned by hand (the
parchment/Vitruvian system) and are the source of truth for the README. This tool
still renders the same *numbers* — from ``benchmarks/manifest.json`` (track catalog)
and the dated ``benchmarks/reference/live_snapshot.json`` (Live figures) — into a set
of **preview** SVGs under ``benchmarks/results/`` so you can diff the data against the
published art and catch a stale number without overwriting the redesigned files.

    python benchmarks/render_assets.py            # writes preview SVGs to benchmarks/results/

To refresh the published art, update the hand-authored SVGs in ``assets/`` (and the
snapshot they cite); this tool is the data-side check on those numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifest.json"
SNAPSHOT = ROOT / "benchmarks" / "reference" / "live_snapshot.json"
_PREVIEW = ROOT / "benchmarks" / "results"
OUT = _PREVIEW / "preview-benchmark-platform.svg"
OUT_PLANE = _PREVIEW / "preview-benchmark-plane.svg"
OUT_UPLIFT = _PREVIEW / "preview-benchmark-uplift.svg"
OUT_H2H = _PREVIEW / "preview-benchmark-headtohead.svg"
OUT_REASONING = _PREVIEW / "preview-benchmark-reasoning.svg"


# Shared parchment/ink palette (id prefix per-asset to avoid gradient id clashes).
def _palette(pfx: str) -> str:
    return f"""  <defs>
    <radialGradient id="{pfx}_parch" cx="40%" cy="26%" r="88%">
      <stop offset="0%" stop-color="#FBF4E2"/>
      <stop offset="62%" stop-color="#F3E8CC"/>
      <stop offset="100%" stop-color="#E7D6AE"/>
    </radialGradient>
    <linearGradient id="{pfx}_gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E6CC78"/>
      <stop offset="52%" stop-color="#C49A3A"/>
      <stop offset="100%" stop-color="#9A721C"/>
    </linearGradient>
    <linearGradient id="{pfx}_goldtext" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#D9B354"/>
      <stop offset="100%" stop-color="#9A721C"/>
    </linearGradient>
  </defs>"""


def render_uplift(snapshot: dict) -> str:
    """The Track-2 grounded-answer uplift bar chart, from the live snapshot."""
    up = snapshot["track2_uplift"]
    rows = [
        (m["model"], round(m["direct"] * 100), round(m["via_vincio"] * 100)) for m in up["models"]
    ]
    rows.append(
        (
            "AGGREGATE",
            round(up["aggregate"]["direct"] * 100),
            round(up["aggregate"]["via_vincio"] * 100),
        )
    )
    W, H = 860, 384
    y0, y100 = 322.0, 140.0
    scale = (y0 - y100) / 100.0
    n = len(rows)
    span = 824 - 72
    step = span / n
    series = "; ".join(f"{lbl} {d} to {v} percent" for lbl, d, v in rows)
    desc = (
        "Bar chart of grounded-answer accuracy on 15 company-specific questions across current "
        f"state-of-the-art models, direct vs through Vincio: {series}. Every routed answer is cited."
    )
    parts = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-labelledby="up_t up_d">',
        '  <title id="up_t">Grounded-answer accuracy: the same model, direct vs. routed through Vincio</title>',
        f'  <desc id="up_d">{escape(desc)}</desc>',
        _palette("up"),
        f'  <rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="16" fill="url(#up_parch)" stroke="#C49A3A" stroke-width="1.5"/>',
        f'  <rect x="6.5" y="6.5" width="{W - 13}" height="{H - 13}" rx="12" fill="none" stroke="#B98B2E" stroke-width="1" opacity="0.4"/>',
        "  <g font-family=\"Georgia, 'Times New Roman', serif\">",
        '    <text x="40" y="46" font-size="12.5" letter-spacing="3" fill="#B98B2E">TRACK 2 &#183; ORCHESTRATOR UPLIFT</text>',
        '    <text x="40" y="78" font-size="26" font-weight="700" fill="url(#up_goldtext)">Grounded-answer accuracy</text>',
        f'    <text x="40" y="102" font-size="13" fill="#6B5836">Same model, direct vs. routed through Vincio &#183; '
        f"15 questions no model can know from pretraining &#183; {len(up['models'])} current SOTA models &#215; 2 runs (live)</text>",
        '    <rect x="600" y="36" width="14" height="14" rx="2" fill="#B3A079"/>',
        '    <text x="620" y="47" font-size="13" fill="#5B4A2E">Direct</text>',
        '    <rect x="686" y="36" width="14" height="14" rx="2" fill="url(#up_gold)"/>',
        '    <text x="706" y="47" font-size="13" fill="#5B4A2E">Through Vincio</text>',
        '    <g stroke="#5B4A2E" stroke-width="1" opacity="0.16">',
    ]
    for pct in (0, 25, 50, 75, 100):
        yy = y0 - pct * scale
        parts.append(f'      <line x1="72" y1="{yy:.1f}" x2="824" y2="{yy:.1f}"/>')
    parts.append("    </g>")
    parts.append('    <g font-size="10.5" fill="#8A7551" text-anchor="end">')
    for pct in (0, 25, 50, 75, 100):
        yy = y0 - pct * scale
        parts.append(f'      <text x="64" y="{yy + 3:.1f}">{pct}{"%" if pct == 100 else ""}</text>')
    parts.append("    </g>")
    parts.append(
        f'    <line x1="72" y1="{y0}" x2="824" y2="{y0}" stroke="#3A2E1C" stroke-width="1.4" opacity="0.55"/>'
    )
    for i, (label, direct, via) in enumerate(rows):
        cx = 72 + step * (i + 0.5)
        dh, vh = direct * scale, via * scale
        agg = label == "AGGREGATE"
        stroke = ' stroke="#9A721C" stroke-width="1"' if agg else ""
        label_weight = 'font-weight="700" ' if agg else ""
        label_color = "9A721C" if agg else "5B4A2E"
        label_size = 11 if len(label) > 14 else 11.5
        parts += [
            f'    <rect x="{cx - 45:.1f}" y="{y0 - dh:.1f}" width="40" height="{dh:.2f}" fill="#B3A079"/>',
            f'    <rect x="{cx + 5:.1f}" y="{y0 - vh:.1f}" width="40" height="{vh:.2f}" rx="3" fill="url(#up_gold)"{stroke}/>',
            f'    <text x="{cx - 25:.1f}" y="{y0 - dh - 5:.1f}" font-size="10" fill="#8A7551" text-anchor="middle">{direct}%</text>',
            f'    <text x="{cx + 25:.1f}" y="{y0 - vh - 7:.1f}" font-size="{16 if agg else 15}" font-weight="700" fill="#9A721C" text-anchor="middle">{via}%</text>',
            f'    <text x="{cx:.1f}" y="340" font-size="{label_size}" {label_weight}'
            f'fill="#{label_color}" text-anchor="middle">{escape(label)}</text>',
        ]
    parts += [
        '    <text x="40" y="368" font-size="10.5" fill="#8A7551">Direct = the model alone &#183; Through Vincio = '
        "the same model with retrieval &amp; grounding &#183; every Vincio answer is cited &#183; 14&#8211;30&#215; cheaper per correct answer.</text>",
        "  </g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def render_plane(manifest: dict) -> str:
    """The Track-1 model plane: the public-benchmark catalog by niche, from the manifest."""
    catalog = manifest["tracks"]["model"]["catalog"]
    total = catalog["total"]
    cards = []
    for niche in catalog["niches"].values():
        titles = [b["title"] for b in niche["benchmarks"]]
        shown = " · ".join(titles[:3]) + (" · …" if len(titles) > 3 else "")
        cards.append((niche["label"], len(niche["benchmarks"]), shown))
    n_niches = len(cards)
    W, H = 860, 486
    desc = (
        f"Track 1, the model plane: {total} standard public benchmarks across {n_niches} niches, each "
        "carrying an enforced provenance tier — Static (fabricated fixture, gates CI), Recorded "
        "(hash-pinned real slice, gates CI), and Live (a live state-of-the-art model, reported)."
    )
    parts = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-labelledby="pl_t pl_d">',
        '  <title id="pl_t">Track 1 — the model benchmark plane</title>',
        f'  <desc id="pl_d">{escape(desc)}</desc>',
        _palette("pl"),
        f'  <rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="16" fill="url(#pl_parch)" stroke="#C49A3A" stroke-width="1.5"/>',
        f'  <rect x="6.5" y="6.5" width="{W - 13}" height="{H - 13}" rx="12" fill="none" stroke="#B98B2E" stroke-width="1" opacity="0.4"/>',
        "  <g font-family=\"Georgia, 'Times New Roman', serif\">",
        '    <text x="40" y="46" font-size="12.5" letter-spacing="3" fill="#B98B2E">TRACK 1 &#183; MODEL &#183; PUBLIC BENCHMARKS</text>',
        '    <text x="40" y="78" font-size="26" font-weight="700" fill="url(#pl_goldtext)">A model on the public benchmarks, tier-honest</text>',
        f'    <text x="40" y="102" font-size="13" fill="#6B5836">{total} benchmarks &#183; {n_niches} niches &#183; '
        "in-process, offline-first, never a hosted leaderboard.</text>",
    ]
    ly = 122
    for i, (code, name, blurb) in enumerate(_TIERS[::-1]):  # S, R, L order for the model track
        x = 40 + i * 266
        parts += [
            f'    <rect x="{x}" y="{ly}" width="253" height="46" rx="9" fill="#FCF6E8" stroke="url(#pl_gold)" stroke-width="1"/>',
            f'    <circle cx="{x + 26}" cy="{ly + 23}" r="14" fill="url(#pl_gold)"/>',
            f'    <text x="{x + 26}" y="{ly + 28}" font-size="16" font-weight="700" fill="#FBF4E2" text-anchor="middle">{code}</text>',
            f'    <text x="{x + 50}" y="{ly + 20}" font-size="14" font-weight="700" fill="#3A2E1C">{name}</text>',
            f'    <text x="{x + 50}" y="{ly + 37}" font-size="10.5" fill="#8A7551">{escape(blurb)}</text>',
        ]
    cols, cw, ch, cg, rg, gx0, gy0 = 5, 148, 118, 12, 14, 40, 196
    for idx, (label, count, shown) in enumerate(cards):
        row, col = divmod(idx, cols)
        x, y = gx0 + col * (cw + cg), gy0 + row * (ch + rg)
        parts += [
            f'    <rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="#FCF6E8" stroke="url(#pl_gold)" stroke-width="1"/>',
            f'    <text x="{x + cw // 2}" y="{y + 40}" font-size="30" font-weight="700" fill="url(#pl_goldtext)" text-anchor="middle">{count}</text>',
            f'    <text x="{x + cw // 2}" y="{y + 62}" font-size="13" font-weight="700" fill="#3A2E1C" text-anchor="middle">{escape(label)}</text>',
        ]
        for j, line in enumerate(_wrap(shown, 22)[:3]):
            parts.append(
                f'    <text x="{x + cw // 2}" y="{y + 82 + j * 13}" font-size="9.5" '
                f'fill="#8A7551" text-anchor="middle">{escape(line)}</text>'
            )
    parts += [
        f'    <text x="{W // 2}" y="{H - 22}" font-size="11.5" fill="#6B5836" text-anchor="middle">'
        "One pluggable contract &#183; reusable metrics &#183; the engine refuses to print a higher tier than the inputs support.</text>",
        "  </g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def render_headtohead(snapshot: dict) -> str:
    """The Track-3 feature head-to-head headline plates, from the live snapshot."""
    plates = snapshot["track3_feature_headline"]["plates"]
    W, H = 860, 300
    plate_desc = "; ".join(f"{p['big']} {p['line']} {p['sub']}" for p in plates)
    desc = f"Four head-to-head stat plates measured live against the real competitor library: {plate_desc}."
    parts = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-labelledby="h2h_t h2h_d">',
        '  <title id="h2h_t">Track 3: a Vincio feature vs. the library you would reach for</title>',
        f'  <desc id="h2h_d">{escape(desc)}</desc>',
        _palette("h2h"),
        f'  <rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="16" fill="url(#h2h_parch)" stroke="#C49A3A" stroke-width="1.5"/>',
        f'  <rect x="6.5" y="6.5" width="{W - 13}" height="{H - 13}" rx="12" fill="none" stroke="#B98B2E" stroke-width="1" opacity="0.4"/>',
        "  <g font-family=\"Georgia, 'Times New Roman', serif\">",
        '    <text x="40" y="46" font-size="12.5" letter-spacing="3" fill="#B98B2E">TRACK 3 &#183; FEATURE &#183; MEASURED LIVE</text>',
        '    <text x="40" y="78" font-size="26" font-weight="700" fill="url(#h2h_goldtext)">A Vincio feature vs. the library you would reach for</text>',
        '    <text x="40" y="102" font-size="13" fill="#6B5836">Each number measured live from both sides; a missing competitor is skipped, never faked; latency is machine-relative.</text>',
    ]
    for i, p in enumerate(plates):
        x = 40 + i * 200
        parts += [
            f'    <rect x="{x}" y="132" width="180" height="128" rx="10" fill="#FCF6E8" stroke="url(#h2h_gold)" stroke-width="1"/>',
            f'    <text x="{x + 90}" y="186" font-size="{26 if len(p["big"]) > 6 else 30}" font-weight="700" '
            f'fill="url(#h2h_goldtext)" text-anchor="middle">{escape(p["big"])}</text>',
            f'    <text x="{x + 90}" y="212" font-size="12.5" fill="#3A2E1C" text-anchor="middle">{escape(p["line"])}</text>',
            f'    <text x="{x + 90}" y="234" font-size="10.5" fill="#8A7551" text-anchor="middle">{escape(p["sub"])}</text>',
        ]
    parts += ["  </g>", "</svg>"]
    return "\n".join(parts) + "\n"


def render_reasoning(snapshot: dict) -> str:
    """The universal-reasoning live capability plates, from the dated snapshot."""
    live = snapshot["universal_reasoning_live"]
    small = live["small_model"]
    multilingual = live["multilingual_routing"]
    direct = round(small["direct_accuracy"] * 100)
    via = round(small["via_vincio_accuracy"] * 100)
    cards = [
        (
            f"{direct}% -> {via}%",
            "exact task accuracy",
            f"n={small['cases']} · {small['model']}",
        ),
        (
            f"{small['deterministically_verified_answers']} + {small['bounded_corrections_accepted']}",
            "verified · repaired",
            "deterministic offline kernels",
        ),
        (
            f"{small['fabricated_sources_delivered'] + small['current_fact_overclaims_delivered']}",
            "fabrications or overclaims",
            f"{small['web_verified_cases']} web-verified · {small['safe_refusals']} safe refusals",
        ),
        (
            f"{multilingual['correct']} / {multilingual['cases']}",
            "multilingual routes correct",
            " · ".join(code.upper() for code in multilingual["languages"]),
        ),
    ]
    desc = "Tier-L universal-reasoning capability sample: " + "; ".join(
        f"{big} {line} ({sub})" for big, line, sub in cards
    )
    parts = [
        '<svg width="860" height="310" viewBox="0 0 860 310" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-labelledby="rs_t rs_d">',
        '  <title id="rs_t">Universal reasoning live capability sample</title>',
        f'  <desc id="rs_d">{escape(desc)}</desc>',
        _palette("rs"),
        '  <rect x="1.5" y="1.5" width="857" height="307" rx="16" fill="url(#rs_parch)" '
        'stroke="#C49A3A" stroke-width="1.5"/>',
        "  <g font-family=\"Georgia, 'Times New Roman', serif\">",
        '    <text x="40" y="44" font-size="12.5" letter-spacing="3" fill="#B98B2E">'
        f"UNIVERSAL REASONING &#183; TIER-L &#183; {escape(live['captured'])}</text>",
        '    <text x="40" y="76" font-size="25" font-weight="700" fill="url(#rs_goldtext)">'
        "Reasoning quality, verification and language routing</text>",
    ]
    for index, (big, line, sub) in enumerate(cards):
        x = 40 + index * 200
        parts += [
            f'    <rect x="{x}" y="104" width="180" height="130" rx="10" fill="#FCF6E8" '
            'stroke="url(#rs_gold)"/>',
            f'    <text x="{x + 90}" y="158" font-size="30" font-weight="700" '
            f'fill="url(#rs_goldtext)" text-anchor="middle">{escape(big)}</text>',
            f'    <text x="{x + 90}" y="184" font-size="12" fill="#3A2E1C" '
            f'text-anchor="middle">{escape(line)}</text>',
            f'    <text x="{x + 90}" y="207" font-size="9.5" fill="#8A7551" '
            f'text-anchor="middle">{escape(sub)}</text>',
        ]
    parts += [
        '    <text x="430" y="270" font-size="10.5" fill="#6B5836" text-anchor="middle">'
        "Small reviewed sample; refusal is safe but not scored correct; reported, never CI-gated.</text>",
        "  </g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


_TIERS = [
    ("L", "Live", "the real thing ran end to end"),
    ("R", "Recorded", "a hash-pinned replay"),
    ("S", "Static / Mockup", "offline, reproducible, gates CI"),
]


# (track key, number, unit, one-liner, command)
def _cards(tracks: dict) -> list[tuple[str, str, str, str, str, str]]:
    order = ["model", "uplift", "feature"]
    labels = {"model": "1 · Model", "uplift": "2 · Uplift", "feature": "3 · Feature"}
    units = {"model": "benchmarks", "uplift": "uplift benchmarks", "feature": "feature contests"}
    lines = {
        "model": "a model on the public benchmarks",
        "uplift": "the same model, Vincio-routed vs direct",
        "feature": "a Vincio feature vs a competitor library",
    }
    out = []
    for k in order:
        t = tracks[k]
        tiers = "·".join(t["tiers"])
        out.append(
            (labels[k], str(t["catalog"]["total"]), units[k], lines[k], f"vincio bench {k}", tiers)
        )
    return out


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def render(manifest: dict) -> str:
    tracks = manifest["tracks"]
    cards = _cards(tracks)
    W, H = 860, 430
    desc = (
        "Three benchmark tracks under one provenance-tier honesty contract: Model (a model on the "
        "public benchmarks), Uplift (the same model routed through Vincio vs direct), and Feature (a "
        "Vincio feature vs a competitor library). Each supports a Live run and an offline mockup; a "
        "lower tier can never print a higher tier label."
    )
    parts: list[str] = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="pf_t pf_d">',
        '  <title id="pf_t">The Vincio benchmark platform</title>',
        f'  <desc id="pf_d">{escape(desc)}</desc>',
        _palette("pf"),
        f'  <rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="16" '
        'fill="url(#pf_parch)" stroke="#C49A3A" stroke-width="1.5"/>',
        f'  <rect x="6.5" y="6.5" width="{W - 13}" height="{H - 13}" rx="12" '
        'fill="none" stroke="#B98B2E" stroke-width="1" opacity="0.4"/>',
        "  <g font-family=\"Georgia, 'Times New Roman', serif\">",
        '    <text x="40" y="46" font-size="12.5" letter-spacing="3" fill="#B98B2E">'
        "THE VINCIO BENCHMARK PLATFORM &#183; TIER-HONEST</text>",
        '    <text x="40" y="78" font-size="26" font-weight="700" fill="url(#pf_goldtext)">'
        "Three tracks, one honesty contract</text>",
        '    <text x="40" y="102" font-size="13" fill="#6B5836">'
        "Every number carries a provenance tier &#183; each track runs live or as an offline mockup "
        "&#183; nothing fabricated.</text>",
    ]

    # Three track cards.
    card_w, gap, x0, y0, card_h = 251, 13, 40, 122, 176
    for i, (label, number, unit, line, cmd, tiers) in enumerate(cards):
        x = x0 + i * (card_w + gap)
        parts += [
            f'    <rect x="{x}" y="{y0}" width="{card_w}" height="{card_h}" rx="10" '
            'fill="#FCF6E8" stroke="url(#pf_gold)" stroke-width="1"/>',
            f'    <text x="{x + 18}" y="{y0 + 30}" font-size="13" font-weight="700" '
            f'letter-spacing="1" fill="#B98B2E">{escape(label.upper())}</text>',
            f'    <text x="{x + 18}" y="{y0 + 78}" font-size="44" font-weight="700" '
            f'fill="url(#pf_goldtext)">{number}</text>',
            f'    <text x="{x + 18}" y="{y0 + 98}" font-size="12" fill="#8A7551">{escape(unit)}</text>',
        ]
        for j, wln in enumerate(_wrap(line, 30)[:3]):
            parts.append(
                f'    <text x="{x + 18}" y="{y0 + 122 + j * 15}" font-size="11.5" '
                f'fill="#3A2E1C">{escape(wln)}</text>'
            )
        parts += [
            f'    <text x="{x + 18}" y="{y0 + card_h - 30}" font-size="11" font-weight="700" '
            f'fill="#9A721C">tiers {escape(tiers)}</text>',
            f'    <text x="{x + 18}" y="{y0 + card_h - 12}" font-size="11" '
            f'font-family="ui-monospace, monospace" fill="#6B5836">{escape(cmd)}</text>',
        ]

    # Tier legend row.
    ly = 322
    pill_w = 253
    for i, (code, name, blurb) in enumerate(_TIERS):
        x = 40 + i * (pill_w + 13)
        parts += [
            f'    <rect x="{x}" y="{ly}" width="{pill_w}" height="44" rx="9" '
            'fill="#FCF6E8" stroke="url(#pf_gold)" stroke-width="1"/>',
            f'    <circle cx="{x + 24}" cy="{ly + 22}" r="13" fill="url(#pf_gold)"/>',
            f'    <text x="{x + 24}" y="{ly + 27}" font-size="15" font-weight="700" '
            f'fill="#FBF4E2" text-anchor="middle">{code}</text>',
            f'    <text x="{x + 46}" y="{ly + 19}" font-size="13" font-weight="700" fill="#3A2E1C">{name}</text>',
            f'    <text x="{x + 46}" y="{ly + 35}" font-size="10" fill="#8A7551">{escape(blurb)}</text>',
        ]

    parts += [
        f'    <text x="{W // 2}" y="{H - 20}" font-size="11.5" fill="#6B5836" text-anchor="middle">'
        'One command &#183; <tspan font-family="ui-monospace, monospace">vincio bench model | uplift | '
        "feature</tspan> &#183; the engine refuses to print a higher tier than the inputs support.</text>",
        "  </g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    OUT.write_text(render(manifest), encoding="utf-8")
    OUT_PLANE.write_text(render_plane(manifest), encoding="utf-8")
    OUT_UPLIFT.write_text(render_uplift(snapshot), encoding="utf-8")
    OUT_H2H.write_text(render_headtohead(snapshot), encoding="utf-8")
    OUT_REASONING.write_text(render_reasoning(snapshot), encoding="utf-8")
    tracks = manifest["tracks"]
    print(
        f"wrote {OUT.name}, {OUT_PLANE.name}, {OUT_UPLIFT.name}, {OUT_H2H.name}, "
        f"{OUT_REASONING.name} — 3 tracks "
        f"(model {tracks['model']['catalog']['total']}, uplift {tracks['uplift']['catalog']['total']}, "
        f"feature {tracks['feature']['catalog']['total']}); uplift snapshot {snapshot['captured']}"
    )


if __name__ == "__main__":
    main()
