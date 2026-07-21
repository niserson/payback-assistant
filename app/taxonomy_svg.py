"""Server-rendered SVG tree of the three partner taxonomies (served at /taxonomy-tree).

Generated from app.catalog.PARTNERS at startup, so the visual can never drift from
the data that is actually indexed.
"""

from html import escape

from .catalog import PARTNERS

_COLORS = {
    "dm": ("#e3f0fa", "#0a6cbd"),
    "edeka": ("#fdf6cf", "#6b5900"),
    "amazon": ("#fff0da", "#9c5f00"),
}

_ROW = 19          # px per product row
_CAT_GAP = 14      # gap between categories
_PARTNER_GAP = 46  # gap between partner sections
_WIDTH = 980


def _svg() -> str:
    parts = []
    y = 30
    for partner, meta in PARTNERS.items():
        fill, stroke = _COLORS[partner]
        section_top = y
        n_products = 0
        for category, items in meta["taxonomy"].items():
            cat_top = y
            for name, _brand, price, _unit, _tags, _pop in items:
                cy = y + _ROW // 2
                parts.append(f'<circle cx="548" cy="{cy}" r="3" fill="{stroke}"/>')
                parts.append(f'<text x="560" y="{cy + 4}" font-size="12" fill="#33415c">'
                             f'{escape(name)} <tspan fill="#8a93a6">· {price:.2f} €</tspan></text>')
                y += _ROW
                n_products += 1
            cat_mid = (cat_top + y) / 2
            parts.append(f'<rect x="290" y="{cat_mid - 14}" width="225" height="28" rx="8" '
                         f'fill="#fff" stroke="{stroke}"/>')
            parts.append(f'<text x="402" y="{cat_mid + 4}" text-anchor="middle" font-size="12" '
                         f'font-weight="600" fill="{stroke}">{escape(category)} ({len(items)})</text>')
            for i in range(len(items)):
                py = cat_top + i * _ROW + _ROW // 2
                parts.append(f'<path d="M 515 {cat_mid} C 532 {cat_mid}, 528 {py}, 544 {py}" '
                             f'fill="none" stroke="{stroke}" stroke-width="1" opacity="0.55"/>')
            parts.append(f'<path d="M 220 {{PMID_{partner}}} C 258 {{PMID_{partner}}}, 252 {cat_mid}, 290 {cat_mid}" '
                         f'fill="none" stroke="{stroke}" stroke-width="1.4" opacity="0.7"/>')
            y += _CAT_GAP
        section_mid = (section_top + y - _CAT_GAP) / 2
        parts.append(f'<rect x="20" y="{section_mid - 34}" width="200" height="68" rx="10" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="120" y="{section_mid - 8}" text-anchor="middle" font-size="14" '
                     f'font-weight="700" fill="{stroke}">{escape(meta["label"])}</text>')
        parts.append(f'<text x="120" y="{section_mid + 10}" text-anchor="middle" font-size="11" '
                     f'fill="{stroke}">{len(meta["taxonomy"])} categories · {n_products} products</text>')
        # resolve the partner-connector midpoints deferred above
        parts = [p.replace(f"{{PMID_{partner}}}", f"{section_mid:.0f}") for p in parts]
        y += _PARTNER_GAP
    height = y
    return (f'<svg viewBox="0 0 {_WIDTH} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="system-ui, sans-serif">{"".join(parts)}</svg>')


def render_page() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PAYBACK Assistant — Taxonomy Tree</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f4f6fb;
         color: #1a2333; margin: 0; padding: 24px 16px 48px; }}
  .wrap {{ max-width: 1020px; margin: 0 auto; }}
  h1 {{ color: #003eb0; font-size: 1.3rem; }}
  p.sub {{ color: #5b6579; font-size: .92rem; }}
  .card {{ background: #fff; border: 1px solid #e2e7f0; border-radius: 14px; padding: 10px; overflow-x: auto; }}
  svg {{ display: block; margin: 0 auto; min-width: 940px; }}
  a {{ color: #003eb0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Partner taxonomies — rendered live from the indexed catalog</h1>
  <p class="sub">Three separate category trees, one per partner ecosystem, normalized into a shared
  schema at ingestion. <a href="/">← demo UI</a> · <a href="/architecture">architecture</a> ·
  <a href="/taxonomy">JSON version</a></p>
  <div class="card">{_svg()}</div>
</div>
</body>
</html>"""
