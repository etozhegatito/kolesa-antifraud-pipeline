# -*- coding: utf-8 -*-
"""Build the HTML cards shown to the anomaly annotator.

Rendering is intentionally isolated from queue construction and journal
writes. It transforms prepared rows into markup and can be tested without
Postgres or filesystem mutations.
"""

import html
import json

import pandas as pd

from kz.report.label_cards.journal import LABELS_CSV

OUT_HTML = "data/eda/label_cards.html"

# Decision guidance for each flag. The key principle is that fraud means
# deception, not poor vehicle condition. An honestly described wreck is legit.
FLAG_HELP = {
    "price_anomaly_low": (
        "The price is far below the market for this model and year.",
        "fraud — when the reason is hidden: the text reports no issue, photos "
        "look intact, and there is no accident badge. This can be bait for a "
        "vehicle that is not actually available at the advertised price.",
        "legit — when the low price is explained by visible or disclosed crash "
        "damage, corrosion, a missing engine, an instalment down payment, or "
        "genuinely poor age and condition.",
    ),
    "young_car_cheap": (
        "A recent vehicle is priced like an old one.",
        "fraud — when nothing explains the price: intact photos and no issue in the text.",
        "legit — when crash damage or a non-running condition is clearly "
        "disclosed by the badge, photos, or description.",
    ),
    "possible_repost": (
        "The same vehicle may have been listed again.",
        "fraud — when one vehicle has conflicting prices, mileage, or years, "
        "suggesting search manipulation or price bait.",
        "legit — when this is a normal repost and colour, mileage, and price "
        "remain consistent. A duplicate is not automatically deceptive.",
    ),
    "shared_photo_diff_car": (
        "The same photo appears in listings with different attributes.",
        "fraud — when the photo shows a different vehicle and was copied from elsewhere.",
        "legit — when it is a dealer template or the same vehicle in two "
        "listings. Inspect visually because pHash has limitations around crops.",
    ),
    "used_but_zero_mileage": (
        "A used vehicle has zero recorded mileage.",
        "fraud — only when the seller explicitly misrepresents an obviously "
        "used vehicle as new or zero-kilometre.",
        "legit / unknown — most cases are missing data rather than deception.",
    ),
    "cheap_and_urgent": (
        "The listing combines a low price with urgency language.",
        "fraud — when unexplained cheapness and urgency pressure the buyer for "
        "a vehicle that may not be available.",
        "legit — when the seller clearly explains the urgency.",
    ),
}

DEAD_HOSTS = {"alakt-photos-kl.kcdn.kz"}  # retired around August 2026


def photo_src(ad_id: str, position: int, url: str, serve_mode: bool) -> str | None:
    """Choose the browser source for a listing photo.

    Prefer a local copy because it loads quickly and survives CDN retirement.
    One historical host is gone; return ``None`` for those URLs so the card
    explicitly reports unavailable photos instead of showing empty frames.
    """
    from kz.collect.photo_fetch import local_path

    p = local_path(ad_id, position)
    if p.exists():
        # Server mode uses the photo route; offline export uses a path relative
        # to data/eda/, where the generated page is written.
        return (
            f"/photos/{p.relative_to('data/photos')}"
            if serve_mode
            else f"../photos/{p.relative_to('data/photos')}"
        )
    host = url.split("/")[2] if "//" in url else ""
    return None if host in DEAD_HOSTS else url


def money(v) -> str:
    """Format a price for people and represent missing values with a dash.

    Use millions only from one million upward; ``240,000 ₸`` is clearer than
    ``0.24M ₸`` and low-price candidates are common in this queue.
    """
    if pd.isna(v) or v is None:
        return "—"
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.2f}".rstrip("0").rstrip(".") + "M ₸"
    return f"{int(v):,}".replace(",", " ") + " ₸"


def fmt(v) -> str:
    """Format a table value, using a dash for missing data and escaping text."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    if isinstance(v, float) and float(v).is_integer():
        v = int(v)
    return html.escape(str(v))


# Bands for listing price divided by Kolesa's reference price. Narrow bands
# avoid contradictory edge cases such as “60% of average — normal price.”
PRICE_BANDS = [
    (0.60, "far below the market"),
    (0.85, "noticeably below average"),
    (1.15, "within the average range"),
    (1.40, "above average"),
    (float("inf"), "well above the market and therefore not low-price bait"),
]


def price_band(ratio: float) -> str:
    """Describe a listing/reference-price ratio in plain language."""
    for limit, label in PRICE_BANDS:
        if ratio < limit:
            return label
    return PRICE_BANDS[-1][1]


def price_verdict_hint(row) -> str:
    """Cross-check against Kolesa avgPrice; this validates but never trains the model."""
    avg = row.get("kolesa_avg_price")
    price = row.get("price_tenge")
    if pd.isna(avg) or avg is None or float(avg) <= 0 or pd.isna(price):
        return ""
    ratio = float(price) / float(avg)
    return (
        f"<b>{ratio * 100:.0f}% of the Kolesa reference price</b> ({money(avg)}) — "
        f"{price_band(ratio)}"
    )


def card_html(row, idx: int, serve_mode: bool = False) -> str:
    """Render one listing card.

    A large photo is the primary evidence, facts sit beside it, and text comes
    below. Decision guidance is collapsible because it helps early in a session
    but becomes distracting once the protocol is familiar.
    """
    aid = html.escape(str(row["ad_id"]))
    reasons = [r for r in str(row.get("suspicion_reasons") or "").split(";") if r]
    reasons = [p for r in reasons for p in r.split("|")]
    dead = row.get("status") in ("archived", "deleted")
    badge = row.get("page_status_badge")
    has_badge = badge not in (None, "-", "") and not pd.isna(badge)

    raw_photos = row["photos"]
    positions = row.get("photo_positions") or list(range(1, len(raw_photos) + 1))
    pairs = [
        (photo_src(str(row["ad_id"]), pos, u, serve_mode), u)
        for pos, u in zip(positions, raw_photos)
    ]
    photos = [src for src, _ in pairs if src]
    n_dead = sum(1 for src, _ in pairs if src is None)
    if photos:
        thumbs = "".join(
            f'<button class="thumb{" on" if i == 0 else ""}" data-i="{i}" '
            f'aria-label="photo {i + 1}">'
            f'<img loading="lazy" src="{html.escape(u)}" alt=""></button>'
            for i, u in enumerate(photos)
        )
        gallery = (
            f"<div class=\"gal\" data-photos='{html.escape(json.dumps(photos))}'>"
            f'  <div class="hero"><img src="{html.escape(photos[0])}" alt="photo 1">'
            f'    <span class="zoom">click to enlarge</span>'
            f'    <span class="counter"><b>1</b>/{len(photos)}</span></div>'
            f'  <div class="thumbs">{thumbs}</div>'
            f"</div>"
        )
    elif n_dead:
        gallery = (
            '<div class="gal"><div class="empty">Photos are unavailable: '
            f"the historical Kolesa image host was retired ({n_dead} files). "
            "Use the text and structured facts for this decision.</div></div>"
        )
    else:
        gallery = '<div class="gal"><div class="empty">No photo URLs were stored.</div></div>'

    help_blocks = ""
    for r in reasons:
        if r in FLAG_HELP:
            what, fr, lg = FLAG_HELP[r]
            help_blocks += (
                f'<div class="help"><code>{html.escape(r)}</code>'
                f'<p class="hwhat">{html.escape(what)}</p>'
                f'<p class="hfraud"><span>fraud</span>{fr[7:] if fr.startswith("fraud —") else fr}</p>'
                f'<p class="hlegit"><span>legit</span>'
                f"{lg.split('—', 1)[1] if '—' in lg else lg}</p></div>"
            )
    helps = (
        f'<details class="helps"><summary>How to evaluate these flags</summary>'
        f"{help_blocks}</details>"
        if help_blocks
        else ""
    )

    facts = [
        ("Price", money(row.get("price_tenge")), "big"),
        ("Year", fmt(row.get("year")), ""),
        ("Mileage, listing", fmt(row.get("mileage_km")), ""),
        ("Mileage, detail page", fmt(row.get("page_mileage_km")), ""),
        ("Engine", f"{fmt(row.get('engine_volume'))} · {fmt(row.get('engine_type'))}", ""),
        ("Transmission", fmt(row.get("transmission")), ""),
        ("Body style", fmt(row.get("body_type")), ""),
        ("Colour", fmt(row.get("color")), ""),
        ("Drive · steering", f"{fmt(row.get('drive'))} · {fmt(row.get('steering'))}", ""),
        ("Customs cleared", fmt(row.get("customs_cleared")), ""),
        ("Price basis", fmt(row.get("price_basis")), ""),
        ("Condition", fmt(row.get("page_condition")), ""),
        ("VIN evidence", fmt(row.get("has_vin")), ""),
        (
            "price_z",
            fmt(round(float(row["price_z"]), 2) if pd.notna(row.get("price_z")) else None),
            "",
        ),
        ("Views", fmt(row.get("views_count")), ""),
        ("Posted", fmt(row.get("posted_date")), ""),
    ]
    facts_html = "".join(
        f'<div class="f {cls}"><dt>{k}</dt><dd>{v}</dd></div>' for k, v, cls in facts
    )

    texts = ""
    for label, key in [
        ("Listing summary", "description"),
        ("Seller comment", "seller_comment"),
        ("Options", "options_text"),
    ]:
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip():
            texts += (
                f'<div class="txt"><div class="tl">{label}</div>'
                f'<div class="tb">{html.escape(str(v))}</div></div>'
            )
    if not texts:
        texts = '<div class="empty">No text is available; use photos and price.</div>'

    notes = ""
    dmg = row.get("damage_keywords")
    if dmg and not (isinstance(dmg, float) and pd.isna(dmg)) and str(dmg).strip():
        notes += (
            f'<div class="note legit-note"><b>{html.escape(str(dmg))}</b>'
            f"<span>the seller disclosed the issue, which supports legit</span></div>"
        )
    if has_badge:
        notes += (
            f'<div class="note legit-note"><b>site badge: '
            f"{html.escape(str(badge))}</b><span>the site discloses a "
            f"problematic condition, which explains the low price</span></div>"
        )

    hint = price_verdict_hint(row)
    if hint:
        notes += f'<div class="note price-note">{hint}</div>'

    STRATUM_HELP = {
        "rule_positive": (
            "Flagged by rules",
            "Question: is the flag correct—deception or an explained low price?",
        ),
        "residual_candidate": (
            "Model: unexpectedly low price",
            "Rules are silent, but price is below the model's expectation. "
            "Question: deception or a disclosed reason?",
        ),
        "random_control": (
            "Control: not flagged by the detector",
            "Most controls should be legit. Their purpose is to find missed "
            "fraud; without them recall cannot be estimated.",
        ),
    }
    st = str(row.get("stratum") or "")
    stratum_html = ""
    if st in STRATUM_HELP:
        title, hint = STRATUM_HELP[st]
        cls = "s-control" if st == "random_control" else "s-flagged"
        stratum_html = f'<div class="stratum {cls}"><b>{title}</b><span>{hint}</span></div>'

    ev = row.get("existing_verdict")
    ev_html = (
        f'<div class="note done-note">Already labelled: '
        f"<b>{html.escape(str(ev))}</b><span>you can review it again</span></div>"
        if ev and not pd.isna(ev)
        else ""
    )

    status_html = (
        f'<span class="dead">{html.escape(str(row.get("status")))} · the listing '
        f"page is gone, but photos remain available</span>"
        if dead
        else f'<a class="live" href="https://kolesa.kz/a/show/{aid}" target="_blank" '
        f'rel="noreferrer">open on Kolesa ↗<em>uses the IP request budget</em></a>'
    )

    title = (
        f"{html.escape(str(row.get('brand') or ''))} {html.escape(str(row.get('model') or ''))}"
    ).strip()

    return f"""
<article class="card" id="ad{aid}" data-id="{aid}" data-idx="{idx}" data-stratum="{st}">
  <header>
    <div class="ttl">
      <h2>{title} <span class="yr">{fmt(row.get("year"))}</span></h2>
      <div class="meta"><span class="num">#{idx + 1}</span> id {aid} · {status_html}</div>
    </div>
    <div class="flags">
      {"".join(f'<span class="flag">{html.escape(r)}</span>' for r in reasons)}
    </div>
  </header>
  {stratum_html}
  {ev_html}
  <div class="body">
    <div class="left">{gallery}</div>
    <div class="right">
      <dl class="facts">{facts_html}</dl>
      {notes}
    </div>
  </div>
  <div class="texts">{texts}</div>
  {helps}
  <footer class="actions">
    <button class="bfraud" data-v="fraud">fraud<kbd>F</kbd></button>
    <button class="blegit" data-v="legit">legit<kbd>L</kbd></button>
    <button class="bunk" data-v="unknown">unknown<kbd>U</kbd></button>
    <input class="cmt" placeholder="Reason (saved to the comment column)">
    <span class="picked"></span>
  </footer>
</article>"""


def build(rows: pd.DataFrame, serve_mode: bool = False, journal_total: int | None = None) -> str:
    if journal_total is None:
        # Compute this from the journal. Counting only labels among currently
        # displayed rows made completed work appear to disappear whenever the
        # queue was rebuilt to contain mostly unlabelled items.
        from kz.report.label_cards.journal import read_journal

        _, jrows = read_journal()
        journal_total = sum(1 for r in jrows if r.get("verdict") in ("fraud", "legit", "unknown"))

    cards = "".join(card_html(r, i, serve_mode) for i, (_, r) in enumerate(rows.iterrows()))
    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_done = int(rows["existing_verdict"].notna().sum())
    n_left = len(rows) - n_done
    strata = rows.get("stratum", pd.Series(dtype=str)).fillna("").value_counts()
    n_rules = int(strata.get("rule_positive", 0))
    n_residual = int(strata.get("residual_candidate", 0))
    n_control = int(strata.get("random_control", 0))
    n_nophoto = sum(
        1
        for _, r in rows.iterrows()
        if r["photos"]
        and not any(
            photo_src(str(r["ad_id"]), p, u, serve_mode)
            for p, u in zip(r.get("photo_positions") or [], r["photos"])
        )
    )
    mode = (
        "verdicts are saved to the journal"
        if serve_mode
        else "browser draft; journal writes are disabled"
    )
    return (
        TEMPLATE.replace("__CARDS__", cards)
        .replace("__N__", str(len(rows)))
        .replace("__NDEAD__", str(n_dead))
        .replace("__NOPHOTO__", str(n_nophoto))
        .replace("__NDONE__", str(n_done))
        .replace("__NLEFT__", str(n_left))
        .replace("__NRULES__", str(n_rules))
        .replace("__NRESIDUAL__", str(n_residual))
        .replace("__NCONTROL__", str(n_control))
        .replace("__HOME__", ('<a class="count" href="/">← Home</a>' if serve_mode else ""))
        .replace("__SERVER__", "true" if serve_mode else "false")
        .replace("__JOURNAL__", str(journal_total))
        .replace("__MODECLS__", "live" if serve_mode else "draft")
        .replace("__MODE__", mode)
        .replace("__LABELS__", html.escape(LABELS_CSV))
    )


# Keep the template separate from Python strings that contain CSS braces. Plain
# replacement avoids escaping the entire stylesheet as an f-string. A raw
# string also preserves JavaScript ``\n`` without double escaping.
TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market anomaly review</title>
<style>
/* Theme follows the system by default; the button sets data-theme on <html>. */
:root{
  --bg:#0c0f16; --surface:#131824; --surface2:#1a2030; --line:#232b3d;
  --text:#e7eaf2; --muted:#8f98ab; --faint:#5f6878;
  --accent:#7aa7ff; --accent-bg:#14203a;
  --fraud:#ff8b8b; --fraud-bg:#2a1518; --fraud-line:#5e2a2e;
  --legit:#7fe0a5; --legit-bg:#102319; --legit-line:#265f3d;
  --warn:#ffc470; --warn-bg:#241c0e;
  --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px -12px rgba(0,0,0,.6);
}
:root[data-theme="light"]{
  --bg:#f6f7f9; --surface:#ffffff; --surface2:#f2f4f7; --line:#e2e6ed;
  --text:#161a22; --muted:#5f6773; --faint:#98a1af;
  --accent:#2563c9; --accent-bg:#eaf1ff;
  --fraud:#b4232c; --fraud-bg:#fdeeef; --fraud-line:#f2c4c7;
  --legit:#1a7a48; --legit-bg:#eaf7ef; --legit-line:#bde3cb;
  --warn:#8a5a00; --warn-bg:#fdf3e0;
  --shadow:0 1px 2px rgba(16,24,40,.05), 0 8px 24px -14px rgba(16,24,40,.18);
}
@media (prefers-color-scheme: light){
  :root:not([data-theme]){
    --bg:#f6f7f9; --surface:#ffffff; --surface2:#f2f4f7; --line:#e2e6ed;
    --text:#161a22; --muted:#5f6773; --faint:#98a1af;
    --accent:#2563c9; --accent-bg:#eaf1ff;
    --fraud:#b4232c; --fraud-bg:#fdeeef; --fraud-line:#f2c4c7;
    --legit:#1a7a48; --legit-bg:#eaf7ef; --legit-line:#bde3cb;
    --warn:#8a5a00; --warn-bg:#fdf3e0;
    --shadow:0 1px 2px rgba(16,24,40,.05), 0 8px 24px -14px rgba(16,24,40,.18);
  }
}

/* System fonts keep the offline page crisp and dependency-free. Tabular
   numerals keep fact and price columns aligned. */
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text",
    "Segoe UI Variable Text","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,
    "Noto Sans",sans-serif;
  font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;
  font-variant-numeric:tabular-nums; font-kerning:normal;
}
h1,h2{font-weight:600; letter-spacing:-.015em; line-height:1.25; margin:0}
h1{font-size:1.5rem}
h2{font-size:1.2rem}
code,kbd,.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,
  Consolas,"Liberation Mono",monospace}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}

.wrap{max-width:1160px; margin:0 auto; padding:0 20px 80px}

/* Top bar */
.top{position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--bg) 88%,transparent);
     backdrop-filter:saturate(1.4) blur(10px); border-bottom:1px solid var(--line);
     margin:0 -20px 24px; padding:0 20px}
.progress{height:2px; background:var(--line); margin:0 -20px}
.progress i{display:block; height:2px; width:0; background:var(--accent);
     transition:width .25s ease}
.topin{display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding:12px 0}
.topin h1{margin-right:auto}
.count{color:var(--muted); font-size:.875rem}
.count b{color:var(--text)}
.tbtn{background:var(--surface); color:var(--text); border:1px solid var(--line);
  border-radius:8px; padding:6px 12px; font:inherit; font-size:.875rem;
  cursor:pointer; line-height:1.4}
.tbtn:hover{background:var(--surface2)}
.tbtn.on{border-color:var(--accent); color:var(--accent); background:var(--accent-bg)}

.lede{color:var(--muted); font-size:.9375rem; margin:0 0 16px; max-width:70ch}
.safe{background:var(--legit-bg); border:1px solid var(--legit-line);
  border-radius:10px; padding:12px 14px; margin:0 0 18px; font-size:.9375rem;
  max-width:80ch}
.safe b{color:var(--legit)}
.keys{display:flex; gap:16px; flex-wrap:wrap; color:var(--muted);
  font-size:.8125rem; margin:0 0 22px}
kbd{background:var(--surface2); border:1px solid var(--line); border-bottom-width:2px;
  border-radius:5px; padding:1px 6px; font-size:.75rem; color:var(--text)}

/* Verdict basket */
.basket{background:var(--surface); border:1px solid var(--line);
  border-radius:12px; margin:0 0 28px; box-shadow:var(--shadow)}
.basket summary{cursor:pointer; padding:13px 16px; font-weight:500;
  display:flex; align-items:center; gap:10px; list-style:none}
.basket summary::-webkit-details-marker{display:none}
.basket summary::before{content:"▸"; color:var(--muted); font-size:.8em}
.basket[open] summary::before{content:"▾"}
.basket .in{padding:0 16px 16px}
.basket textarea{width:100%; height:130px; resize:vertical; background:var(--bg);
  color:var(--text); border:1px solid var(--line); border-radius:8px; padding:11px 12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8125rem;
  line-height:1.6}
.basket .row{display:flex; gap:9px; margin-top:10px; flex-wrap:wrap}
.hintline{color:var(--muted); font-size:.8125rem; margin:9px 0 0}

/* Card */
.card{background:var(--surface); border:1px solid var(--line); border-radius:14px;
  padding:20px; margin:0 0 22px; scroll-margin-top:76px; box-shadow:var(--shadow);
  border-left:3px solid transparent}
.card.cur{border-color:var(--accent)}
.card[data-verdict="fraud"]{border-left-color:var(--fraud)}
.card[data-verdict="legit"]{border-left-color:var(--legit)}
.card[data-verdict="unknown"]{border-left-color:var(--faint)}
body.hide-done .card[data-verdict], body.hide-done .card.done{display:none}
body.only-control .card:not([data-stratum="random_control"]){display:none}

.card header{display:flex; justify-content:space-between; align-items:flex-start;
  gap:16px; flex-wrap:wrap; margin-bottom:14px}
.yr{color:var(--muted); font-weight:400}
.meta{color:var(--muted); font-size:.8125rem; margin-top:5px}
.num{color:var(--faint)}
.dead{color:var(--warn)}
.live em{color:var(--faint); font-style:normal; font-size:.9em; margin-left:6px}
.flags{display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end}
.flag{background:var(--fraud-bg); border:1px solid var(--fraud-line); color:var(--fraud);
  border-radius:999px; padding:2px 10px; font-size:.75rem; white-space:nowrap;
  font-family:ui-monospace,Menlo,monospace}

.body{display:grid; gap:20px; grid-template-columns:1fr}
@media (min-width:1000px){ .body{grid-template-columns:minmax(0,1.5fr) minmax(290px,1fr)} }

/* Gallery: fixed height plus contain keeps portrait photos large enough to
   assess and prevents layout jumps while navigating. */
.hero{position:relative; background:var(--surface2); border:1px solid var(--line);
  border-radius:11px; overflow:hidden; cursor:zoom-in;
  height:clamp(320px, 54vh, 580px)}
.hero img{width:100%; height:100%; object-fit:contain; display:block}
.hero .zoom,.hero .counter{position:absolute; bottom:9px; font-size:.75rem;
  background:rgba(8,10,15,.72); color:#f0f2f7; padding:3px 9px; border-radius:6px;
  backdrop-filter:blur(4px)}
.hero .zoom{left:9px; opacity:0; transition:opacity .15s}
.hero:hover .zoom{opacity:1}
.hero .counter{right:9px}
.thumbs{display:flex; gap:7px; margin-top:9px; overflow-x:auto; padding-bottom:4px;
  scrollbar-width:thin}
.thumb{flex:0 0 auto; width:74px; height:56px; padding:0; border-radius:7px;
  border:1px solid var(--line); background:var(--surface2); overflow:hidden;
  cursor:pointer; opacity:.62; transition:opacity .15s, border-color .15s}
.thumb img{width:100%; height:100%; object-fit:cover; display:block}
.thumb:hover{opacity:.9}
.thumb.on{opacity:1; border-color:var(--accent); box-shadow:0 0 0 1px var(--accent)}
.empty{color:var(--muted); font-style:italic; padding:14px 0; font-size:.9375rem}

/* Facts */
.facts{margin:0; display:grid; gap:0}
.f{display:flex; justify-content:space-between; align-items:baseline; gap:12px;
  padding:7px 0; border-bottom:1px solid var(--line); font-size:.875rem}
.f:last-child{border-bottom:none}
.f dt{color:var(--muted); margin:0}
.f dd{margin:0; text-align:right; font-weight:500}
.f.big{padding:4px 0 10px; border-bottom:1px solid var(--line)}
.f.big dt{font-size:.875rem}
.f.big dd{font-size:1.35rem; font-weight:600; letter-spacing:-.01em}

.note{border-radius:9px; padding:10px 12px; margin-top:11px; font-size:.875rem;
  line-height:1.5}
.note b{display:block; margin-bottom:2px}
.note span{color:var(--muted); display:block}
.legit-note{background:var(--legit-bg); border:1px solid var(--legit-line)}
.legit-note b{color:var(--legit)}
.price-note{background:var(--accent-bg); border:1px solid var(--line)}
.stratum{border-radius:9px;padding:10px 13px;margin-bottom:12px;font-size:.9rem}
.stratum b{display:block;margin-bottom:3px}
.stratum span{color:var(--muted);display:block;line-height:1.45}
.s-flagged{background:var(--fraud-bg);border:1px solid var(--fraud-line)}
.s-flagged b{color:var(--fraud)}
.s-control{background:var(--accent-bg);border:1px solid var(--line)}
.s-control b{color:var(--accent)}
.done-note{background:var(--surface2); border:1px solid var(--line); margin:0 0 14px}

/* Listing text uses a limited measure and generous line height for review. */
.texts{margin-top:20px}
.txt+.txt{margin-top:14px}
.tl{color:var(--muted); font-size:.75rem; text-transform:uppercase;
  letter-spacing:.06em; font-weight:500; margin-bottom:5px}
.tb{background:var(--bg); border:1px solid var(--line); border-radius:9px;
  padding:12px 14px; white-space:pre-wrap; max-width:78ch; line-height:1.7}

/* Flag guidance */
.helps{margin-top:18px; border-top:1px solid var(--line); padding-top:12px}
.helps summary{cursor:pointer; color:var(--muted); font-size:.875rem;
  list-style:none}
.helps summary::-webkit-details-marker{display:none}
.helps summary::before{content:"▸ "; font-size:.8em}
.helps[open] summary::before{content:"▾ "}
.help{margin:14px 0 0; padding-left:13px; border-left:2px solid var(--line)}
.help code{font-size:.75rem; color:var(--muted)}
.help p{margin:4px 0 0; font-size:.875rem; line-height:1.55}
.hwhat{color:var(--text)}
.hfraud span,.hlegit span{font-family:ui-monospace,Menlo,monospace; font-size:.75rem;
  padding:1px 7px; border-radius:5px; margin-right:7px}
.hfraud span{background:var(--fraud-bg); color:var(--fraud)}
.hlegit span{background:var(--legit-bg); color:var(--legit)}

/* Verdict buttons */
.actions{display:flex; gap:9px; align-items:center; flex-wrap:wrap; margin-top:18px;
  border-top:1px solid var(--line); padding-top:16px}
.actions button{display:inline-flex; align-items:center; gap:7px; font:inherit;
  font-size:.9375rem; font-weight:500; padding:8px 15px; border-radius:9px;
  cursor:pointer; border:1px solid var(--line); background:var(--surface2);
  color:var(--text); transition:transform .08s}
.actions button:active{transform:translateY(1px)}
.actions button kbd{border-bottom-width:1px; opacity:.65}
.bfraud{background:var(--fraud-bg); border-color:var(--fraud-line); color:var(--fraud)}
.blegit{background:var(--legit-bg); border-color:var(--legit-line); color:var(--legit)}
.cmt{flex:1 1 240px; min-width:200px; background:var(--bg); color:var(--text);
  border:1px solid var(--line); border-radius:9px; padding:9px 12px; font:inherit;
  font-size:.9375rem}
.cmt:focus{outline:2px solid var(--accent); outline-offset:-1px; border-color:transparent}
.picked{font-size:.875rem; font-weight:500; color:var(--muted)}
.picked[data-state="local"]{color:var(--warn)}
.picked[data-state="saved"]{color:var(--legit)}
.picked[data-state="error"]{color:var(--fraud)}
.mode{font-size:.75rem; padding:3px 10px; border-radius:999px; white-space:nowrap}
.mode.live{background:var(--legit-bg); border:1px solid var(--legit-line);
  color:var(--legit)}
.mode.draft{background:var(--warn-bg); border:1px solid var(--line);
  color:var(--warn)}

/* Lightbox */
#box{position:fixed; inset:0; z-index:100; display:none; align-items:center;
  justify-content:center; background:rgba(6,8,12,.94); backdrop-filter:blur(3px)}
#box.open{display:flex}
#box img{max-width:93vw; max-height:88vh; object-fit:contain; border-radius:8px}
#box .bx{position:absolute; background:rgba(255,255,255,.1); color:#fff;
  border:none; border-radius:9px; width:46px; height:60px; font-size:1.4rem;
  cursor:pointer}
#box .bx:hover{background:rgba(255,255,255,.2)}
#box .prev{left:18px} #box .next{right:18px}
#box .cls{top:18px; right:18px; height:40px; width:40px; font-size:1.1rem}
#box .n{position:absolute; bottom:20px; color:#cfd4de; font-size:.875rem}
</style>

<div class="wrap">
<div class="top">
  <div class="progress"><i id="bar"></i></div>
  <div class="topin">
    <h1>Market anomaly review</h1>
    __HOME__
    <span class="count"><b id="cnt">0</b> labelled of <b id="scope-total">__N__</b> visible</span>
    <span class="count total-note" title="including previous queues and unknown verdicts">journal total: <b>__JOURNAL__</b></span>
    <span class="count" id="restored"></span>
    <span class="mode __MODECLS__">__MODE__</span>
    <button class="tbtn" id="filter">hide labelled</button>
    <button class="tbtn" id="only-control">controls only</button>
    <button class="tbtn" id="theme">theme</button>
  </div>
</div>

<p class="lede"><b>__N__ listings</b>: __NRULES__ were flagged by rules,
__NRESIDUAL__ came from the residual detector, and __NCONTROL__ were sampled
randomly to measure misses. __NDONE__ already have final fraud/legit verdicts;
__NLEFT__ still need a decision. Before manual review these are candidates, not
accusations against sellers. __NDEAD__ have closed Kolesa pages and __NOPHOTO__
have no available photos because their historical image host was retired.</p>

<div class="safe"><b>This page does not request kolesa.kz.</b> Photos load from
a separate CDN and all text and facts come from the local database, so normal
labelling does not consume the daily Kolesa request budget. The only exception
is the explicit “open on Kolesa” link on a card.</div>

<div class="keys">
  <span><kbd>J</kbd><kbd>K</kbd> next / previous listing</span>
  <span><kbd>F</kbd> fraud &nbsp;<kbd>L</kbd> legit &nbsp;<kbd>U</kbd> unknown</span>
  <span><kbd>←</kbd><kbd>→</kbd> previous / next photo</span>
  <span><kbd>C</kbd> focus comment</span>
</div>

<details class="basket" open>
  <summary>Labelled in this session — <span id="cnt2">0</span></summary>
  <div class="in">
    <textarea id="out" readonly placeholder="Choose fraud, legit, or unknown on a card."></textarea>
    <div class="row">
      <button class="tbtn" id="copy">copy all</button>
      <button class="tbtn" id="clear">clear browser draft</button>
    </div>
    <p class="hintline" id="baskethint"></p>
  </div>
</details>

__CARDS__
</div>

<div id="box">
  <button class="bx cls" title="Esc">✕</button>
  <button class="bx prev" title="←">‹</button>
  <img id="boximg" alt="">
  <button class="bx next" title="→">›</button>
  <span class="n" id="boxn"></span>
</div>

<script>
const SERVER = __SERVER__;   /* true on /label, where journal writes are allowed */
const cards = Array.from(document.querySelectorAll('.card'));
/* Listings that already have a journal verdict. */
const ALREADY = new Set(cards.filter(c => c.querySelector('.done-note'))
                             .map(c => c.dataset.id));
const picks = new Map();
let cur = 0;

function visibleCards(){
  return cards.filter(card => getComputedStyle(card).display !== 'none');
}

/* Escape quotes and newlines according to CSV rules so punctuation in a
   comment cannot shift journal columns. */
function esc(s){ return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }

function render(){
  const lines = [];
  /* Journal schema: ad_id, seven descriptive fields, verdict, comment, then
     sampling metadata. The comma count must match the header. */
  for (const [id, v] of picks) lines.push(id + ',,,,,,,,' + v + ',,');
  document.getElementById('out').value = lines.join('\n');
  /* Show the total of existing journal rows and new selections. Counting only
     the browser draft made completed work appear lost on another origin. */
  const scope = visibleCards();
  const completed = scope.filter(card =>
    ALREADY.has(card.dataset.id) || picks.has(card.dataset.id)).length;
  document.getElementById('cnt').textContent = completed;
  document.getElementById('scope-total').textContent = scope.length;
  document.getElementById('cnt2').textContent = picks.size;
  document.getElementById('bar').style.width =
    (scope.length ? completed / scope.length * 100 : 0) + '%';
}

/* A selection has three representations with different purposes: the current
   session basket, localStorage for reload recovery, and the on-disk journal as
   the sole source of truth. */
const STORE = 'label_cards_picks';

function saveLocal(){
  const obj = {};
  for (const [id, v] of picks) obj[id] = v;
  try { localStorage.setItem(STORE, JSON.stringify(obj)); } catch (e) {}
}

function loadLocal(){
  try { return JSON.parse(localStorage.getItem(STORE) || '{}'); }
  catch (e) { return {}; }
}

function mark(card, state, text){
  const el = card.querySelector('.picked');
  el.dataset.state = state;
  el.textContent = text;
}

function setVerdict(card, v){
  if (!card) return;
  const cmt = card.querySelector('.cmt').value.trim();
  picks.set(card.dataset.id, v + ',' + esc(cmt));
  card.dataset.verdict = v;
  mark(card, 'local', '→ ' + v);
  saveLocal();
  render();
  if (!SERVER) return;                    /* file:// supports copy/paste only */
  fetch('/verdict', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ad_id: card.dataset.id, verdict: v, comment: cmt}),
  }).then(r => r.json())
    .then(d => mark(card, d.ok ? 'saved' : 'error',
                    d.ok ? '✓ saved to journal' : '✗ ' + (d.error || 'Error')))
    /* A network failure leaves the selection recoverable in localStorage. */
    .catch(() => mark(card, 'error', '✗ not saved; retained in browser'));
}

function focusCard(i){
  if (!cards.length) return;
  cur = Math.max(0, Math.min(cards.length - 1, i));
  cards.forEach((c, j) => c.classList.toggle('cur', j === cur));
  cards[cur].scrollIntoView({block: 'start', behavior: 'smooth'});
}

function focusFirstVisible(){
  const i = cards.findIndex(card => getComputedStyle(card).display !== 'none');
  if (i >= 0) focusCard(i);
}

function stepVisible(direction){
  for (let i = cur + direction; i >= 0 && i < cards.length; i += direction){
    if (getComputedStyle(cards[i]).display !== 'none'){
      focusCard(i);
      return;
    }
  }
}

/* Galleries */
cards.forEach(card => {
  if (card.querySelector('.done-note')) card.classList.add('done');
  card.querySelectorAll('.actions button').forEach(b => {
    b.onclick = () => { focusCard(+card.dataset.idx); setVerdict(card, b.dataset.v); };
  });
  const gal = card.querySelector('.gal');
  if (!gal || !gal.dataset.photos) return;
  const photos = JSON.parse(gal.dataset.photos);
  const hero = gal.querySelector('.hero img');
  const counter = gal.querySelector('.counter b');
  let i = 0;
  const show = n => {
    i = (n + photos.length) % photos.length;
    hero.src = photos[i];
    counter.textContent = i + 1;
    gal.querySelectorAll('.thumb').forEach((t, j) => t.classList.toggle('on', j === i));
  };
  gal.querySelectorAll('.thumb').forEach(t => t.onclick = () => show(+t.dataset.i));
  gal.querySelector('.hero').onclick = () => openBox(photos, i);
  card._show = show;
  card._at = () => i;
});

/* Lightbox */
const box = document.getElementById('box');
let bp = [], bi = 0;
function paint(){
  document.getElementById('boximg').src = bp[bi];
  document.getElementById('boxn').textContent = (bi + 1) + ' / ' + bp.length;
}
function openBox(p, i){ bp = p; bi = i; box.classList.add('open'); paint(); }
function moveBox(d){ bi = (bi + d + bp.length) % bp.length; paint(); }
function closeBox(){ box.classList.remove('open'); }
box.querySelector('.cls').onclick = closeBox;
box.querySelector('.prev').onclick = e => { e.stopPropagation(); moveBox(-1); };
box.querySelector('.next').onclick = e => { e.stopPropagation(); moveBox(1); };
box.onclick = e => { if (e.target === box) closeBox(); };

/* Keyboard */
document.addEventListener('keydown', e => {
  if (box.classList.contains('open')){
    if (e.key === 'Escape') closeBox();
    else if (e.key === 'ArrowRight') moveBox(1);
    else if (e.key === 'ArrowLeft') moveBox(-1);
    return;
  }
  if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName)){
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const card = cards[cur] && getComputedStyle(cards[cur]).display !== 'none'
    ? cards[cur] : null;
  const k = e.key.toLowerCase();
  if (k === 'j') stepVisible(1);
  else if (k === 'k') stepVisible(-1);
  else if (k === 'f') setVerdict(card, 'fraud');
  else if (k === 'l') setVerdict(card, 'legit');
  else if (k === 'u') setVerdict(card, 'unknown');
  else if (k === 'c'){ card && card.querySelector('.cmt').focus(); e.preventDefault(); }
  else if (e.key === 'ArrowRight'){ card && card._show && card._show(card._at() + 1); }
  else if (e.key === 'ArrowLeft'){ card && card._show && card._show(card._at() - 1); }
  else return;
  if (k === 'j' || k === 'k') e.preventDefault();
});

/* Controls */
document.getElementById('copy').onclick = () => {
  const t = document.getElementById('out');
  if (!t.value) return;
  const text = t.value + '\n';
  /* The Clipboard API may be unavailable on file://; use the legacy fallback. */
  const fallback = () => {
    t.removeAttribute('readonly'); t.select();
    document.execCommand('copy');
    t.setAttribute('readonly', ''); window.getSelection().removeAllRanges();
  };
  if (navigator.clipboard) navigator.clipboard.writeText(text).catch(fallback);
  else fallback();
  const b = document.getElementById('copy');
  b.textContent = 'copied';
  setTimeout(() => b.textContent = 'copy all', 1400);
};
document.getElementById('clear').onclick = () => {
  /* Clear only the browser draft. Never remove existing journal verdicts. */
  if (!confirm('Clear the browser draft? Verdicts already saved to the journal '
             + 'will remain unchanged.')) return;
  picks.clear();
  try { localStorage.removeItem(STORE); } catch (e) {}
  cards.forEach(c => {
    delete c.dataset.verdict;
    c.querySelector('.picked').textContent = '';
  });
  render();
};
document.getElementById('filter').onclick = e => {
  document.body.classList.toggle('hide-done');
  const on = document.body.classList.contains('hide-done');
  e.target.classList.toggle('on', on);
  e.target.textContent = on ? 'show all' : 'hide labelled';
  focusFirstVisible();
  render();
};
document.getElementById('only-control').onclick = e => {
  /* Controls were not flagged. They are required to estimate how many fraud
     cases the detector missed, so this filter makes them directly accessible. */
  document.body.classList.toggle('only-control');
  const on = document.body.classList.contains('only-control');
  e.target.classList.toggle('on', on);
  e.target.textContent = on ? 'show all' : 'controls only';
  focusFirstVisible();
  render();
};
document.getElementById('theme').onclick = () => {
  const root = document.documentElement;
  const now = root.dataset.theme ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  root.dataset.theme = now === 'dark' ? 'light' : 'dark';
};

/* Restore the previous browser draft, including comments. */
(() => {
  const saved = loadLocal();
  let n = 0;
  for (const card of cards){
    const v = saved[card.dataset.id];
    if (!v) continue;
    picks.set(card.dataset.id, v);
    const i = v.indexOf(',');
    const verdict = i < 0 ? v : v.slice(0, i);
    let cmt = i < 0 ? '' : v.slice(i + 1);
    if (cmt.startsWith('"') && cmt.endsWith('"'))
      cmt = cmt.slice(1, -1).replace(/""/g, '"');
    card.dataset.verdict = verdict;
    card.querySelector('.cmt').value = cmt;
    mark(card, 'local', '→ ' + verdict);
    n++;
  }
  if (n) document.getElementById('restored').textContent =
    'restored from browser: ' + n;
})();

/* Explain persistence differently for live server mode and offline export. */
const _hint = document.getElementById('baskethint');
if (_hint) _hint.innerHTML = SERVER
  ? 'Verdicts are already stored in <b>__LABELS__</b>; no copying is required. '
    + 'This list only summarizes the current session. Next: '
    + '<span class="mono">python -m kz.ops.run_all --ml</span>.'
  : 'A file:// page cannot write to the journal. Copy these rows into '
    + '<b>__LABELS__</b>, or preferably use '
    + '<span class="mono">python -m kz.web</span>.';

focusCard(0);
render();
</script>
"""
