# -*- coding: utf-8 -*-
"""Implementation for the `kz.ops.pipeline_status` module."""

import pathlib as _p

_expected = "pipeline_status.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files were mixed up during copying."
    )


import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from kz.core.db import get_engine


from kz.ops.catch_up import STATUS_STALE_DAYS, STATUS_RECHECK_DAYS, DAILY_BUDGET


ENRICH_PER_DAY = DAILY_BUDGET["kolesa"]
PHOTOS_PER_DAY = DAILY_BUDGET["cdn"]

LABELS_CSV = "data/manual_labels.csv"
PARSER_STATUS_JSON = "logs/parser_last_run.json"

LINE = "─" * 64


def eta_days(pending: int, per_day: int) -> str:
    if pending <= 0:
        return "complete"
    return f"~{math.ceil(pending / per_day)} days at {per_day}/day"


def bar(done: int, total: int, width: int = 24) -> str:
    """Implement `bar`."""
    if total <= 0:
        return "─" * width + "   0%"
    frac = done / total
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled) + f" {frac:5.1%}"


def main():
    from kz.core import freshness as fr

    _state = fr.measure()
    fr.report(_state)
    for _w in fr.stale_warnings(_state):
        print(f"  \u26a0 {_w}")
    print()
    engine = get_engine()

    with engine.begin() as conn:
        has_clean = conn.execute(text("SELECT to_regclass('public.clean_data')")).scalar()
    if not has_clean:
        raise SystemExit("clean_data does not exist; run clean.py first (or run_all.py --fast).")

    clean = pd.read_sql(
        "SELECT ad_id, is_suspicious, seller_comment, description, status FROM clean_data",
        engine,
        dtype={"ad_id": str},
    )
    enriched = pd.read_sql("SELECT ad_id, http_status FROM enriched", engine, dtype={"ad_id": str})
    photos = pd.read_sql("SELECT ad_id, url FROM photos", engine, dtype={"ad_id": str})
    hashes = pd.read_sql("SELECT ad_id, url, phash FROM photo_hashes", engine, dtype={"ad_id": str})

    total = len(clean)
    clean_ids = set(clean["ad_id"])
    matched_enriched = enriched[enriched["ad_id"].isin(clean_ids)].copy()
    matched_ids = set(matched_enriched["ad_id"])
    usable_enriched = matched_enriched[matched_enriched["http_status"] == 200]
    usable_ids = set(usable_enriched["ad_id"])
    dead_enriched = matched_enriched[matched_enriched["http_status"] != 200]
    orphan_enriched = enriched[~enriched["ad_id"].isin(clean_ids)]

    pending_mask = ~clean["ad_id"].isin(matched_ids)
    pending = int(pending_mask.sum())
    pending_susp = int((pending_mask & (clean["is_suspicious"] == 1)).sum())
    enr_ok = len(usable_ids)
    enr_fail = len(dead_enriched)

    print(LINE)
    print(f"TOTAL LISTINGS: {total}   suspicious: {int(clean['is_suspicious'].sum())}")
    print(LINE)

    print("\n► Detail-page enrichment (enrich.py)")
    print(f"  useful: {bar(enr_ok, total)}   {enr_ok}/{total}")
    print(f"  rows in enriched: {len(enriched)}; matched raw/clean: {len(matched_enriched)}")
    print(f"  remaining: {pending}  → {eta_days(pending, ENRICH_PER_DAY)}")
    if pending_susp:
        print(f"  ⚠ suspicious listings waiting: {pending_susp} (they will be prioritized next)")
    else:
        print("  ✓ all suspicious listings are enriched")
    if enr_fail:
        print(f"  pages unavailable before enrichment (404/archive/etc.): {enr_fail}")
    if len(orphan_enriched):
        print(
            f"  ⚠ orphan rows without a raw/clean listing: {len(orphan_enriched)} "
            "(excluded from coverage)"
        )

    print("\n► Latest parser.py run")
    try:
        parser_run = json.loads(Path(PARSER_STATUS_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("  no structured status yet; it will appear after the next run")
    else:
        run_status = parser_run.get("status", "unknown")
        print(
            f"  status: {run_status}; "
            f"started: {parser_run.get('started_at', '?')}; "
            f"finished: {parser_run.get('finished_at') or 'still running'}"
        )
        truncated = parser_run.get("freshness_truncated_segments") or []
        if truncated:
            print(
                "  ⚠ page limit was reached while unseen listings continued: "
                + ", ".join(truncated)
            )
        elif run_status == "success":
            print("  ✓ no unseen listings were observed at the page-limit boundary")
        elif parser_run.get("message"):
            print(f"  ⚠ run did not complete successfully: {parser_run['message']}")

    hashable = photos[photos["url"].fillna("").str.startswith("http")]
    hashed_urls = set(hashes["url"])
    ph_pending = int((~hashable["url"].isin(hashed_urls)).sum())
    ph_done = len(hashable) - ph_pending
    ph_bad = int((hashes["phash"].isna() | (hashes["phash"] == "")).sum())

    print("\n► Photo hashes (photo_dedup.py)")
    print(
        f"  {bar(ph_done, len(hashable))}   {ph_done}/{len(hashable)}"
        f"   (+{len(photos) - len(hashable)} no-photo placeholders excluded)"
    )
    print(f"  remaining: {ph_pending}  → {eta_days(ph_pending, PHOTOS_PER_DAY)}")
    if ph_bad:
        print(f"  downloaded but unreadable (corrupt/timeouts): {ph_bad}")

    sc = clean["seller_comment"].fillna("").astype(str).str.len() > 0
    desc = clean["description"].fillna("").astype(str).str.len() > 0
    print("\n► Text (text_full in clean_data)")
    print(f"  full seller comment : {int(sc.sum())}")
    print(f"  listing snippet only: {int((~sc & desc).sum())}")
    print(f"  no text             : {int((~sc & ~desc).sum())}")

    st = clean["status"].fillna("active").value_counts()
    print("\n► Status checks (check_status.py)")
    for name, cnt in st.items():
        print(f"  {name:<10} {cnt}")

    last_seen = pd.read_sql(
        "SELECT ad_id, MAX(seen_date) AS seen FROM sightings GROUP BY ad_id",
        engine,
        dtype={"ad_id": str},
    )
    statuses = pd.read_sql(
        "SELECT ad_id, status, checked_at FROM ad_status", engine, dtype={"ad_id": str}
    )
    last_seen = last_seen.merge(statuses, on="ad_id", how="left")
    today = pd.Timestamp.today().normalize()
    days_gone = (today - pd.to_datetime(last_seen["seen"])).dt.days
    checked_days = (today - pd.to_datetime(last_seen["checked_at"])).dt.days
    terminal = last_seen["status"].isin(["archived", "deleted"])
    recently_checked = checked_days < STATUS_RECHECK_DAYS  # NaN<7 → False
    st_pending = int(((~terminal) & (days_gone >= STATUS_STALE_DAYS) & (~recently_checked)).sum())
    print(
        f"  waiting for status check: {st_pending}  → "
        f"{eta_days(st_pending, DAILY_BUDGET['kolesa'])} (Kolesa 24h ceiling)"
    )

    print("\n► Manual labels (data/manual_labels.csv)")
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        verdict = lab["verdict"].astype("string").str.strip().str.lower()
        latest = lab.assign(_verdict=verdict).drop_duplicates("ad_id", keep="last")
        n_valid = int(latest["_verdict"].isin(["fraud", "legit"]).sum())
        n_fraud = int((latest["_verdict"] == "fraud").sum())
        print(f"  valid verdicts: {n_valid} (fraud: {n_fraud}, legit: {n_valid - n_fraud})")
        print(
            f"  journal rows: {len(lab)}; "
            f"empty/unknown: {int((~verdict.isin(['fraud', 'legit'])).sum())}"
        )
    else:
        print("  file does not exist yet — 0 verdicts (queue: data/eda/labeling_queue.csv)")
    print(LINE)

    maybe_run_enrichment(pending, ph_pending)


from kz.ops import run_all as _ra


def run_enrichment_jobs():
    t0 = time.time()
    _ra.run_step(_ra.STEP_STATUS)
    _ra.run_parallel(_ra.STEP_ENRICH, _ra.STEP_PHOTOS)
    _ra.run_step(_ra.STEP_CLEAN)
    _ra.run_step(_ra.STEP_EXPLORE)
    print(f"\n✔ Enrichment completed in {(time.time() - t0) / 60:.1f} min")


def maybe_run_enrichment(pending: int, ph_pending: int):
    if pending <= 0 and ph_pending <= 0:
        print("\nThe backlog is empty; there is nothing to enrich.")
        return

    if "--run" in sys.argv:
        run_enrichment_jobs()
        return

    if not sys.stdin.isatty():
        print("\nRun enrichment: python -m kz.ops.pipeline_status --run")
        return

    print(f"\nQueued: {pending} pages (batch ~120) and {ph_pending} photos (batch ~300).")
    print("These are network requests to kolesa.kz; do not run repeated batches.")
    print("Jobs are resumable and can safely continue after budget capacity returns.")
    ans = input("Run enrichment jobs now? [y/N] ").strip().lower()
    if ans in ("y", "yes"):
        run_enrichment_jobs()
    else:
        print("Not started. Run later with: python -m kz.ops.pipeline_status --run")


if __name__ == "__main__":
    main()
