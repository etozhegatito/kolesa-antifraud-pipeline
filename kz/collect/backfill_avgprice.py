# -*- coding: utf-8 -*-
"""Implementation for the `kz.collect.backfill_avgprice` module."""

import pathlib as _p

_expected = "backfill_avgprice.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named {_p.Path(__file__).name}."
    )

import csv
import logging
import sys
import time

import pandas as pd
import requests
from sqlalchemy import text

from kz.core import pacing
from kz.core.db import get_engine
from kz.collect.enrich import HEADERS, extract_avg_price, extract_status_badge, ENRICHED_CSV

MAX_PER_RUN = 20
DELAY_RANGE = (4.0, 8.0)
MAX_CONSECUTIVE_FAILS = 3
LOG_FILE = "logs/enrich.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def pick_targets() -> list[str]:
    """Implement `pick_targets`."""
    engine = get_engine()
    todo = pd.read_sql(
        "SELECT ad_id FROM enriched WHERE http_status = 200 "
        "AND (kolesa_avg_price IS NULL OR page_status_badge IS NULL)",
        engine,
        dtype={"ad_id": str},
    )
    susp = pd.read_sql("SELECT ad_id, is_suspicious FROM clean_data", engine, dtype={"ad_id": str})
    todo = todo.merge(susp, on="ad_id", how="left")
    todo["is_suspicious"] = todo["is_suspicious"].fillna(0)
    todo = todo.sort_values("is_suspicious", ascending=False)
    ids = todo["ad_id"].tolist()
    return ids if "--all" in sys.argv else ids[:MAX_PER_RUN]


def update_stores(ad_id: str, avg, badge: str):
    """Implement `update_stores`."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE enriched SET "
                "kolesa_avg_price = COALESCE(kolesa_avg_price, :a), "
                "page_status_badge = COALESCE(NULLIF(page_status_badge, ''), :b) "
                "WHERE ad_id = :id"
            ),
            {"a": avg, "b": badge, "id": ad_id},
        )

    rows = list(csv.DictReader(open(ENRICHED_CSV, encoding="utf-8")))
    fields = list(rows[0].keys()) if rows else []
    for r in rows:
        if r["ad_id"] == ad_id:
            if not str(r.get("kolesa_avg_price", "")).strip():
                r["kolesa_avg_price"] = str(avg)
            if not str(r.get("page_status_badge", "")).strip():
                r["page_status_badge"] = badge
    with open(ENRICHED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    targets = pick_targets()
    log.info(f"avgPrice + badge backfill: {len(targets)} rows pending")
    session = requests.Session()
    avg_filled = badge_filled = 0
    fails = 0
    for i, ad_id in enumerate(targets, 1):
        try:
            resp = session.get(f"https://kolesa.kz/a/show/{ad_id}", headers=HEADERS, timeout=20)
        except requests.RequestException as e:
            log.warning(f"{ad_id}: {e}")
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Stopped after consecutive failures; resume later.")
                sys.exit(1)
            time.sleep(30)
            continue

        if resp.status_code == 429:
            log.warning("HTTP 429: pausing for 120 seconds")
            time.sleep(120)
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Stopped after consecutive 429 responses; the site is rate-limiting us.")
                sys.exit(1)
            continue
        fails = 0

        if resp.status_code == 200:
            avg = extract_avg_price(resp.text)
            badge = extract_status_badge(resp.text)
        else:
            avg, badge = None, "-"

        update_stores(ad_id, avg if avg is not None else -1, badge)
        if avg:
            avg_filled += 1
        if badge and badge != "-":
            badge_filled += 1
        if i % 20 == 0:
            log.info(f"  {i}/{len(targets)} (avgPrice: {avg_filled}, badges: {badge_filled})")
        pacing.polite_sleep(i, DELAY_RANGE, log)
    log.info(
        f"Completed {len(targets)} rows: filled {avg_filled} avgPrice values and "
        f"found {badge_filled} status badges"
    )


if __name__ == "__main__":
    main()
