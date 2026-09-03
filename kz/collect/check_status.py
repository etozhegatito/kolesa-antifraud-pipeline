"""Implementation for the `kz.collect.check_status` module."""

import pathlib as _p

_expected = "check_status.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import csv
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path

import requests

from kz.core import pacing
from kz.core.db import upsert

RAW_CSV = "data/raw/raw_data.csv"
SIGHTINGS_CSV = "data/raw/sightings.csv"
STATUS_CSV = "data/raw/ad_status.csv"
LOG_FILE = "logs/status.log"

STALE_DAYS = 2


RECHECK_DAYS = 7


MAX_CHECKS_PER_RUN = 20
DELAY_RANGE = (4.0, 8.0)
MAX_CONSECUTIVE_FAILS = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


ARCHIVE_MARKERS = ["в&nbsp;архиве", "в\u00a0архиве", "в архиве", ">В архиве<"]

STATUS_FIELDS = ["ad_id", "status", "checked_at"]


def infer_active_from_listing(cur_status, seen_days, seen_after_check) -> bool:
    """Implement `infer_active_from_listing`."""
    if seen_days is None or seen_days >= STALE_DAYS:
        return False
    if cur_status is None:
        return True
    if cur_status in ("archived", "deleted") and seen_after_check:
        return True
    return False


def needs_status_check(cur_status, seen_days, checked_days) -> bool:
    """Implement `needs_status_check`."""
    if cur_status in ("archived", "deleted"):
        return False
    if seen_days is not None and seen_days < STALE_DAYS:
        return False
    if checked_days is not None and checked_days < RECHECK_DAYS:
        return False
    return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_last_seen() -> dict:
    """Implement `load_last_seen`."""
    last = {}
    if not Path(SIGHTINGS_CSV).exists():
        return last
    with open(SIGHTINGS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["seen_date"]
            if r["ad_id"] not in last or d > last[r["ad_id"]]:
                last[r["ad_id"]] = d
    return last


def load_status_rows() -> dict:
    """Implement `load_status_rows`."""
    rows = {}
    if not Path(STATUS_CSV).exists():
        with open(STATUS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=STATUS_FIELDS).writeheader()
        return rows
    with open(STATUS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            checked = (r.get("checked_at") or "")[:10]
            try:
                d = date.fromisoformat(checked) if checked else None
            except ValueError:
                d = None
            rows[r["ad_id"]] = (r["status"], d)
    return rows


def append_status(ad_id: str, status: str):
    checked_at = datetime.now().isoformat(timespec="seconds")
    with open(STATUS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=STATUS_FIELDS).writerow(
            {
                "ad_id": ad_id,
                "status": status,
                "checked_at": checked_at,
            }
        )

    try:
        upsert(
            "ad_status",
            [{"ad_id": ad_id, "status": status, "checked_at": checked_at}],
            ["ad_id"],
            update_cols=["status", "checked_at"],
        )
    except Exception as e:
        log.warning(f"PostgreSQL upsert failed for {ad_id}: {e}")


def check_ad(ad_id: str, session: requests.Session) -> str | None:
    """Implement `check_ad`."""
    url = f"https://kolesa.kz/a/show/{ad_id}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        log.warning(f"{ad_id}: network error: {e}")
        return None
    if resp.status_code == 404:
        return "deleted"
    if resp.status_code == 200:
        archived = any(m in resp.text for m in ARCHIVE_MARKERS)
        return "archived" if archived else "active"
    if resp.status_code == 429:
        log.warning(f"{ad_id}: HTTP 429; pausing for 120 seconds")
        time.sleep(120)
        return None
    log.warning(f"{ad_id}: unexpected HTTP status {resp.status_code}")
    return None


def main():
    last_seen = load_last_seen()
    rows = load_status_rows()
    today = date.today()

    marked = 0
    for ad_id, seen in last_seen.items():
        seen_d = date.fromisoformat(seen)
        seen_days = (today - seen_d).days
        cur_status, cur_checked = rows.get(ad_id, (None, None))
        seen_after_check = cur_checked is None or seen_d > cur_checked
        if infer_active_from_listing(cur_status, seen_days, seen_after_check):
            append_status(ad_id, "active")
            marked += 1
    if marked:
        log.info(f"Marked active from listing pages without extra requests: {marked}")

    candidates = []
    for ad_id, seen in last_seen.items():
        cur_status, cur_checked = rows.get(ad_id, (None, None))
        seen_days = (today - date.fromisoformat(seen)).days
        checked_days = None if cur_checked is None else (today - cur_checked).days
        if needs_status_check(cur_status, seen_days, checked_days):
            candidates.append((seen_days, ad_id))
    candidates.sort(reverse=True)
    batch = [ad for _, ad in candidates[:MAX_CHECKS_PER_RUN]]
    log.info(f"Candidates: {len(candidates)}; checking: {len(batch)}")

    session = requests.Session()
    fails = 0
    counts = {"active": 0, "archived": 0, "deleted": 0}
    for i, ad_id in enumerate(batch, 1):
        status = check_ad(ad_id, session)
        if status is None:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Stopped after consecutive failures; resume later.")
                sys.exit(1)
            time.sleep(30)
            continue
        fails = 0
        append_status(ad_id, status)
        counts[status] += 1
        if i % 25 == 0:
            log.info(f"  {i}/{len(batch)}: {counts}")
        pacing.polite_sleep(i, DELAY_RANGE, log)

    log.info(f"Completed: {counts} → {STATUS_CSV}")


if __name__ == "__main__":
    main()
