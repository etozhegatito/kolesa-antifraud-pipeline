# -*- coding: utf-8 -*-
"""Implementation for the `kz.collect.photo_dedup` module."""

import pathlib as _p

_expected = "photo_dedup.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import csv
import io
import logging
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image
from sqlalchemy import text

from kz.core.db import get_engine, upsert

HASHES_CSV = "data/enriched/photo_hashes.csv"
DUPLICATES_CSV = "data/enriched/photo_duplicates.csv"
LOG_FILE = "logs/photo_dedup.log"

MAX_PER_RUN = 300
DELAY_RANGE = (1.5, 3.0)
MAX_CONSECUTIVE_FAILS = 5
MAX_IMAGE_BYTES = 5_000_000


HAMMING_THRESHOLD = 0
PRICE_DIFF_RATIO = 0.15
YEAR_DIFF_MIN = 2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

HASH_FIELDS = ["ad_id", "position", "url", "phash", "fetched_at", "http_status"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_hashes() -> pd.DataFrame:
    """Implement `load_hashes`."""
    if not Path(HASHES_CSV).exists():
        with open(HASHES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HASH_FIELDS).writeheader()
        return pd.DataFrame(columns=HASH_FIELDS)
    return pd.read_csv(HASHES_CSV, dtype={"ad_id": str})


def live_hosts(urls) -> set[str]:
    """Implement `live_hosts`."""
    hosts = {str(u).split("/")[2] for u in urls if str(u).startswith("http")}
    alive = set()
    for host in sorted(hosts):
        try:
            socket.getaddrinfo(host, 443)
            alive.add(host)
        except OSError:
            log.warning(f"Host {host} does not resolve; skipping its URLs")
    return alive


def append_hash(row: dict):
    with open(HASHES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HASH_FIELDS, extrasaction="ignore").writerow(row)

    try:
        pg_row = {k: (v if v != "" else None) for k, v in row.items()}
        upsert("photo_hashes", [pg_row], ["ad_id", "position"])
    except Exception as e:
        log.warning(f"PostgreSQL upsert failed for {row.get('url')}: {e}")


def pick_targets(done_urls: set) -> pd.DataFrame:
    """Implement `pick_targets`."""
    engine = get_engine()
    photos = pd.read_sql("SELECT ad_id, position, url FROM photos", engine, dtype={"ad_id": str})
    photos = photos[~photos["url"].isin(done_urls)]

    photos = photos[photos["url"].fillna("").str.startswith("http")]

    alive = live_hosts(photos["url"])
    photos = photos[photos["url"].str.split("/").str[2].isin(alive)]

    clean = pd.read_sql("SELECT ad_id, is_suspicious FROM clean_data", engine, dtype={"ad_id": str})
    photos = photos.merge(clean, on="ad_id", how="left")
    photos["is_suspicious"] = photos["is_suspicious"].fillna(0)

    photos["_kolesa_host"] = photos["url"].str.contains(r"//[^/]*kolesa\.kz/")

    photos = photos.sort_values(
        ["is_suspicious", "_kolesa_host", "position"], ascending=[False, True, True]
    )
    return photos.head(MAX_PER_RUN)


def fetch_phash(url: str, session: requests.Session) -> tuple[str | None, int]:
    """Implement `fetch_phash`."""
    resp = session.get(url, headers=HEADERS, timeout=15, stream=True)
    if resp.status_code != 200:
        return None, resp.status_code
    content = resp.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    if len(content) > MAX_IMAGE_BYTES:
        log.warning(f"{url}: file exceeds {MAX_IMAGE_BYTES} bytes; skipping")
        return None, resp.status_code
    img = Image.open(io.BytesIO(content))
    return str(imagehash.phash(img)), resp.status_code


def collect_hashes():
    done = load_hashes()
    done_urls = set(done["url"])
    targets = pick_targets(done_urls)
    log.info(f"Queued for hashing: {len(targets)} (already complete: {len(done_urls)})")

    session = requests.Session()
    fails = 0
    for i, row in enumerate(targets.itertuples(), 1):
        try:
            phash, http_status = fetch_phash(row.url, session)
        except requests.RequestException as e:
            log.warning(f"{row.url}: network error: {e}")
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Stopped after consecutive failures; resume later.")
                sys.exit(1)
            time.sleep(30)
            continue
        except Exception as e:
            log.warning(f"{row.url}: could not decode image: {e}")
            phash, http_status = None, -1

        if http_status == 429:
            log.warning("HTTP 429: pausing for 120 seconds")
            time.sleep(120)
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Stopped after consecutive 429 responses; the CDN is rate-limiting us.")
                sys.exit(1)
            continue

        fails = 0
        append_hash(
            {
                "ad_id": row.ad_id,
                "position": row.position,
                "url": row.url,
                "phash": phash or "",
                "http_status": http_status,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if i % 50 == 0:
            log.info(f"  {i}/{len(targets)}")
        time.sleep(random.uniform(*DELAY_RANGE))

    log.info(f"Completed → {HASHES_CSV}")


def find_cross_car_duplicates(hashes: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Implement `find_cross_car_duplicates`."""
    valid = hashes[hashes["phash"].notna() & (hashes["phash"] != "")].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "ad_id_a",
                "ad_id_b",
                "hamming_distance",
                "model_key_a",
                "price_a",
                "year_a",
                "model_key_b",
                "price_b",
                "year_b",
            ]
        )

    keep = ["brand", "model", "year", "price_tenge"]
    for extra in ("condition", "labels"):
        if extra in clean.columns:
            keep.append(extra)
    cars = clean.set_index("ad_id")[keep]
    cars["model_key"] = (cars["brand"].fillna("") + " " + cars["model"].fillna("")).str.strip()

    _cond = cars["condition"] if "condition" in cars.columns else pd.Series("", index=cars.index)
    _lab = cars["labels"] if "labels" in cars.columns else pd.Series("", index=cars.index)
    cars["dealer_new"] = _cond.eq("новый").fillna(False) | _lab.fillna("").str.contains(
        "дилер|Новая", case=False
    )

    def both_dealer_new(a: str, b: str) -> bool:
        if a not in cars.index or b not in cars.index:
            return False
        return bool(cars.loc[a, "dealer_new"]) and bool(cars.loc[b, "dealer_new"])

    def different_cars(a: str, b: str) -> bool:
        if a not in cars.index or b not in cars.index:
            return False
        ca, cb = cars.loc[a], cars.loc[b]
        if ca["model_key"] != cb["model_key"]:
            return True
        if (
            pd.notna(ca["year"])
            and pd.notna(cb["year"])
            and abs(ca["year"] - cb["year"]) >= YEAR_DIFF_MIN
        ):
            return True
        if pd.notna(ca["price_tenge"]) and pd.notna(cb["price_tenge"]):
            hi = max(ca["price_tenge"], cb["price_tenge"])
            if hi > 0 and abs(ca["price_tenge"] - cb["price_tenge"]) / hi > PRICE_DIFF_RATIO:
                return True
        return False

    pairs = {}

    def consider(a: str, b: str, dist: int):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if key not in pairs or dist < pairs[key]:
            pairs[key] = dist

    for _, ad_ids in valid.groupby("phash")["ad_id"]:
        uniq = sorted(set(ad_ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                consider(uniq[i], uniq[j], 0)

    valid["_bucket"] = valid["phash"].str[:4]
    for _, bucket in valid.groupby("_bucket"):
        rows = list(bucket[["ad_id", "phash"]].itertuples(index=False))
        for i in range(len(rows)):
            hi = imagehash.hex_to_hash(rows[i].phash)
            for j in range(i + 1, len(rows)):
                if rows[i].ad_id == rows[j].ad_id:
                    continue
                hj = imagehash.hex_to_hash(rows[j].phash)
                dist = hi - hj
                if dist <= HAMMING_THRESHOLD:
                    consider(rows[i].ad_id, rows[j].ad_id, dist)

    records = []
    for (a, b), dist in pairs.items():
        if both_dealer_new(a, b):
            continue
        if not different_cars(a, b):
            continue
        ca = cars.loc[a] if a in cars.index else None
        cb = cars.loc[b] if b in cars.index else None
        records.append(
            {
                "ad_id_a": a,
                "ad_id_b": b,
                "hamming_distance": dist,
                "model_key_a": ca["model_key"] if ca is not None else "",
                "price_a": ca["price_tenge"] if ca is not None else None,
                "year_a": ca["year"] if ca is not None else None,
                "model_key_b": cb["model_key"] if cb is not None else "",
                "price_b": cb["price_tenge"] if cb is not None else None,
                "year_b": cb["year"] if cb is not None else None,
            }
        )

    cols = [
        "ad_id_a",
        "ad_id_b",
        "hamming_distance",
        "model_key_a",
        "price_a",
        "year_a",
        "model_key_b",
        "price_b",
        "year_b",
    ]

    out = (
        pd.DataFrame.from_records(records, columns=cols) if records else pd.DataFrame(columns=cols)
    )
    if not out.empty:
        out = out.sort_values("hamming_distance")
    return out


def main():
    collect_hashes()

    engine = get_engine()
    with engine.begin() as conn:
        has_clean = conn.execute(text("SELECT to_regclass('public.clean_data')")).scalar()
    if not has_clean:
        log.warning("clean_data was not found; skipping vehicle-level grouping.")
        pd.DataFrame(columns=["ad_id_a", "ad_id_b"]).to_csv(DUPLICATES_CSV, index=False)
        return

    hashes = pd.read_csv(HASHES_CSV, dtype={"ad_id": str})
    clean = pd.read_sql(
        "SELECT ad_id, brand, model, year, price_tenge, condition, labels FROM clean_data",
        engine,
        dtype={"ad_id": str},
    )
    dups = find_cross_car_duplicates(hashes, clean)
    dups.to_csv(DUPLICATES_CSV, index=False)

    with engine.begin() as conn:
        dups.to_sql("photo_duplicates", conn, if_exists="replace", index=False)
    log.info(
        f"Found {len(dups)} 'same photo, different vehicle' pairs → {DUPLICATES_CSV} "
        f"and the photo_duplicates table"
    )


if __name__ == "__main__":
    main()
