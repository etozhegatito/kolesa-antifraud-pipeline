# -*- coding: utf-8 -*-
"""Implementation for the `kz.collect.enrich` module."""

import pathlib as _p

_expected = "enrich.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import csv
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from kz.core import pacing
from kz.core.db import get_engine, upsert

ENRICHED_CSV = "data/enriched/enriched.csv"
LOG_FILE = "logs/enrich.log"

MAX_PER_RUN = 20
CHEAP_EDGE = 5_000_000
FRESH_RESERVE = 0.25
SUSPICIOUS_SHARE = 0.5
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


DT_MAP = {
    "Город": "page_city",
    "Поколение": "generation",
    "Кузов": "page_body",
    "Объем двигателя, л": "page_engine",
    "Пробег": "page_mileage",
    "Коробка передач": "page_transmission",
    "Привод": "drive",
    "Руль": "steering",
    "Цвет": "color",
    "Растаможен в Казахстане": "customs_cleared",
    "Аварийная": "is_emergency_field",
    "Состояние": "page_condition",
    "VIN": "has_vin",
}


def _normalise_parameter_label(value: str) -> str:
    """Implement `_normalise_parameter_label`."""
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip().rstrip(":").strip()
    return value.casefold()


_NORMALISED_DT_MAP = {_normalise_parameter_label(label): field for label, field in DT_MAP.items()}
_NORMALISED_DT_MAP.update(
    {
        "состояние автомобиля": "page_condition",
        "vin-код": "has_vin",
        "vin код": "has_vin",
        "vin-номер": "has_vin",
        "vin номер": "has_vin",
    }
)

_VIN_NEGATIVE_VALUES = {"нет", "не указан", "не указано", "отсутствует", "-"}
_VIN_HISTORY_MARKER = "у этого объявления есть история авто"


def _vin_presence(value: str) -> str | None:
    """Implement `_vin_presence`."""
    normalised = _normalise_parameter_label(value)
    if not normalised:
        return None
    if normalised in _VIN_NEGATIVE_VALUES:
        return "Нет"
    return "Да"


from kz.transform.damage import DAMAGE_PATTERNS, find_damage_keywords  # noqa: F401

FIELDS = [
    "ad_id",
    "fetched_at",
    "http_status",
    "is_archived",
    "customs_cleared",
    "drive",
    "steering",
    "color",
    "generation",
    "page_mileage_km",
    "page_condition",
    "has_vin",
    "damage_keywords",
    "seller_comment",
    "options_text",
    "kolesa_avg_price",
    "page_status_badge",
]


_AVGPRICE_RE = re.compile(r'"avgPrice"\s*:\s*(\d+)')


def extract_avg_price(html: str):
    m = _AVGPRICE_RE.search(html)
    return int(m.group(1)) if m else None


def extract_status_badge(html: str) -> str:
    """Implement `extract_status_badge`."""
    badge = BeautifulSoup(html, "html.parser").select_one(".offer__parameters-mortgaged")
    return badge.get_text(" ", strip=True) if badge else "-"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ARCHIVE_MARKERS = ["в&nbsp;архиве", "в\u00a0архиве", "в архиве", ">В архиве<"]


_DESC_RE = re.compile(r'"descriptionText"\s*:\s*"((?:[^"\\]|\\.)*)"')


def extract_seller_comment(html: str) -> str:
    m = _DESC_RE.search(html)
    if not m:
        return ""
    raw = m.group(1)
    try:
        import json

        text = json.loads(f'"{raw}"')
    except Exception:
        text = raw

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:2000]


def parse_ad_page(html: str) -> dict:
    """Implement `parse_ad_page`."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}

    for dt in soup.select("dt"):
        dd = dt.find_next_sibling("dd")
        if dd is None and dt.parent is not None:
            dd = dt.parent.select_one("dd")
        if dd is None:
            continue
        label = _normalise_parameter_label(dt.get_text(" ", strip=True))
        key = _NORMALISED_DT_MAP.get(label)
        if key:
            value = dd.get_text(" ", strip=True)
            if key == "has_vin":
                value = _vin_presence(value)
            if value:
                out[key] = value

    page_text = _normalise_parameter_label(soup.get_text(" ", strip=True))
    if "has_vin" not in out and _VIN_HISTORY_MARKER in page_text:
        out["has_vin"] = "Да"

    if "page_mileage" in out:
        digits = re.sub(r"\D", "", out.pop("page_mileage"))
        out["page_mileage_km"] = int(digits) if digits else None

    opts = soup.select_one(".offer__description .text")
    options_text = opts.get_text(" ", strip=True) if opts else ""
    out["options_text"] = options_text[:800]

    seller_comment = extract_seller_comment(html)
    out["seller_comment"] = seller_comment

    h1 = soup.select_one("h1")
    searchable = (
        (h1.get_text(" ", strip=True) if h1 else "") + " " + options_text + " " + seller_comment
    )
    out["damage_keywords"] = "|".join(find_damage_keywords(searchable))

    out["is_archived"] = int(any(m in html for m in ARCHIVE_MARKERS))
    out["kolesa_avg_price"] = extract_avg_price(html)

    out["page_status_badge"] = extract_status_badge(html)
    return out


def load_done() -> set:
    """Implement `load_done`."""
    if not Path(ENRICHED_CSV).exists():
        with open(ENRICHED_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        csv_done = set()
    else:
        with open(ENRICHED_CSV, encoding="utf-8") as f:
            csv_done = {r["ad_id"] for r in csv.DictReader(f)}
    try:
        db_done = set(
            pd.read_sql("SELECT ad_id FROM enriched", get_engine(), dtype={"ad_id": str})[
                "ad_id"
            ].astype(str)
        )
    except Exception as e:
        log.warning(f"Could not reconcile enriched data with PostgreSQL: {e}")
        db_done = set()
    return csv_done | db_done


def pick_targets(done: set) -> list[str]:
    """Implement `pick_targets`."""
    df = pd.read_sql(
        "SELECT c.ad_id, c.is_suspicious, c.price_tenge, r.scraped_at "
        "FROM clean_data c JOIN raw_ads r ON r.ad_id = c.ad_id",
        get_engine(),
        dtype={"ad_id": str},
    )
    df = df[~df["ad_id"].isin(done)]
    if df.empty:
        return []
    df = df.sort_values("scraped_at", ascending=False)
    df["cheap"] = (df.price_tenge.fillna(0) > 0) & (df.price_tenge < CHEAP_EDGE)

    out: list[str] = []

    def take(rows, n):
        """Implement `take`."""
        for a in rows["ad_id"]:
            if len(out) >= n or len(out) >= MAX_PER_RUN:
                break
            if a not in out:
                out.append(a)

    take(df[df.is_suspicious == 1], round(MAX_PER_RUN * SUSPICIOUS_SHARE))

    take(df, len(out) + max(1, round(MAX_PER_RUN * FRESH_RESERVE)))

    take(df[df.cheap], MAX_PER_RUN)

    take(df, MAX_PER_RUN)
    return out


def main():
    done = load_done()
    targets = pick_targets(done)
    log.info(
        f"Pending enrichment: {len(targets)} (reserve {int(FRESH_RESERVE * 100)}% "
        f"for fresh listings, then anomalies and cheap cars; already done: {len(done)})"
    )

    session = requests.Session()
    fails = 0
    for i, ad_id in enumerate(targets, 1):
        url = f"https://kolesa.kz/a/show/{ad_id}"
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
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
                sys.exit(1)
            continue
        fails = 0

        row = {
            "ad_id": ad_id,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "http_status": resp.status_code,
        }
        if resp.status_code == 200:
            row.update(parse_ad_page(resp.text))

        with open(ENRICHED_CSV, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore").writerow(row)

        try:
            pg_row = {f: (row.get(f) if row.get(f) != "" else None) for f in FIELDS}
            upsert("enriched", [pg_row], ["ad_id"])
        except Exception as e:
            log.warning(f"PostgreSQL upsert failed for {ad_id}: {e}")

        if i % 20 == 0:
            log.info(f"  {i}/{len(targets)}")
        pacing.polite_sleep(i, DELAY_RANGE, log)

    log.info(f"Completed → {ENRICHED_CSV}")


if __name__ == "__main__":
    main()
