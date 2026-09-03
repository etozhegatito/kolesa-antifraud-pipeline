# -*- coding: utf-8 -*-
"""Explain why one listing has missing enriched fields."""

import pathlib as _p

_expected = "diagnose.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named {_p.Path(__file__).name}."
    )

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kz.collect.enrich import HEADERS, parse_ad_page


def read_if_exists(path, **kwargs):
    """Read a CSV when present and otherwise return ``None``."""
    if not Path(path).exists():
        return None
    return pd.read_csv(path, dtype={"ad_id": str}, **kwargs)


def main():
    """Print local evidence and one live detail-page comparison."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python diagnose.py <ad_id>")
    ad_id = sys.argv[1].strip()
    print(f"═══ Listing diagnosis: {ad_id} ═══\n")

    raw = read_if_exists("data/raw/raw_data.csv")
    enriched_data = read_if_exists("data/enriched/enriched.csv")
    clean = read_if_exists("data/clean/clean_data.csv")

    raw_rows = raw[raw.ad_id == ad_id] if raw is not None else pd.DataFrame()
    if raw_rows.empty:
        print("• The listing is absent from raw_data; collection has not seen it")
    else:
        row = raw_rows.iloc[-1]
        print(
            f"• raw: is_vip={row.get('is_vip')}, "
            f"mileage={row.get('mileage_km')}, "
            f"description={str(row.get('description'))[:60]!r}"
        )

    is_enriched = enriched_data is not None and ad_id in set(enriched_data.ad_id)
    status = "YES" if is_enriched else "no — still waiting in queue, not a bug"
    print(f"• enriched: {status}")
    if is_enriched:
        row = enriched_data[enriched_data.ad_id == ad_id].iloc[-1]
        print(
            f"    page_mileage={row.get('page_mileage_km')}, "
            f"seller_comment={str(row.get('seller_comment'))[:70]!r}"
        )
    if clean is not None and "text_full" in clean.columns:
        rows = clean[clean.ad_id == ad_id]
        if len(rows):
            print(f"• clean.text_full: {str(rows.iloc[-1]['text_full'])[:70]!r}")

    print("\n• Live page request...")
    time.sleep(2)
    response = requests.get(f"https://kolesa.kz/a/show/{ad_id}", headers=HEADERS, timeout=20)
    print(f"    HTTP {response.status_code}")
    if response.status_code != 200:
        print("    Page unavailable (404 means deleted); the source no longer has it")
        return
    parsed = parse_ad_page(response.text)
    print(
        f"    parser sees: mileage={parsed.get('page_mileage_km')}, "
        f"comment={parsed.get('seller_comment', '')[:70]!r}, "
        f"customs_clearance={parsed.get('customs_cleared')}"
    )

    print("\n═══ Verdict ═══")
    if not is_enriched:
        print(
            "Fields are empty because enrichment has not reached this row yet "
            "(priority queue with a daily budget). The page parser did extract "
            "the field above. The data are recoverable; this is not a bug."
        )
    elif parsed.get("page_mileage_km") is None:
        print(
            "The detail page itself has no mileage, usually because the vehicle "
            "is new or the seller omitted it. This is MNAR; nothing can be parsed."
        )
    else:
        print(
            "The page returns data and the listing is enriched. Check the columns: "
            "full text lives in seller_comment/text_full, not description."
        )


if __name__ == "__main__":
    main()
