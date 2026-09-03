# -*- coding: utf-8 -*-
"""Implementation for the `kz.ops.migrate_to_postgres` module."""

import pathlib as _p

_expected = "migrate_to_postgres.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files were mixed up during copying."
    )


import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from kz.core.db import get_engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


TABLES = [
    ("data/raw/raw_data.csv", "raw_ads", ["ad_id"], False, ["scraped_at"]),
    ("data/raw/sightings.csv", "sightings", ["ad_id", "seen_date"], False, ["seen_date"]),
    ("data/raw/photos.csv", "photos", ["ad_id", "position"], False, []),
    ("data/raw/ad_status.csv", "ad_status", ["ad_id"], True, ["checked_at"]),
    ("data/enriched/enriched.csv", "enriched", ["ad_id"], False, ["fetched_at"]),
    (
        "data/enriched/photo_hashes.csv",
        "photo_hashes",
        ["ad_id", "position"],
        False,
        ["fetched_at"],
    ),
]


def migrate_table(
    csv_path: str, table: str, key_cols: list[str], dedup: bool, date_cols: list[str]
):
    if not Path(csv_path).exists():
        log.info(f"{csv_path} not found; skipping")
        return

    df = pd.read_csv(csv_path, dtype={"ad_id": str})
    if dedup:
        df = df.drop_duplicates(subset="ad_id", keep="last")
    if df.empty:
        log.info(f"{csv_path} is empty; skipping")
        return
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    engine = get_engine()
    staging = f"_staging_{table}"
    df.to_sql(staging, engine, if_exists="replace", index=False)

    cols = ", ".join(df.columns)
    keys = ", ".join(key_cols)
    with engine.begin() as conn:
        conn.execute(
            text(
                f'INSERT INTO {table} ({cols}) SELECT {cols} FROM "{staging}" '
                f"ON CONFLICT ({keys}) DO NOTHING"
            )
        )
        conn.execute(text(f'DROP TABLE "{staging}"'))
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    log.info(f"{table:<15} csv={len(df):>6} rows → table now contains {count}")


def main():
    for csv_path, table, keys, dedup, date_cols in TABLES:
        migrate_table(csv_path, table, keys, dedup, date_cols)
    log.info("Done. manual_labels.csv and clean_data are not migrated; see the docstring.")


if __name__ == "__main__":
    main()
