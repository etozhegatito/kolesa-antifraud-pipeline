# -*- coding: utf-8 -*-
"""
migrate_to_postgres.py — одноразовый идемпотентный бэкфилл CSV → Postgres.

Безопасно перезапускать: staging-таблица каждый раз пересоздаётся
(`to_sql(if_exists="replace")`), а перенос в целевую идёт через
`ON CONFLICT DO NOTHING` — если целевая таблица уже содержит более
свежие данные (например, джобы уже переключены на живую запись в
Postgres после первого запуска этой миграции), устаревший CSV-снимок
их не перезатрёт.

НЕ переносятся: manual_labels.csv (правится человеком руками, тащить
в psql — трение без пользы) и clean_data (disposable, пересобирается
из raw самим clean.py при первом же прогоне после миграции).

Перед первым запуском:
    docker compose up -d
    psql "$DATABASE_URL" -f schema.sql

Запуск: python migrate_to_postgres.py
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
import pathlib as _p
_expected = "migrate_to_postgres.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# (csv-файл, таблица, ключевые колонки для ON CONFLICT, дедуп CSV перед
#  заливкой, колонки-даты — иначе staging заведёт их как TEXT и упрётся
#  в "column is of type timestamp but expression is of type text")
TABLES = [
    ("data/raw/raw_data.csv", "raw_ads", ["ad_id"], False, ["scraped_at"]),
    ("data/raw/sightings.csv", "sightings", ["ad_id", "seen_date"], False, ["seen_date"]),
    ("data/raw/photos.csv", "photos", ["ad_id", "position"], False, []),
    # ad_status.csv исторически append-only (может содержать несколько
    # строк на ad_id) — целевая таблица однострочная, дедуп обязателен
    ("data/raw/ad_status.csv", "ad_status", ["ad_id"], True, ["checked_at"]),
    ("data/enriched/enriched.csv", "enriched", ["ad_id"], False, ["fetched_at"]),
    ("data/enriched/photo_hashes.csv", "photo_hashes", ["ad_id", "position"], False, ["fetched_at"]),
]


def migrate_table(csv_path: str, table: str, key_cols: list[str], dedup: bool,
                   date_cols: list[str]):
    if not Path(csv_path).exists():
        log.info(f"{csv_path} не найден — пропуск")
        return

    df = pd.read_csv(csv_path, dtype={"ad_id": str})
    if dedup:
        df = df.drop_duplicates(subset="ad_id", keep="last")
    if df.empty:
        log.info(f"{csv_path} пуст — пропуск")
        return
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    engine = get_engine()
    staging = f"_staging_{table}"
    df.to_sql(staging, engine, if_exists="replace", index=False)

    cols = ", ".join(df.columns)
    keys = ", ".join(key_cols)
    with engine.begin() as conn:
        conn.execute(text(
            f'INSERT INTO {table} ({cols}) SELECT {cols} FROM "{staging}" '
            f'ON CONFLICT ({keys}) DO NOTHING'))
        conn.execute(text(f'DROP TABLE "{staging}"'))
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    log.info(f"{table:<15} csv={len(df):>6} строк → в таблице теперь {count}")


def main():
    for csv_path, table, keys, dedup, date_cols in TABLES:
        migrate_table(csv_path, table, keys, dedup, date_cols)
    log.info("Готово. manual_labels.csv и clean_data не мигрируются (см. docstring).")


if __name__ == "__main__":
    main()
