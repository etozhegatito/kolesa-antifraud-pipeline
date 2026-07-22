# -*- coding: utf-8 -*-
"""
db.py — тонкий слой доступа к Postgres. Никакого ORM: pandas.read_sql/
to_sql закрывают чтение, один helper upsert() закрывает запись с
ON CONFLICT. Осознанно маленький файл — проект однопользовательский,
не тянет модельный слой.
"""

from functools import lru_cache

import psycopg2.extras
from sqlalchemy import create_engine, text

from config import DATABASE_URL


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(DATABASE_URL)


def upsert(table: str, rows: list[dict], conflict_cols: list[str],
           update_cols: list[str] | None = None):
    """INSERT нескольких строк с ON CONFLICT. update_cols=None (или []) →
    ON CONFLICT DO NOTHING (append-only/резюмируемые джобы); update_cols
    задан → DO UPDATE SET (для "последняя запись побеждает", напр. ad_status).
    rows — список словарей с ОДИНАКОВЫМ набором ключей."""
    if not rows:
        return
    cols = list(rows[0].keys())
    values = [[r[c] for c in cols] for r in rows]

    col_list = ", ".join(cols)
    conflict = ", ".join(conflict_cols)
    if update_cols:
        set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        on_conflict = f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"
    else:
        on_conflict = f"ON CONFLICT ({conflict}) DO NOTHING"

    sql = f"INSERT INTO {table} ({col_list}) VALUES %s {on_conflict}"

    engine = get_engine()
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values)
        raw.commit()
    finally:
        raw.close()


def run_sql(sql: str, params: dict | None = None):
    """Разовый DDL/UPDATE без возврата строк (например TRUNCATE)."""
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})
