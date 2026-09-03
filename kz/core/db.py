# -*- coding: utf-8 -*-
"""Small PostgreSQL access layer using Pandas, SQLAlchemy, and one upsert helper."""

from functools import lru_cache

import psycopg2.extras
from sqlalchemy import create_engine

from kz.core.config import DATABASE_URL


@lru_cache(maxsize=1)
def get_engine():
    """Return the cached SQLAlchemy engine or a clear configuration error."""
    if not DATABASE_URL:
        raise RuntimeError(
            "PostgreSQL settings are missing (POSTGRES_USER/PASSWORD/DB). "
            "Create .env from .env.example for pipeline work. The public "
            "model-only estimator does not require a database."
        )
    return create_engine(DATABASE_URL)


def upsert(
    table: str, rows: list[dict], conflict_cols: list[str], update_cols: list[str] | None = None
):
    """Insert rows with ``ON CONFLICT``.

    No update columns means append-only ``DO NOTHING``; otherwise listed
    columns use last-write-wins ``DO UPDATE``. Every row must share keys.
    """
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
