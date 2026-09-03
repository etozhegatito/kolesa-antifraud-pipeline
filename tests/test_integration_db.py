# -*- coding: utf-8 -*-
"""Implementation for the `tests.test_integration_db` module."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

SCHEMA_FILE = "sql/init/01_schema.sql"


def _test_dsn() -> str | None:
    """Implement `_test_dsn`."""
    dsn = os.environ.get("KZ_TEST_DSN")
    if dsn:
        return dsn
    user = os.environ.get("POSTGRES_USER")
    pwd = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")
    if not (user and pwd and db):
        return None

    if "test" not in db.lower():
        return None
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


@pytest.fixture(scope="module")
def engine():
    """Implement `engine`."""
    dsn = _test_dsn()
    in_ci = os.environ.get("CI", "").lower() == "true"
    if not dsn:
        if in_ci:
            pytest.fail(
                "В CI поднят Postgres, но подключиться не удалось: "
                "проверьте POSTGRES_* и что имя базы содержит 'test'"
            )
        pytest.skip("нет тестовой базы (задайте KZ_TEST_DSN или POSTGRES_DB=*test*)")

    eng = create_engine(dsn)
    try:
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        if in_ci:
            pytest.fail(f"В CI база недоступна: {e}")
        pytest.skip(f"база недоступна: {type(e).__name__}")

    ddl = open(SCHEMA_FILE, encoding="utf-8").read()
    with eng.begin() as c:
        for tbl in ("raw_ads", "sightings", "photos", "ad_status", "enriched", "photo_hashes"):
            c.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE"))
        c.execute(text("DROP TABLE IF EXISTS clean_data CASCADE"))
        c.execute(text(ddl))
    yield eng
    eng.dispose()


@pytest.fixture
def clean_tables(engine):
    """Implement `clean_tables`."""
    with engine.begin() as c:
        for tbl in ("raw_ads", "sightings", "photos", "ad_status", "enriched", "photo_hashes"):
            c.execute(text(f"TRUNCATE {tbl} CASCADE"))
    return engine


def _patch_engine(monkeypatch, eng):
    """Implement `_patch_engine`."""
    import kz.core.db as db

    db.get_engine.cache_clear()
    monkeypatch.setattr(db, "get_engine", lambda: eng)
    return db


def test_upsert_do_nothing_is_idempotent(clean_tables, monkeypatch):
    """Regression coverage for `test_upsert_do_nothing_is_idempotent`."""
    db = _patch_engine(monkeypatch, clean_tables)
    rows = [
        {"ad_id": "1", "position": 1, "url": "http://x/1.jpg"},
        {"ad_id": "1", "position": 2, "url": "http://x/2.jpg"},
    ]
    db.upsert("photos", rows, ["ad_id", "position"])
    db.upsert("photos", rows, ["ad_id", "position"])
    n = pd.read_sql("SELECT COUNT(*) c FROM photos", clean_tables).c[0]
    assert n == 2


def test_upsert_do_update_keeps_the_latest_status(clean_tables, monkeypatch):
    """Regression coverage for `test_upsert_do_update_keeps_the_latest_status`."""
    db = _patch_engine(monkeypatch, clean_tables)
    db.upsert(
        "ad_status", [{"ad_id": "1", "status": "active", "checked_at": "2026-08-01"}], ["ad_id"]
    )
    db.upsert(
        "ad_status",
        [{"ad_id": "1", "status": "archived", "checked_at": "2026-08-20"}],
        ["ad_id"],
        update_cols=["status", "checked_at"],
    )
    got = pd.read_sql("SELECT status FROM ad_status WHERE ad_id='1'", clean_tables).status[0]
    assert got == "archived"


def test_upsert_of_nothing_is_not_an_error(clean_tables, monkeypatch):
    """Regression coverage for `test_upsert_of_nothing_is_not_an_error`."""
    db = _patch_engine(monkeypatch, clean_tables)
    db.upsert("photos", [], ["ad_id", "position"])


def test_enrichment_queue_join_actually_runs(clean_tables, monkeypatch):
    """Regression coverage for `test_enrichment_queue_join_actually_runs`."""
    eng = clean_tables
    _patch_engine(monkeypatch, eng)
    with eng.begin() as c:
        c.execute(
            text(
                "INSERT INTO raw_ads (ad_id, scraped_at) VALUES "
                "('old', '2026-07-01'), ('new', '2026-08-20'), ('susp', '2026-07-01')"
            )
        )
    pd.DataFrame(
        {
            "ad_id": ["old", "new", "susp"],
            "is_suspicious": [0, 0, 1],
            "price_tenge": [10_000_000, 10_000_000, 10_000_000],
        }
    ).to_sql("clean_data", eng, if_exists="replace", index=False)

    from kz.collect import enrich

    monkeypatch.setattr(enrich, "get_engine", lambda: eng)
    assert enrich.pick_targets(set()) == ["susp", "new", "old"]
    assert enrich.pick_targets({"susp"}) == ["new", "old"]


def test_freshness_queries_run_against_a_real_schema(clean_tables, monkeypatch):
    """Regression coverage for `test_freshness_queries_run_against_a_real_schema`."""
    eng = clean_tables
    _patch_engine(monkeypatch, eng)
    with eng.begin() as c:
        c.execute(
            text(
                "INSERT INTO sightings (ad_id, seen_date, price_tenge) "
                "VALUES ('1','2026-08-01',100), ('1','2026-08-05',100)"
            )
        )
        c.execute(
            text(
                "INSERT INTO ad_status (ad_id, status, checked_at) "
                "VALUES ('1','active','2026-08-05')"
            )
        )
    pd.DataFrame({"ad_id": ["1", "2"]}).to_sql("clean_data", eng, if_exists="replace", index=False)

    from kz.core import freshness

    f = freshness.measure()
    assert f.collect_days == 2
    assert f.span_days == 5
    assert f.ads_total == 2 and f.ads_status_checked == 1
    assert 0 < f.status_coverage < 1
    assert freshness.stale_warnings(f)


def test_schema_matches_what_the_code_inserts(clean_tables):
    """Regression coverage for `test_schema_matches_what_the_code_inserts`."""
    from kz.collect.parser import FIELDS, PHOTO_FIELDS, SIGHTING_FIELDS

    def columns_of(table: str) -> set:
        return set(
            pd.read_sql(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'",
                clean_tables,
            ).column_name
        )

    missing = {}
    for table, fields in [
        ("raw_ads", FIELDS),
        ("sightings", SIGHTING_FIELDS),
        ("photos", PHOTO_FIELDS),
    ]:
        gap = [f for f in fields if f not in columns_of(table)]
        if gap:
            missing[table] = gap
    assert not missing, f"парсер пишет поля, которых нет в схеме: {missing}"
