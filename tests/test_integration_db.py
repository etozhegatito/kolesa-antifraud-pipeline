# -*- coding: utf-8 -*-
"""Тесты, которые исполняют НАСТОЯЩИЙ SQL.

ЗАЧЕМ ОТДЕЛЬНЫМ ФАЙЛОМ. Весь остальной набор офлайновый: он проверяет чистые
функции и не поднимает базу. Это правильно для скорости, но оставляет целый
слепой класс — ни один запрос проекта не исполнялся в CI ни разу. Сломанный
JOIN или разъехавшаяся с кодом схема обнаруживались только на машине
разработчика, и то случайно.

Дважды за одну сессию выстрелила ровно эта форма ошибки: пины версий,
снятые с локального Python, не ставились на CI; тест читал каталог, которого
в чистом клоне нет. Оба раза локально было зелено. SQL — та же дыра, просто
ещё не выстрелившая.

ПОЧЕМУ БЕЗОПАСНО ЗАПУСКАТЬ ЛОКАЛЬНО. Тесты создают схему и пишут строки,
поэтому направить их на рабочую базу нельзя — они бы затёрли собранные
данные. Защита двойная:

  1. используется отдельное подключение из KZ_TEST_DSN, а если его нет —
     собранное из POSTGRES_*, и ТОЛЬКО когда имя базы содержит «test»;
  2. таблицы создаются с префиксом и удаляются после прогона.

Если подходящей базы нет, тесты пропускаются — кроме случая CI, где
пропуск означал бы, что мы завели сервис Postgres и ничего им не проверили.
Там отсутствие базы — ошибка.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

SCHEMA_FILE = "sql/init/01_schema.sql"


def _test_dsn() -> str | None:
    """Строка подключения, которую МОЖНО затирать."""
    dsn = os.environ.get("KZ_TEST_DSN")
    if dsn:
        return dsn
    user = os.environ.get("POSTGRES_USER")
    pwd = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")
    if not (user and pwd and db):
        return None
    # Единственный предохранитель от «прогнал тесты — потерял базу»: имя
    # рабочей базы (market_db) слова «test» не содержит.
    if "test" not in db.lower():
        return None
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


@pytest.fixture(scope="module")
def engine():
    """Движок к тестовой базе со свежей схемой."""
    dsn = _test_dsn()
    in_ci = os.environ.get("CI", "").lower() == "true"
    if not dsn:
        if in_ci:
            pytest.fail("В CI поднят Postgres, но подключиться не удалось: "
                        "проверьте POSTGRES_* и что имя базы содержит 'test'")
        pytest.skip("нет тестовой базы (задайте KZ_TEST_DSN или POSTGRES_DB=*test*)")

    eng = create_engine(dsn)
    try:
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:                      # noqa: BLE001
        if in_ci:
            pytest.fail(f"В CI база недоступна: {e}")
        pytest.skip(f"база недоступна: {type(e).__name__}")

    ddl = open(SCHEMA_FILE, encoding="utf-8").read()
    with eng.begin() as c:
        for tbl in ("raw_ads", "sightings", "photos", "ad_status",
                    "enriched", "photo_hashes"):
            c.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE"))
        c.execute(text("DROP TABLE IF EXISTS clean_data CASCADE"))
        c.execute(text(ddl))
    yield eng
    eng.dispose()


@pytest.fixture
def clean_tables(engine):
    """Каждый тест начинает с пустых таблиц."""
    with engine.begin() as c:
        for tbl in ("raw_ads", "sightings", "photos", "ad_status",
                    "enriched", "photo_hashes"):
            c.execute(text(f"TRUNCATE {tbl} CASCADE"))
    return engine


def _patch_engine(monkeypatch, eng):
    """Направить код проекта на тестовую базу вместо рабочей."""
    import kz.core.db as db
    db.get_engine.cache_clear()
    monkeypatch.setattr(db, "get_engine", lambda: eng)
    return db


# ─── db.upsert: единственный путь записи в проекте ──────────────────────────

def test_upsert_do_nothing_is_idempotent(clean_tables, monkeypatch):
    """Append-only джобы (photos, sightings) обязаны переживать повторный
    запуск: у них ON CONFLICT DO NOTHING, и второй прогон не должен ни
    падать, ни задваивать строки. Резюмируемость всего сбора держится на
    этом свойстве."""
    db = _patch_engine(monkeypatch, clean_tables)
    rows = [{"ad_id": "1", "position": 1, "url": "http://x/1.jpg"},
            {"ad_id": "1", "position": 2, "url": "http://x/2.jpg"}]
    db.upsert("photos", rows, ["ad_id", "position"])
    db.upsert("photos", rows, ["ad_id", "position"])      # повторный прогон
    n = pd.read_sql("SELECT COUNT(*) c FROM photos", clean_tables).c[0]
    assert n == 2


def test_upsert_do_update_keeps_the_latest_status(clean_tables, monkeypatch):
    """ad_status — не история, а «последняя проверка побеждает». Если бы
    здесь стоял DO NOTHING, объявление навсегда осталось бы в первом
    увиденном статусе и никогда не стало бы archived."""
    db = _patch_engine(monkeypatch, clean_tables)
    db.upsert("ad_status", [{"ad_id": "1", "status": "active",
                             "checked_at": "2026-08-01"}], ["ad_id"])
    db.upsert("ad_status", [{"ad_id": "1", "status": "archived",
                             "checked_at": "2026-08-20"}], ["ad_id"],
              update_cols=["status", "checked_at"])
    got = pd.read_sql("SELECT status FROM ad_status WHERE ad_id='1'",
                      clean_tables).status[0]
    assert got == "archived"


def test_upsert_of_nothing_is_not_an_error(clean_tables, monkeypatch):
    """Пустая порция случается штатно: джоб добрал всё и на следующем круге
    писать нечего. Падать на этом нельзя."""
    db = _patch_engine(monkeypatch, clean_tables)
    db.upsert("photos", [], ["ad_id", "position"])        # не должно бросить


# ─── Запросы, которые до сих пор исполнялись только вручную ─────────────────

def test_enrichment_queue_join_actually_runs(clean_tables, monkeypatch):
    """Очередь обогащения соединяет clean_data с raw_ads ради scraped_at.
    Этот JOIN написан сегодня и до сих пор ни разу не исполнялся в CI: любая
    опечатка в имени колонки всплыла бы только у пользователя."""
    eng = clean_tables
    _patch_engine(monkeypatch, eng)
    with eng.begin() as c:
        c.execute(text(
            "INSERT INTO raw_ads (ad_id, scraped_at) VALUES "
            "('old', '2026-07-01'), ('new', '2026-08-20'), ('susp', '2026-07-01')"))
    pd.DataFrame({"ad_id": ["old", "new", "susp"],
                  "is_suspicious": [0, 0, 1],
                  # pick_targets приоритизирует дешёвый сегмент, поэтому
                  # реальный контракт запроса включает price_tenge. Fixture
                  # обязан повторять его схему, иначе CI проверяет старую
                  # версию функции и падает на корректном SQL.
                  "price_tenge": [10_000_000, 10_000_000, 10_000_000]}).to_sql(
        "clean_data", eng, if_exists="replace", index=False)

    from kz.collect import enrich
    monkeypatch.setattr(enrich, "get_engine", lambda: eng)
    assert enrich.pick_targets(set()) == ["susp", "new", "old"]
    assert enrich.pick_targets({"susp"}) == ["new", "old"]


def test_freshness_queries_run_against_a_real_schema(clean_tables, monkeypatch):
    """freshness.measure() делает четыре запроса подряд. Он печатается первым
    в каждом отчёте, поэтому его падение обесценивает весь вывод."""
    eng = clean_tables
    _patch_engine(monkeypatch, eng)
    with eng.begin() as c:
        c.execute(text("INSERT INTO sightings (ad_id, seen_date, price_tenge) "
                       "VALUES ('1','2026-08-01',100), ('1','2026-08-05',100)"))
        c.execute(text("INSERT INTO ad_status (ad_id, status, checked_at) "
                       "VALUES ('1','active','2026-08-05')"))
    pd.DataFrame({"ad_id": ["1", "2"]}).to_sql("clean_data", eng,
                                               if_exists="replace", index=False)

    from kz.core import freshness
    f = freshness.measure()
    assert f.collect_days == 2
    assert f.span_days == 5
    assert f.ads_total == 2 and f.ads_status_checked == 1
    assert 0 < f.status_coverage < 1
    assert freshness.stale_warnings(f)          # покрытие 50% → предупреждение


def test_schema_matches_what_the_code_inserts(clean_tables):
    """DDL и код обязаны сходиться по колонкам. Схема лежит в файле, а пишут
    в неё словари, собранные в питоне: разъехаться они могут молча, и
    заметно это станет на живом сборе."""
    from kz.collect.parser import FIELDS, PHOTO_FIELDS, SIGHTING_FIELDS

    def columns_of(table: str) -> set:
        return set(pd.read_sql(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name='{table}'", clean_tables).column_name)

    missing = {}
    for table, fields in [("raw_ads", FIELDS), ("sightings", SIGHTING_FIELDS),
                          ("photos", PHOTO_FIELDS)]:
        gap = [f for f in fields if f not in columns_of(table)]
        if gap:
            missing[table] = gap
    assert not missing, f"парсер пишет поля, которых нет в схеме: {missing}"
