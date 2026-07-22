-- schema.sql — DDL для сырых/резюмируемых слоёв пайплайна.
--
-- НЕ включены сюда: clean_data и photo_duplicates. Обе таблицы
-- disposable (пересобираются каждый прогон из raw) — clean.py/
-- photo_dedup.py создают их сами через
-- `df.to_sql(..., if_exists="replace")` внутри транзакции, схема
-- всегда выводится из текущего DataFrame и не расходится с кодом.
-- manual_labels.csv тоже не мигрируется — это файл, который человек
-- правит руками, тащить его в psql — трение без пользы.
--
-- Применить один раз: psql "$DATABASE_URL" -f schema.sql
-- (или через миграционный скрипт — см. migrate_to_postgres.py)

CREATE TABLE IF NOT EXISTS raw_ads (
    ad_id               TEXT PRIMARY KEY,
    url                 TEXT,
    title               TEXT,
    brand               TEXT,
    model               TEXT,
    price_tenge         BIGINT,
    year                SMALLINT,
    mileage_km          INTEGER,
    engine_volume       REAL,
    engine_type         TEXT,
    transmission        TEXT,
    body_type           TEXT,
    condition           TEXT,
    city                TEXT,
    description         TEXT,
    photos_count        INTEGER,
    photo_url           TEXT,
    views_count         INTEGER,
    posted_date         TEXT,
    labels              TEXT,
    is_vip              SMALLINT,          -- 0/1, как в raw_data.csv (не BOOLEAN —
    has_monthly_price   SMALLINT,          -- избегаем bigint→boolean cast при миграции)
    category            TEXT,
    scraped_at          TIMESTAMP
);

-- append-only: одна строка на (ad_id, seen_date) — "это объявление было
-- живо в листинге в этот день". UNIQUE делает повторный прогон parser.py
-- в тот же день идемпотентным (ON CONFLICT DO NOTHING на стороне кода).
CREATE TABLE IF NOT EXISTS sightings (
    ad_id        TEXT NOT NULL,
    seen_date    DATE NOT NULL,
    price_tenge  BIGINT,
    views_count  INTEGER,
    is_vip       SMALLINT,
    category     TEXT,
    UNIQUE (ad_id, seen_date)
);

-- пишется один раз при первом появлении ad_id, дальше не трогается
-- (подтверждено чтением parser.py) — UNIQUE тоже для идемпотентности.
CREATE TABLE IF NOT EXISTS photos (
    ad_id      TEXT NOT NULL,
    position   SMALLINT NOT NULL,
    url        TEXT,
    UNIQUE (ad_id, position)
);

-- Однострочный upsert по ad_id, НЕ time-series: сам пайплайн трактует
-- "последняя запись побеждает", историю статусов никто не читает.
CREATE TABLE IF NOT EXISTS ad_status (
    ad_id       TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    checked_at  TIMESTAMP NOT NULL
);

-- upsert по ad_id; load_done() в enrich.py никогда не перезаписывает
-- уже готовую строку — ON CONFLICT DO NOTHING на стороне кода.
CREATE TABLE IF NOT EXISTS enriched (
    ad_id              TEXT PRIMARY KEY,
    fetched_at         TIMESTAMP,
    http_status        INTEGER,
    is_archived        SMALLINT,
    customs_cleared    TEXT,
    drive              TEXT,
    steering           TEXT,
    color              TEXT,
    generation         TEXT,
    page_mileage_km    INTEGER,
    page_condition     TEXT,
    has_vin            TEXT,
    damage_keywords    TEXT,
    seller_comment     TEXT,
    options_text       TEXT,
    kolesa_avg_price   BIGINT,     -- средняя рыночная цена от kolesa (кросс-чек детектора)
    page_status_badge  TEXT        -- бейдж сайта: Аварийная/Не на ходу, Заложенная и т.п.
);

-- append-only кэш pHash (photo_dedup.py); UNIQUE = резюмируемость,
-- тот же паттерн, что load_done() в enrich.py.
CREATE TABLE IF NOT EXISTS photo_hashes (
    ad_id        TEXT NOT NULL,
    position     SMALLINT NOT NULL,
    url          TEXT,
    phash        TEXT,
    fetched_at   TIMESTAMP,
    http_status  INTEGER,
    UNIQUE (ad_id, position)
);
