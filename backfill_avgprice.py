# -*- coding: utf-8 -*-
"""
backfill_avgprice.py — разовое дозаполнение полей со страницы объявления
(kolesa_avg_price И page_status_badge) у уже обогащённых объявлений.

Зачем отдельно: обе колонки добавили ПОСЛЕ того, как часть объявлений уже
обогатилась, и их страницы тогда парсились без них. Значения живут только
на живой странице (не выводятся из наших данных), поэтому единственный
способ дозаполнить старые строки — перекачать их страницы. Новые
объявления enrich.py заполняет сам, бесплатно.

Ничего не удаляет и заново не парсит листинг. Целится в enriched-строки,
где пусто ЛЮБОЕ из двух полей; заполняет через COALESCE — уже заполненное
НЕ трогает (напр. avgPrice, добранный прошлым заходом, переживёт добор
бейджа). Подозрительные первыми. Бюджет и паузы — как у enrich.py.

Запуск: python backfill_avgprice.py            (следующая порция ~120)
        python backfill_avgprice.py --all       (все разом, ~1.5 часа, риск лимитов)

Рекомендация: гоняй БЕЗ --all по паре раз в день — резюмируемо и безопасно
по лимитам; прервать/продолжить можно в любой момент.
"""

import pathlib as _p
_expected = "backfill_avgprice.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import csv
import logging
import random
import sys
import time

import pandas as pd
import requests
from sqlalchemy import text

from db import get_engine
from enrich import HEADERS, extract_avg_price, extract_status_badge, ENRICHED_CSV

MAX_PER_RUN           = 120
DELAY_RANGE           = (4.0, 8.0)
MAX_CONSECUTIVE_FAILS = 3           # предохранитель: N сбоев подряд → стоп
LOG_FILE              = "logs/enrich.log"   # тот же лог, что у обогащения

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()])
log = logging.getLogger(__name__)


def pick_targets() -> list[str]:
    """enriched-строки, где пусто ЛЮБОЕ из двух полей; подозрительные вперёд."""
    engine = get_engine()
    todo = pd.read_sql(
        "SELECT ad_id FROM enriched WHERE http_status = 200 "
        "AND (kolesa_avg_price IS NULL OR page_status_badge IS NULL)",
        engine, dtype={"ad_id": str})
    susp = pd.read_sql("SELECT ad_id, is_suspicious FROM clean_data",
                        engine, dtype={"ad_id": str})
    todo = todo.merge(susp, on="ad_id", how="left")
    todo["is_suspicious"] = todo["is_suspicious"].fillna(0)
    todo = todo.sort_values("is_suspicious", ascending=False)
    ids = todo["ad_id"].tolist()
    return ids if "--all" in sys.argv else ids[:MAX_PER_RUN]


def update_stores(ad_id: str, avg, badge: str):
    """Пишем в Postgres и в CSV-снимок. COALESCE/fill-if-empty: уже
    заполненные поля НЕ трогаем — добор одного поля не затирает другое."""
    with get_engine().begin() as conn:
        conn.execute(text(
            "UPDATE enriched SET "
            "kolesa_avg_price = COALESCE(kolesa_avg_price, :a), "
            "page_status_badge = COALESCE(NULLIF(page_status_badge, ''), :b) "
            "WHERE ad_id = :id"),
            {"a": avg, "b": badge, "id": ad_id})
    # CSV: csv-модулем (не pandas — иначе float-порча целых колонок), только пустые
    rows = list(csv.DictReader(open(ENRICHED_CSV, encoding="utf-8")))
    fields = list(rows[0].keys()) if rows else []
    for r in rows:
        if r["ad_id"] == ad_id:
            if not str(r.get("kolesa_avg_price", "")).strip():
                r["kolesa_avg_price"] = str(avg)
            if not str(r.get("page_status_badge", "")).strip():
                r["page_status_badge"] = badge
    with open(ENRICHED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    targets = pick_targets()
    log.info(f"бэкфилл avgPrice+бейдж: к дозаполнению {len(targets)}")
    session = requests.Session()
    avg_filled = badge_filled = 0
    fails = 0
    for i, ad_id in enumerate(targets, 1):
        try:
            resp = session.get(f"https://kolesa.kz/a/show/{ad_id}",
                               headers=HEADERS, timeout=20)
        except requests.RequestException as e:
            log.warning(f"{ad_id}: {e}")
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Стоп: сбои подряд — продолжим в другой раз.")
                sys.exit(1)
            time.sleep(30)
            continue

        # 429 = сайт просит притормозить. Это инструкция, а не ошибка:
        # тормозим НАДОЛГО и считаем к предохранителю (иначе --all
        # молотил бы сотни запросов сквозь rate-limit → бан).
        if resp.status_code == 429:
            log.warning("429: пауза 120с")
            time.sleep(120)
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Стоп: 429 подряд — сайт лимитирует, продолжим позже.")
                sys.exit(1)
            continue
        fails = 0

        if resp.status_code == 200:
            avg = extract_avg_price(resp.text)      # число или None
            badge = extract_status_badge(resp.text)  # текст или "-"
        else:
            avg, badge = None, "-"
        # avgPrice: -1 = «у модели нет эталона», чтобы не перекачивать снова
        update_stores(ad_id, avg if avg is not None else -1, badge)
        if avg:
            avg_filled += 1
        if badge and badge != "-":
            badge_filled += 1
        if i % 20 == 0:
            log.info(f"  {i}/{len(targets)} (avgPrice: {avg_filled}, бейджей: {badge_filled})")
        time.sleep(random.uniform(*DELAY_RANGE))
    log.info(f"Готово из {len(targets)}: avgPrice дозаполнено {avg_filled}, "
             f"статус-бейджей найдено {badge_filled}")


if __name__ == "__main__":
    main()
