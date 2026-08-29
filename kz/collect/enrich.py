# -*- coding: utf-8 -*-
"""
Job 1b: обогащение (enrichment) — дотягиваем поля со страниц объявлений.

Зачем: детектор аномалий видит только листинг и не отличает
«мошенник-приманку» от «честной развалюхи за 200к». Разводящие их
признаки — растаможка, привод, руль, цвет, аварийность, слова про
состояние — живут на странице объявления. Она отдаётся простым
HTTP-запросом (мы проверяли), поэтому берём requests, не Playwright.

Приоритизированное обогащение: сначала обогащаем ПОДОЗРИТЕЛЬНЫЕ
(их ~46 — копейки запросов), потом остальных по лимиту. Принцип:
дорогой ресурс (запросы) тратим там, где выше ценность информации.

Ограничение (честно): полный «Комментарий продавца» подгружается
JS-ом и в статичном HTML отсутствует. Слова о состоянии ищем в том,
что доступно: заголовок + блок опций со страницы (+ обрезанное
описание из листинга уже есть в raw/clean).

Запуск: python -m kz.collect.enrich   (после clean.py, раз в день)
Выход:  enriched.csv (append-only, по строке на объявление)
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "enrich.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


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
LOG_FILE     = "logs/enrich.log"

MAX_PER_RUN           = 20      # мелкая порция: анти-бан (2026-07-23 IP словил блок на больших)
CHEAP_EDGE            = 5_000_000   # граница дешёвого сегмента, ₸
FRESH_RESERVE         = 0.25    # доля прогона на самые свежие вне зависимости от цены
SUSPICIOUS_SHARE      = 0.5     # потолок доли подозрительных в одном прогоне
DELAY_RANGE           = (4.0, 8.0)
MAX_CONSECUTIVE_FAILS = 3

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Маппинг «подпись на сайте → имя колонки». Всё, что не в маппинге,
# пропускаем (лучше явный список, чем случайные колонки из вёрстки).
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

# Лексикон «убитости» и поиск с учётом отрицаний — в damage.py
# (единственный источник; раньше список дублировался здесь и в clean.py).
# DAMAGE_PATTERNS здесь не вызывается, но импортируется НАМЕРЕННО:
# test_damage_patterns_in_sync сторожит, что enrich.py и clean.py смотрят
# в один лексикон, а не завели по своей копии. Копий уже было две, и они
# разъехались — поэтому noqa, а не удаление.
from kz.transform.damage import DAMAGE_PATTERNS, find_damage_keywords  # noqa: F401

FIELDS = ["ad_id", "fetched_at", "http_status", "is_archived",
          "customs_cleared", "drive", "steering", "color", "generation",
          "page_mileage_km", "page_condition", "has_vin",
          "damage_keywords", "seller_comment", "options_text",
          "kolesa_avg_price", "page_status_badge"]

# Средняя рыночная цена, которую kolesa считает САМА (в embedded JSON
# страницы, ключ "avgPrice"): сравнение внутри точного год+поколение+
# привод+двигатель+кузов+КПП+растаможка, с ИСКЛЮЧЕНИЕМ выбросов. Это
# независимый, более чистый эталон, чем наша грубая корзина «0-3 года» —
# используем как кросс-чек детектора (см. clean.py), НЕ как признак
# модели цены (это была бы утечка — модель училась бы копировать kolesa).
_AVGPRICE_RE = re.compile(r'"avgPrice"\s*:\s*(\d+)')


def extract_avg_price(html: str):
    m = _AVGPRICE_RE.search(html)
    return int(m.group(1)) if m else None


def extract_status_badge(html: str) -> str:
    """Статус-бейдж сайта из div.offer__parameters-mortgaged. Возвращает
    текст («Аварийная/Не на ходу», «Заложенная»...) либо маркер "-" =
    «проверено, бейджа нет» (чтобы отличать от NULL = «ещё не смотрели»
    и не перекачивать такие страницы повторно в бэкфилле)."""
    badge = BeautifulSoup(html, "html.parser").select_one(".offer__parameters-mortgaged")
    return badge.get_text(" ", strip=True) if badge else "-"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ARCHIVE_MARKERS = ["в&nbsp;архиве", "в\u00a0архиве", "в архиве", ">В архиве<"]


# Комментарий продавца НЕ виден в HTML-вёрстке (грузится JS-ом), но лежит
# в embedded JSON страницы под ключом descriptionText в unicode-escape виде
# (\u0440\u0443... вместо русских букв). Урок: прежде чем объявлять данные
# «динамическими», ищи их в исходнике страницы в других кодировках.
_DESC_RE = re.compile(r'"descriptionText"\s*:\s*"((?:[^"\\]|\\.)*)"')


def extract_seller_comment(html: str) -> str:
    m = _DESC_RE.search(html)
    if not m:
        return ""
    raw = m.group(1)
    try:
        # json.loads корректно разворачивает \uXXXX, \n, \" и т.д.
        import json
        text = json.loads(f'"{raw}"')
    except Exception:
        text = raw
    # В тексте остаётся HTML-разметка (<br/>, <p>) — вычищаем:
    # тексту для анализа нужны слова, а не вёрстка
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:2000]


def parse_ad_page(html: str) -> dict:
    """Чистая функция парсинга: HTML на входе, словарь на выходе.
    «Чистая» = без сети и файлов внутри — такие функции тестируются
    на сохранённых страницах без единого запроса к сайту."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}

    # Характеристики: пары dt/dd
    for dl in soup.select("dl"):
        dt, dd = dl.select_one("dt"), dl.select_one("dd")
        if not (dt and dd):
            continue
        key = DT_MAP.get(dt.get_text(" ", strip=True))
        if key:
            out[key] = dd.get_text(" ", strip=True)

    # Пробег со страницы → число (заполнит дыры VIP-паспортов!)
    if "page_mileage" in out:
        digits = re.sub(r"\D", "", out.pop("page_mileage"))
        out["page_mileage_km"] = int(digits) if digits else None

    # Блок опций (статичный) — источник слов о состоянии
    opts = soup.select_one(".offer__description .text")
    options_text = opts.get_text(" ", strip=True) if opts else ""
    out["options_text"] = options_text[:800]

    # Полный комментарий продавца — из embedded JSON
    seller_comment = extract_seller_comment(html)
    out["seller_comment"] = seller_comment

    # Поиск слов «убитости»: заголовок + опции + КОММЕНТАРИЙ ПРОДАВЦА.
    # С учётом отрицаний («вложения не требует» — НЕ повреждение),
    # см. damage.find_damage_keywords
    h1 = soup.select_one("h1")
    searchable = ((h1.get_text(" ", strip=True) if h1 else "")
                  + " " + options_text + " " + seller_comment)
    out["damage_keywords"] = "|".join(find_damage_keywords(searchable))

    out["is_archived"] = int(any(m in html for m in ARCHIVE_MARKERS))
    out["kolesa_avg_price"] = extract_avg_price(html)

    # Статус-бейдж от САМОГО сайта: div.offer__parameters-mortgaged —
    # «Аварийная/Не на ходу», «Заложенная», «Не растаможена» и т.п.
    # Это структурный сигнал, надёжнее вылавливания слов из текста
    # комментария (там отрицания и опечатки). Раньше не собирался вообще:
    # маппинг «Аварийная» искал в dt/dd, а бейдж — отдельный div.
    out["page_status_badge"] = extract_status_badge(html)
    return out


def load_done() -> set:
    if not Path(ENRICHED_CSV).exists():
        with open(ENRICHED_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        return set()
    with open(ENRICHED_CSV, encoding="utf-8") as f:
        return {r["ad_id"] for r in csv.DictReader(f)}


def pick_targets(done: set) -> list[str]:
    """Очередь на обогащение: подозрительные вперёд, дальше — новейшие.

    clean_data — disposable Postgres-таблица (см. clean.py), не CSV: там
    всегда самый свежий пасс.

    Два ключа сортировки, и второй важнее, чем кажется.

    is_suspicious — дорогой ресурс (запросы) тратим там, где выше ценность
      информации: обогащение заполняет поля, на которые смотрит exculpate(),
      снимая ложные подозрения. Таких строк единицы, стоят они копейки.

    scraped_at — СТРАНИЦЫ СМЕРТНЫ. У объявления, увиденного сегодня,
      страница гарантированно жива; у того, что ждёт в очереди третью
      неделю, она может уже отдавать 404 — и комментарий продавца с бейджем
      состояния потеряны навсегда. Ровно так мы потеряли 16 объявлений.

      Раньше второго ключа не было вовсе, и порядок внутри «неподозрительных»
      определялся тем, как база вернула строки. Новое объявление вставало в
      хвост очереди из тысяч старых. А приток новых при ежедневном сборе
      больше суточного бюджета обогащения, то есть хвост растёт быстрее, чем
      разбирается: свежая строка не дождалась бы никогда.

    Следствие, которое стоит понимать: исторический бэклог теперь будет
    убывать медленно или не убывать вовсе. Это осознанный размен — страницы
    старых объявлений всё равно умирают, и гнаться за ними дороже, чем не
    терять новые.

    ТРЕТИЙ КЛЮЧ, добавлен 29 августа: дешёвый сегмент вперёд.

    Вся ошибка модели сидит в машинах до 5 млн — 29,16% против 15,5% у
    остальных, при 40% выборки. Замер на полностью обогащённой подвыборке
    показал, что текст продавца и комплектация дают там −3,4 п.п.
    (FINDINGS §17). Дорогой сегмент уже укладывается в цель, и обогащать
    его ради MAPE незачем.

    ПОЧЕМУ НЕ ТОЛЬКО ДЕШЁВЫЕ. Если заполнить признаки у дешёвых и оставить
    пустыми у дорогих, «есть комплектация» станет меткой «дешёвая машина» —
    и модель выучит нашу очередь обогащения вместо свойств автомобиля. Это
    та же ловушка, что с длиной текста (FINDINGS §17, поправка). Поэтому
    доля FRESH_RESERVE каждого прогона уходит просто самым свежим, вне
    зависимости от цены: и страницы не мрут, и пропущенность не становится
    идеально совпадающей с сегментом.

    Полностью развязать это отбором нельзя. Правильное лекарство — при
    добавлении признаков в модель обучать НА ДЕШЁВОМ СЕГМЕНТЕ отдельно, а
    не подмешивать полупустые колонки в общую таблицу.
    """
    df = pd.read_sql(
        "SELECT c.ad_id, c.is_suspicious, c.price_tenge, r.scraped_at "
        "FROM clean_data c JOIN raw_ads r ON r.ad_id = c.ad_id",
        get_engine(), dtype={"ad_id": str})
    df = df[~df["ad_id"].isin(done)]
    if df.empty:
        return []
    df = df.sort_values("scraped_at", ascending=False)
    df["cheap"] = (df.price_tenge.fillna(0) > 0) & (df.price_tenge < CHEAP_EDGE)

    out: list[str] = []

    def take(rows, n):
        """Дописать до n идентификаторов, не повторяя уже взятые."""
        for a in rows["ad_id"]:
            if len(out) >= n or len(out) >= MAX_PER_RUN:
                break
            if a not in out:
                out.append(a)

    # 1. Подозрительные — всегда первыми, но не больше половины прогона:
    #    иначе очередь из семидесяти подозрительных заняла бы четыре дня и
    #    дешёвый сегмент не начался бы вовсе.
    take(df[df.is_suspicious == 1], round(MAX_PER_RUN * SUSPICIOUS_SHARE))
    # 2. Резерв на самые свежие вне зависимости от цены — см. докстроку.
    take(df, len(out) + max(1, round(MAX_PER_RUN * FRESH_RESERVE)))
    # 3. Дешёвый сегмент, внутри — свежие вперёд.
    take(df[df.cheap], MAX_PER_RUN)
    # 4. Остаток, если дешёвые кончились.
    take(df, MAX_PER_RUN)
    return out


def main():
    done = load_done()
    targets = pick_targets(done)
    log.info(f"К обогащению: {len(targets)} (свежие {int(FRESH_RESERVE*100)}%, "
             f"дальше подозрительные и дешёвые; уже готово: {len(done)})")

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
                log.error("Стоп: сбои подряд, продолжим завтра.")
                sys.exit(1)
            time.sleep(30)
            continue

        if resp.status_code == 429:
            # Rate limit: сайт просит притормозить — тормозим НАДОЛГО,
            # это не ошибка, а инструкция.
            log.warning("429: пауза 120с")
            time.sleep(120)
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                sys.exit(1)
            continue
        fails = 0

        row = {"ad_id": ad_id,
               "fetched_at": datetime.now().isoformat(timespec="seconds"),
               "http_status": resp.status_code}
        if resp.status_code == 200:
            row.update(parse_ad_page(resp.text))

        with open(ENRICHED_CSV, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore") \
               .writerow(row)

        # Двойная запись (пилот миграции на Postgres, см. план): CSV
        # остаётся источником истины, сбой записи в БД не должен ронять
        # прогон — обогащение стоит дорого (запросы), терять его нельзя.
        try:
            # "" → NULL: в CSV пустое значение — это "", но в БД пустая
            # строка и NULL — разные вещи, и смешивать их нельзя (бэкфилл
            # из CSV даёт NULL, живая запись давала "" — в таблице был
            # зоопарк из двух видов «пусто»). Конвенция: пусто = NULL.
            pg_row = {f: (row.get(f) if row.get(f) != "" else None)
                      for f in FIELDS}
            upsert("enriched", [pg_row], ["ad_id"])
        except Exception as e:
            log.warning(f"Postgres upsert не удался для {ad_id}: {e}")

        if i % 20 == 0:
            log.info(f"  {i}/{len(targets)}")
        pacing.polite_sleep(i, DELAY_RANGE, log)

    log.info(f"Готово → {ENRICHED_CSV}")


if __name__ == "__main__":
    main()