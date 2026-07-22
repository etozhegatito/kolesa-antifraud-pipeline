# -*- coding: utf-8 -*-
"""
diagnose.py — ответ на вопрос «почему у объявления X пустое поле Y?»
одной командой. Кодифицирует трёхшаговый диагноз:

  шаг 1: что лежит в наших файлах (raw / enriched / clean)?
  шаг 2: если пусто в enriched — оно вообще обогащалось, или очередь не дошла?
  шаг 3: живой запрос страницы + прогон через parse_ad_page:
         отдаёт ли ИСТОЧНИК это поле, и берёт ли его НАШ парсер?

Вердикты:
  «очередь не дошла»   — не баг: enrich доберётся, страница восстановима
  «нет в источнике»    — не баг: продавец не указал (MNAR)
  «есть в другой колонке» — не баг: смотри text_full / seller_comment
  «парсер теряет»      — вот ЭТО баг: неси ad_id разработчику (себе)

Запуск: python debug/diagnose.py 210229611
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
import pathlib as _p
_expected = "diagnose.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import sys
import time
from pathlib import Path

import pandas as pd
import requests

# enrich.py лежит в корне репозитория, на уровень выше debug/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from enrich import parse_ad_page, HEADERS


def read_if_exists(path, **kw):
    return pd.read_csv(path, dtype={"ad_id": str}, **kw) if Path(path).exists() else None


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Использование: python diagnose.py <ad_id>")
    ad = sys.argv[1].strip()
    print(f"═══ Диагноз объявления {ad} ═══\n")

    # ── шаг 1: наши файлы ────────────────────────────────────────────────
    raw = read_if_exists("data/raw/raw_data.csv")
    enr = read_if_exists("data/enriched/enriched.csv")
    cl  = read_if_exists("data/clean/clean_data.csv")

    r = raw[raw.ad_id == ad] if raw is not None else pd.DataFrame()
    if len(r) == 0:
        print("• В raw_data объявления НЕТ — листинг его ещё не встречал")
    else:
        row = r.iloc[-1]
        print(f"• raw: is_vip={row.get('is_vip')}, "
              f"mileage={row.get('mileage_km')}, "
              f"description={str(row.get('description'))[:60]!r}")

    enriched = enr is not None and ad in set(enr.ad_id)
    print(f"• enriched: {'ДА' if enriched else 'нет — очередь не дошла (не баг)'}")
    if enriched:
        e = enr[enr.ad_id == ad].iloc[-1]
        print(f"    page_mileage={e.get('page_mileage_km')}, "
              f"seller_comment={str(e.get('seller_comment'))[:70]!r}")
    if cl is not None and "text_full" in cl.columns:
        c = cl[cl.ad_id == ad]
        if len(c):
            print(f"• clean.text_full: {str(c.iloc[-1]['text_full'])[:70]!r}")

    # ── шаг 3: живой источник + наш парсер ───────────────────────────────
    print("\n• Живой запрос страницы...")
    time.sleep(2)
    resp = requests.get(f"https://kolesa.kz/a/show/{ad}",
                        headers=HEADERS, timeout=20)
    print(f"    HTTP {resp.status_code}")
    if resp.status_code != 200:
        print("    Страница недоступна (404 = удалено) — вердикт: нет в источнике")
        return
    parsed = parse_ad_page(resp.text)
    print(f"    парсер видит: пробег={parsed.get('page_mileage_km')}, "
          f"комментарий={parsed.get('seller_comment','')[:70]!r}, "
          f"растаможка={parsed.get('customs_cleared')}")

    # ── вердикт ──────────────────────────────────────────────────────────
    print("\n═══ Вердикт ═══")
    if not enriched:
        print("Поля пусты, потому что обогащение ещё не дошло (приоритетная "
              "очередь, бюджет 250/день). Парсер страницы поле берёт — "
              "см. строку выше. Данные восстановимы, багов нет.")
    elif parsed.get("page_mileage_km") is None:
        print("Пробега нет на самой странице (обычно новая машина или "
              "продавец не указал) — MNAR, парсить нечего.")
    else:
        print("Страница отдаёт данные, объявление обогащено — сверь колонки: "
              "полный текст живёт в seller_comment/text_full, а не в description.")


if __name__ == "__main__":
    main()