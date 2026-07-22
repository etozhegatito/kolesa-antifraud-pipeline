# -*- coding: utf-8 -*-
"""
pipeline_status.py — «пульт»: сколько чего собрано, обогащено и сколько
осталось. Полностью офлайн (только чтение Postgres + один локальный CSV),
ни одного запроса к сайту — можно гонять сколько угодно.

Отвечает на вопросы:
  - сколько объявлений ждут обогащения (и сколько из них подозрительных);
  - сколько фото ждут хэширования;
  - при текущих дневных бюджетах — за сколько дней рассосётся бэклог;
  - покрытие текстом (полный комментарий / огрызок листинга / пусто);
  - статусы жизненного цикла (active/archived/deleted/не проверялось);
  - сколько ручных вердиктов уже размечено.

Запуск: python pipeline_status.py          (отчёт + вопрос «запустить обогащение?»)
        python pipeline_status.py --run    (отчёт + запуск без вопроса)

Если бэклог не пуст и скрипт запущен в интерактивном терминале, в конце
спрашивает, запустить ли джобы обогащения (enrich → photo_dedup →
clean пасс 2 → explore). В неинтерактивном режиме (пайп/cron) вопрос
не задаётся — иначе input() повесил бы процесс навсегда.
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
import pathlib as _p
_expected = "pipeline_status.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import math
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine

# Дневные бюджеты — должны совпадать с константами самих джобов
ENRICH_PER_DAY = 120    # enrich.py MAX_PER_RUN
PHOTOS_PER_DAY = 300    # photo_dedup.py MAX_PER_RUN

LABELS_CSV = "data/manual_labels.csv"

LINE = "─" * 64


def eta_days(pending: int, per_day: int) -> str:
    if pending <= 0:
        return "готово"
    return f"~{math.ceil(pending / per_day)} дн. при {per_day}/день"


def bar(done: int, total: int, width: int = 24) -> str:
    """Псевдографический прогресс-бар: наглядно, без зависимостей."""
    if total <= 0:
        return "─" * width + "   0%"
    frac = done / total
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled) + f" {frac:5.1%}"


def main():
    engine = get_engine()

    with engine.begin() as conn:
        has_clean = conn.execute(
            text("SELECT to_regclass('public.clean_data')")).scalar()
    if not has_clean:
        raise SystemExit("clean_data ещё нет — сначала прогони clean.py "
                         "(или run_all.py --fast).")

    clean = pd.read_sql(
        "SELECT ad_id, is_suspicious, seller_comment, description, status "
        "FROM clean_data", engine, dtype={"ad_id": str})
    enriched = pd.read_sql(
        "SELECT ad_id, http_status FROM enriched", engine, dtype={"ad_id": str})
    photos = pd.read_sql("SELECT ad_id, url FROM photos", engine,
                         dtype={"ad_id": str})
    hashes = pd.read_sql("SELECT ad_id, url, phash FROM photo_hashes", engine,
                         dtype={"ad_id": str})

    total = len(clean)
    done_ids = set(enriched["ad_id"])

    # ── Обогащение страниц ───────────────────────────────────────────────
    pending_mask = ~clean["ad_id"].isin(done_ids)
    pending = int(pending_mask.sum())
    pending_susp = int((pending_mask & (clean["is_suspicious"] == 1)).sum())
    enr_ok = int((enriched["http_status"] == 200).sum())
    enr_fail = len(enriched) - enr_ok

    print(LINE)
    print(f"ОБЪЯВЛЕНИЙ ВСЕГО: {total}   подозрительных: "
          f"{int(clean['is_suspicious'].sum())}")
    print(LINE)

    print("\n► Обогащение страниц (enrich.py)")
    print(f"  {bar(len(enriched), total)}   {len(enriched)}/{total}")
    print(f"  осталось: {pending}  → {eta_days(pending, ENRICH_PER_DAY)}")
    if pending_susp:
        print(f"  ⚠ среди ожидающих ПОДОЗРИТЕЛЬНЫХ: {pending_susp} "
              f"(они пойдут первыми в следующем прогоне)")
    else:
        print("  ✓ все подозрительные уже обогащены")
    if enr_fail:
        print(f"  страниц, умерших до обогащения (404/архив и т.п.): {enr_fail}")

    # ── Фото-хэши ────────────────────────────────────────────────────────
    # Заглушки «нет фото» (protocol-relative //...) не хэшируются намеренно —
    # исключаем их из знаменателя, иначе прогресс никогда не дойдёт до 100%.
    hashable = photos[photos["url"].fillna("").str.startswith("http")]
    hashed_urls = set(hashes["url"])
    ph_pending = int((~hashable["url"].isin(hashed_urls)).sum())
    ph_done = len(hashable) - ph_pending
    ph_bad = int((hashes["phash"].isna() | (hashes["phash"] == "")).sum())

    print("\n► Фото-хэши (photo_dedup.py)")
    print(f"  {bar(ph_done, len(hashable))}   {ph_done}/{len(hashable)}"
          f"   (+{len(photos) - len(hashable)} заглушек «нет фото» — не считаем)")
    print(f"  осталось: {ph_pending}  → {eta_days(ph_pending, PHOTOS_PER_DAY)}")
    if ph_bad:
        print(f"  скачано, но не разобрано (битые/таймауты): {ph_bad}")

    # ── Текстовое покрытие ───────────────────────────────────────────────
    sc = clean["seller_comment"].fillna("").astype(str).str.len() > 0
    desc = clean["description"].fillna("").astype(str).str.len() > 0
    print("\n► Текст (text_full в clean_data)")
    print(f"  полный комментарий продавца : {int(sc.sum())}")
    print(f"  только огрызок из листинга  : {int((~sc & desc).sum())}")
    print(f"  текста нет вообще           : {int((~sc & ~desc).sum())}")

    # ── Жизненный цикл ───────────────────────────────────────────────────
    st = clean["status"].fillna("active").value_counts()
    print("\n► Статусы (check_status.py)")
    for name, cnt in st.items():
        print(f"  {name:<10} {cnt}")
    # бэклог проверки: не видели в листинге >=2 дней и не в терминальном
    # статусе — та же логика отбора кандидатов, что в самом check_status.py
    last_seen = pd.read_sql(
        "SELECT ad_id, MAX(seen_date) AS seen FROM sightings GROUP BY ad_id",
        engine, dtype={"ad_id": str})
    statuses = pd.read_sql("SELECT ad_id, status FROM ad_status", engine,
                            dtype={"ad_id": str})
    last_seen = last_seen.merge(statuses, on="ad_id", how="left")
    days_gone = (pd.Timestamp.today().normalize()
                 - pd.to_datetime(last_seen["seen"])).dt.days
    st_pending = int(((days_gone >= 2)
                      & ~last_seen["status"].isin(["archived", "deleted"])).sum())
    print(f"  ждут проверки статуса: {st_pending}  → "
          f"{eta_days(st_pending, 150)} (лимит check_status.py)")

    # ── Ручная разметка ──────────────────────────────────────────────────
    print("\n► Ручная разметка (data/manual_labels.csv)")
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        print(f"  вердиктов: {len(lab)}")
    else:
        print("  файла ещё нет — 0 вердиктов "
              "(очередь: data/eda/labeling_queue.csv)")
    print(LINE)

    maybe_run_enrichment(pending, ph_pending)


# ─── Запуск джобов обогащения прямо отсюда ───────────────────────────────────
# Переиспользуем шаги и раннеры run_all.py (run_step с fail-fast,
# run_parallel для пары «разные хосты») — та же логика, один источник.
# Порядок: check_status добирает свою порцию статусов (тот же хост
# kolesa.kz, поэтому СТРОГО до enrich, не параллельно!), затем
# enrich ∥ photo_dedup (kolesa.kz ∥ CDN), затем clean пасс 2 + explore
# вливают результат в clean_data и отчёт.
import run_all as _ra


def run_enrichment_jobs():
    t0 = time.time()
    _ra.run_step(_ra.STEP_STATUS)
    _ra.run_parallel(_ra.STEP_ENRICH, _ra.STEP_PHOTOS)
    _ra.run_step(_ra.STEP_CLEAN)
    _ra.run_step(_ra.STEP_EXPLORE)
    print(f"\n✔ Обогащение завершено за {(time.time() - t0) / 60:.1f} мин")


def maybe_run_enrichment(pending: int, ph_pending: int):
    if pending <= 0 and ph_pending <= 0:
        print("\nБэклог пуст — обогащать нечего.")
        return

    if "--run" in sys.argv:
        run_enrichment_jobs()
        return

    if not sys.stdin.isatty():
        # пайп/cron: вопрос задавать некому, просто подсказываем
        print("\nЗапустить обогащение: python pipeline_status.py --run")
        return

    print(f"\nВ очереди: {pending} страниц (порция ~120) "
          f"и {ph_pending} фото (порция ~300).")
    print("Это СЕТЕВЫЕ запросы к kolesa.kz — не гоняй много раз подряд,")
    print("джобы резюмируемые: прерваться и продолжить завтра безопасно.")
    ans = input("Запустить джобы обогащения сейчас? [y/N] ").strip().lower()
    if ans in ("y", "yes", "д", "да"):
        run_enrichment_jobs()
    else:
        print("Ок, не запускаю. Когда решишь: python pipeline_status.py --run")


if __name__ == "__main__":
    main()
