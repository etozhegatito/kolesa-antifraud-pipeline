# -*- coding: utf-8 -*-
"""
pipeline_status.py — «пульт»: сколько чего собрано, обогащено и сколько
осталось. Полностью офлайн (только чтение Postgres + один локальный CSV),
ни одного запроса к сайту — можно гонять сколько угодно.

Отвечает на вопросы:
  - сколько объявлений ждут обогащения (и сколько из них подозрительных);
  - сколько фото ждут хэширования;
  - при текущих 24-часовых бюджетах — за сколько дней рассосётся бэклог;
  - покрытие текстом (полный комментарий / огрызок листинга / пусто);
  - статусы жизненного цикла (active/archived/deleted/не проверялось);
  - сколько ручных вердиктов уже размечено.

Запуск: python -m kz.ops.pipeline_status          (отчёт + вопрос «запустить обогащение?»)
        python -m kz.ops.pipeline_status --run    (отчёт + запуск без вопроса)

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


import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from kz.core.db import get_engine
# Пороги «пробел статуса» берём из catch_up (а он синхронен с check_status —
# см. test_catch_up_status_thresholds_match_check_status): единый источник,
# чтобы пульт, catch_up и сам джоб не разошлись в определении бэклога.
from kz.ops.catch_up import STATUS_STALE_DAYS, STATUS_RECHECK_DAYS, DAILY_BUDGET

# ETA считаем по 24-ЧАСОВОМУ бюджету хоста (потолок при catch_up --until-done),
# а не по размеру одной порции — берём из catch_up, чтобы не разъезжалось при
# смене лимитов. Это оценка «при полном дневном доборе на хост».
ENRICH_PER_DAY = DAILY_BUDGET["kolesa"]
PHOTOS_PER_DAY = DAILY_BUDGET["cdn"]

LABELS_CSV = "data/manual_labels.csv"
PARSER_STATUS_JSON = "logs/parser_last_run.json"

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
    # Свежесть — первое, что нужно увидеть, вернувшись к проекту через две
    # недели: остальные числа читаются по-разному в зависимости от того,
    # когда данные собирали последний раз.
    from kz.core import freshness as fr
    _state = fr.measure()
    fr.report(_state)
    for _w in fr.stale_warnings(_state):
        print(f"  \u26a0 {_w}")
    print()
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
    clean_ids = set(clean["ad_id"])
    matched_enriched = enriched[enriched["ad_id"].isin(clean_ids)].copy()
    matched_ids = set(matched_enriched["ad_id"])
    usable_enriched = matched_enriched[matched_enriched["http_status"] == 200]
    usable_ids = set(usable_enriched["ad_id"])
    dead_enriched = matched_enriched[matched_enriched["http_status"] != 200]
    orphan_enriched = enriched[~enriched["ad_id"].isin(clean_ids)]

    # ── Обогащение страниц ───────────────────────────────────────────────
    # 404/архив уже проверен и повторно сеть не тратим, но в полезное покрытие
    # его не записываем. Поэтому backlog и прогресс имеют разные знаменатели.
    pending_mask = ~clean["ad_id"].isin(matched_ids)
    pending = int(pending_mask.sum())
    pending_susp = int((pending_mask & (clean["is_suspicious"] == 1)).sum())
    enr_ok = len(usable_ids)
    enr_fail = len(dead_enriched)

    print(LINE)
    print(f"ОБЪЯВЛЕНИЙ ВСЕГО: {total}   подозрительных: "
          f"{int(clean['is_suspicious'].sum())}")
    print(LINE)

    print("\n► Обогащение страниц (enrich.py)")
    print(f"  полезно: {bar(enr_ok, total)}   {enr_ok}/{total}")
    print(f"  строк в enriched: {len(enriched)}; совпало с raw/clean: "
          f"{len(matched_enriched)}")
    print(f"  осталось: {pending}  → {eta_days(pending, ENRICH_PER_DAY)}")
    if pending_susp:
        print(f"  ⚠ среди ожидающих ПОДОЗРИТЕЛЬНЫХ: {pending_susp} "
              f"(они пойдут первыми в следующем прогоне)")
    else:
        print("  ✓ все подозрительные уже обогащены")
    if enr_fail:
        print(f"  страниц, умерших до обогащения (404/архив и т.п.): {enr_fail}")
    if len(orphan_enriched):
        print(f"  ⚠ orphan-строк без объявления в raw/clean: {len(orphan_enriched)} "
              "(не входят в покрытие)")

    # ── Здоровье последнего листингового прогона ────────────────────────
    print("\n► Последний запуск parser.py")
    try:
        parser_run = json.loads(Path(PARSER_STATUS_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("  структурного статуса пока нет (появится после следующего запуска)")
    else:
        run_status = parser_run.get("status", "unknown")
        print(f"  статус: {run_status}; "
              f"начало: {parser_run.get('started_at', '?')}; "
              f"конец: {parser_run.get('finished_at') or 'ещё идёт'}")
        truncated = parser_run.get("freshness_truncated_segments") or []
        if truncated:
            print("  ⚠ лимит страниц сработал, пока unseen объявления ещё шли: "
                  + ", ".join(truncated))
        elif run_status == "success":
            print("  ✓ на границе лимита unseen-объявлений не зафиксировано")
        elif parser_run.get("message"):
            print(f"  ⚠ запуск не завершён успешно: {parser_run['message']}")

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
    # бэклог проверки: пропал из листинга >=STALE_DAYS, НЕ терминальный И
    # давно (>=RECHECK_DAYS) не проверялся напрямую — та же логика, что
    # needs_status_check() в check_status.py и compute_gaps() в catch_up.py
    last_seen = pd.read_sql(
        "SELECT ad_id, MAX(seen_date) AS seen FROM sightings GROUP BY ad_id",
        engine, dtype={"ad_id": str})
    statuses = pd.read_sql("SELECT ad_id, status, checked_at FROM ad_status",
                            engine, dtype={"ad_id": str})
    last_seen = last_seen.merge(statuses, on="ad_id", how="left")
    today = pd.Timestamp.today().normalize()
    days_gone = (today - pd.to_datetime(last_seen["seen"])).dt.days
    checked_days = (today - pd.to_datetime(last_seen["checked_at"])).dt.days
    terminal = last_seen["status"].isin(["archived", "deleted"])
    recently_checked = checked_days < STATUS_RECHECK_DAYS          # NaN<7 → False
    st_pending = int(((~terminal) & (days_gone >= STATUS_STALE_DAYS)
                      & (~recently_checked)).sum())
    print(f"  ждут проверки статуса: {st_pending}  → "
          f"{eta_days(st_pending, DAILY_BUDGET['kolesa'])} (потолок kolesa/24ч)")

    # ── Ручная разметка ──────────────────────────────────────────────────
    print("\n► Ручная разметка (data/manual_labels.csv)")
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        verdict = lab["verdict"].astype("string").str.strip().str.lower()
        latest = lab.assign(_verdict=verdict).drop_duplicates("ad_id", keep="last")
        n_valid = int(latest["_verdict"].isin(["fraud", "legit"]).sum())
        n_fraud = int((latest["_verdict"] == "fraud").sum())
        print(f"  валидных вердиктов: {n_valid} "
              f"(fraud: {n_fraud}, legit: {n_valid - n_fraud})")
        print(f"  строк журнала: {len(lab)}; "
              f"пустых/unknown: {int((~verdict.isin(['fraud', 'legit'])).sum())}")
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
from kz.ops import run_all as _ra


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
        print("\nЗапустить обогащение: python -m kz.ops.pipeline_status --run")
        return

    print(f"\nВ очереди: {pending} страниц (порция ~120) "
          f"и {ph_pending} фото (порция ~300).")
    print("Это СЕТЕВЫЕ запросы к kolesa.kz — не гоняй много раз подряд,")
    print("джобы резюмируемые: прерваться и продолжить после освобождения квоты безопасно.")
    ans = input("Запустить джобы обогащения сейчас? [y/N] ").strip().lower()
    if ans in ("y", "yes", "д", "да"):
        run_enrichment_jobs()
    else:
        print("Ок, не запускаю. Когда решишь: python -m kz.ops.pipeline_status --run")


if __name__ == "__main__":
    main()
