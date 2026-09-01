# -*- coding: utf-8 -*-
"""
catch_up.py — «умный догоняльщик»: сам смотрит, каких сетевых данных не
хватает, и добивает пробелы БЕЗОПАСНО.

Зачем отдельно от run_all.py: run_all — это полный ежедневный цикл (сбор
листинга + всё остальное). А catch_up НЕ ходит в листинг вообще — он
только дозаполняет уже известные пробелы по РАЗ собранным объявлениям
(статусы, обогащение, avgPrice/бейдж, фото-хэши). Удобно гонять между
полными прогонами, когда бэклоги накопились.

ГЛАВНОЕ ПРО БЕЗОПАСНОСТЬ (почему не просто parallel всё):
  check_status, enrich, backfill_avgprice — все стучатся в kolesa.kz
  (ОДИН хост). Их нельзя запускать одновременно — удвоится частота
  запросов с одного IP → бан. Поэтому они идут СТРОГО ПОСЛЕДОВАТЕЛЬНО.
  photo_dedup — единственный по другому хосту (CDN kcdn.kz), у него
  СВОЙ бюджет.
  Между джобами — детект 429 (сайт просит притормозить): если новый 429
  появился в логах, обрываем оставшиеся сетевые джобы (circuit breaker
  уровня оркестратора, поверх внутренних предохранителей самих джобов).

РИТМ ЗАПРОСОВ (pacing.py): паузы не плоские — базовая 4-8с, изредка
  затяжная «отвлёкся», каждые 15 запросов длинный перерыв 30-90с. Это
  politeness (запросов в час МЕНЬШЕ, чем было), а не маскировка бота.

СКОЛЬЗЯЩИЙ БЮДЖЕТ ЗАПРОСОВ НА ХОСТ (главный анти-бан-рычаг):
  Бан ловится по ОБЪЁМУ запросов с одного IP за короткое окно, а не по паузе
  между ними (паузы 4-8с уже внутри джобов). Поэтому поверх всего —
  общий на хост потолок числа запросов за последние 24 часа (DAILY_BUDGET). Он
  ОБЩИЙ для трёх kolesa-джобов (это один IP!), у CDN — отдельный.
  Счётчик живёт в logs/.catch_up_budget.json. Полночь его НЕ обнуляет:
  каждое списание отпадает ровно через 24 часа. Как только квота выбрана —
  джобы встают до освобождения окна. Так НИ ОДИН запуск (даже случайный
  ручной, даже --until-done) не пробьёт допустимый объём.

Сентинелы (важно для подсчёта «пробелов»): avgPrice = -1 и бейдж = "-"
означают «проверено, значения у объявления НЕТ» — это НЕ пробел, повторно
не качаем. Пробел = только NULL («ещё не смотрели»).

Джоб запускается, ТОЛЬКО если у него реально есть пробел (нечего качать —
не ходим в сеть зря). В конце — офлайн-пересборка clean+отчёт, чтобы
свежие данные попали во флаги.

Запуск: python -m kz.ops.catch_up             (отчёт + вопрос, запускать ли)
        python -m kz.ops.catch_up --run        (одна порция на джоб, без вопроса)
        python -m kz.ops.catch_up --run --until-done
                                        (использовать всю квоту rolling-окна:
                                         крутит порциями, пока не выбран
                                         бюджет хоста за 24 часа / не закрыты
                                         пробелы / не пришёл 429; резюмируемо
                                         после освобождения окна)
        python -m kz.ops.catch_up --run --values
                                        (приоритетно ТОЛЬКО ценные-для-оправдания
                                         поля: enrich + backfill = avgPrice/бейдж/
                                         цвет/damage/растаможка. Статусы и фото
                                         пропускает. Быстро чистит подозрительных
                                         под разметку; сочетается с --until-done)
        python -m kz.ops.catch_up --run --backfill
                                        (ещё уже: ТОЛЬКО добор avgPrice+бейджа у
                                         УЖЕ обогащённых строк — целится в незапол-
                                         ненные, заполненные пропускает, новые
                                         объявления не обогащает; тоже с --until-done)
        python -m kz.ops.catch_up --run --backfill --budget 300
                                        (СКОЛЬКО СПАРСИТЬ: потолок запросов к
                                         kolesa за 24 часа. Зоны риска печатаются
                                         при запуске без --run; коротко:
                                         ≤100 спокойно, ≤200 безопасно (дефолт),
                                         ≤270 риск, >270 высокий риск — на ~270
                                         домашний IP уже ложился 2026-07-23.
                                         Альтернатива: env KOLESA_BUDGET=300)
"""

import pathlib as _p
_expected = "catch_up.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta

import fcntl

import pandas as pd

from kz.core import pacing
from kz.core.db import get_engine

LINE = "─" * 64

# ─── Суточный бюджет запросов на ХОСТ (анти-бан, см. докстринг) ──────────────
# ОБЩИЙ для всех kolesa-джобов (один IP!); у CDN — свой.
# 2026-07-23: домашний IP словил ВРЕМЕННЫЙ бан kolesa (~270 запросов за день с
# одного IP: catch_up + run_all + ручной браузинг). Снизили kolesa 400→200 с
# запасом. Parser с 2026-08-29 списывает top-level переходы в тот же файл.
# Ручной браузинг счётчик по-прежнему не видит, поэтому его нельзя складывать
# с полным сетевым прогоном. При budget 200 обычный --run делает ~одну порцию
# (по 20 на джоб); точечный добор avgPrice/бейджа — --backfill (порции по 20).
# Дефолт kolesa=200 — БЕЗОПАСНЫЙ потолок для домашнего IP (его и банили).
# Бюджет НАСТРАИВАЕМЫЙ, приоритет: --budget N  >  env KOLESA_BUDGET  >  200.
# Зоны риска и рекомендации — в RISK_ZONES/risk_zone() ниже.
# Реактивная защита (детект 429 + внутренние предохранители джобов) работает
# независимо от этого числа: даже с огромным бюджетом цепочка оборвётся, если
# сайт начнёт лимитировать.
DEFAULT_KOLESA_BUDGET = 200
# CDN раздаёт статические файлы и к блокировке 2026-07-23 отношения не имел —
# та случилась на сайте объявлений. Потолок здесь консервативный по привычке,
# и он тоже настраивается: CDN_BUDGET.
DEFAULT_CDN_BUDGET = 1200
DAILY_BUDGET = {"kolesa": int(os.environ.get("KOLESA_BUDGET",
                                             DEFAULT_KOLESA_BUDGET)),
                "cdn": int(os.environ.get("CDN_BUDGET", DEFAULT_CDN_BUDGET))}
BUDGET_FILE  = "logs/.catch_up_budget.json"

# ─── Зоны риска по числу запросов к kolesa за 24 часа с ОДНОГО IP ────────────
# Границы — не из статей, а из собственного опыта (правило проекта: калибруй
# на своих данных). Единственный жёсткий факт: 2026-07-23 домашний IP словил
# временный бан на ~270 запросах за сутки. Всё, что ниже 200, гонялось
# многократно без последствий. Отсюда зоны:
#   (порог_включительно, метка, пояснение)
RISK_ZONES = [
    (100,   "спокойно",
     "многократно проверено на этом проекте, последствий не было"),
    (200,   "безопасно",
     "рабочая зона (дефолт 200); банов на ней не наблюдали"),
    (270,   "риск",
     "подходит к зафиксированному бану: 2026-07-23 IP лёг на ~270"),
    (10**9, "высокий риск",
     "выше уже случившегося бана — только с динамического IP и осознанно"),
]


def risk_zone(n: int):
    """(метка, пояснение) для объёма n запросов к kolesa за 24 часа."""
    for limit, label, note in RISK_ZONES:
        if n <= limit:
            return label, note
    return RISK_ZONES[-1][1], RISK_ZONES[-1][2]


def eta_minutes(n: int, lo: float = 4.0, hi: float = 8.0) -> float:
    """Честная оценка времени на n запросов: пауза по факту (с хвостом и
    перерывами из pacing) + ~3с на сам запрос. Чтобы «сколько спарсить»
    сразу переводилось в «сколько это займёт»."""
    return n * (pacing.mean_pause(lo, hi) + 3.0) / 60


def parse_budget(argv) -> int | None:
    """--budget N | --budget=N → N (None, если флага нет). Валидирует."""
    for i, a in enumerate(argv):
        raw = None
        if a == "--budget" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif a.startswith("--budget="):
            raw = a.split("=", 1)[1]
        if raw is not None:
            try:
                n = int(raw)
            except ValueError:
                raise SystemExit(
                    f"--budget: нужно целое число, получено {raw!r}") from None
            if n < 1:
                raise SystemExit("--budget: должно быть >= 1")
            return n
    return None


def print_risk_help(current: int):
    """Табличка «сколько запросов = насколько рискованно» + как задать."""
    print("\nСколько запросов к kolesa за скользящие 24 часа (--budget N):")
    prev = 0
    for limit, label, note in RISK_ZONES:
        rng = f"{prev+1}–{limit}" if limit < 10**8 else f"{prev+1}+"
        print(f"  {rng:<10} {label:<14} — {note}")
        prev = limit
    label, _ = risk_zone(current)
    print(f"\nСейчас потолок: {current} ({label}); "
          f"на весь объём ≈{eta_minutes(current):.0f} мин.")
    print("Задать: python -m kz.ops.catch_up --run --backfill --budget 300")
    print("Parser делит этот счётчик с catch_up. Ручной браузинг kolesa "
          "по-прежнему не считается и не должен идти рядом с полным сбором.")

# Верхняя оценка запросов за ОДНУ порцию джоба (= его MAX_PER_RUN). Держим
# копией здесь, чтобы не импортировать джобы (у них при импорте открываются
# лог-файлы и настраивается root logger). Синхронность с источником стережёт
# test_catch_up_chunk_sizes_match_jobs.
# photo=300: это CDN (kcdn.kz), другой хост, бан был на kolesa — не режем.
CHUNK_MAX = {"status": 20, "enrich": 20, "backfill": 20, "photo": 300}

# Пороги «нужен ли пере-запрос статуса» ДОЛЖНЫ совпадать с check_status.py,
# иначе счётчик пробелов разошёлся бы с реальной выборкой джоба. Синхронность
# стережёт test_catch_up_status_thresholds_match_check_status.
STATUS_STALE_DAYS   = 2     # пропал из листинга дольше → статус под сомнением
STATUS_RECHECK_DAYS = 7     # проверяли напрямую позже → не считаем пробелом


# `events` — источник истины для скользящего окна 24 часа. `days` остаётся
# человекочитаемой календарной историей для аудита. До schema_version=2 файл
# хранил только `days`; миграция консервативно переносит расход сегодня и
# вчера, чтобы обновление кода само не подарило вторую квоту.
BUDGET_SCHEMA_VERSION = 2
BUDGET_WINDOW_HOURS = 24
BUDGET_EVENT_KEEP_HOURS = 48
BUDGET_KEEP_DAYS = 7


class BudgetStateError(RuntimeError):
    """Budget-файл существует, но его нельзя безопасно интерпретировать."""


def _now() -> datetime:
    """Текущее локальное время с timezone; вынесено для точных тестов."""
    return datetime.now().astimezone()


def _read_state() -> dict:
    """Нормализованный бюджет; читает оба прежних формата без потери квоты."""
    try:
        d = json.loads(_p.Path(BUDGET_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": BUDGET_SCHEMA_VERSION,
                "days": {}, "events": []}
    except (OSError, ValueError) as e:
        raise BudgetStateError(
            f"budget-файл {BUDGET_FILE} повреждён/не читается; "
            "сеть заблокирована, чтобы не обнулить квоту: {e}"
        ) from e
    if not isinstance(d, dict):
        raise BudgetStateError(
            f"budget-файл {BUDGET_FILE} должен содержать JSON-объект; "
            "сеть заблокирована"
        )
    if isinstance(d.get("days"), dict):
        days = d["days"]
    elif d.get("date"):                    # формат до 2026-07-30
        days = {d["date"]: {"kolesa": int(d.get("kolesa", 0)),
                             "cdn": int(d.get("cdn", 0))}}
    else:
        days = {}
    events = d.get("events") if isinstance(d.get("events"), list) else []
    return {"schema_version": int(d.get("schema_version", 1)),
            "days": days, "events": events}


def _read_days() -> dict:
    """Совместимый helper: календарная история расхода для аудита/тестов."""
    return _read_state()["days"]


def _parse_event_time(raw, fallback_tz) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=fallback_tz)


def _migrate_legacy_state(state: dict, now: datetime) -> dict:
    """Один раз переносит календарные суммы в безопасное rolling-окно.

    Точного времени старых запросов нет. Сегодняшний расход считаем сделанным
    сейчас, вчерашний — в 23:59:59 вчера. Это может немного передержать квоту,
    зато миграция никогда не разрешит опасный двойной прогон.
    """
    if int(state.get("schema_version", 1)) >= BUDGET_SCHEMA_VERSION:
        return state
    today = now.date()
    yesterday = today - timedelta(days=1)
    migrated = []
    for raw_day, used in state.get("days", {}).items():
        try:
            day = datetime.strptime(raw_day, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if day == today:
            at = now
        elif day == yesterday:
            at = datetime.combine(day, dt_time(23, 59, 59), tzinfo=now.tzinfo)
        else:
            continue
        for host in ("kolesa", "cdn"):
            cost = int((used or {}).get(host, 0))
            if cost > 0:
                migrated.append({"at": at.isoformat(), "host": host,
                                 "cost": cost, "legacy": True})
    state["events"] = migrated
    state["schema_version"] = BUDGET_SCHEMA_VERSION
    return state


def _active_events(state: dict, now: datetime) -> list[dict]:
    cutoff = now - timedelta(hours=BUDGET_WINDOW_HOURS)
    active = []
    for event in state.get("events", []):
        at = _parse_event_time(event.get("at"), now.tzinfo)
        try:
            cost = int(event.get("cost", 0))
        except (TypeError, ValueError):
            continue
        if at is None or cost <= 0 or event.get("host") not in {"kolesa", "cdn"}:
            continue
        if at > cutoff:  # ровно 24 часа назад уже вышло из окна
            clean = {"at": at.isoformat(), "host": event["host"], "cost": cost}
            if event.get("legacy"):
                clean["legacy"] = True
            active.append(clean)
    return active


def _rolling_used(state: dict, now: datetime) -> dict:
    used = {"kolesa": 0, "cdn": 0}
    for event in _active_events(state, now):
        used[event["host"]] += event["cost"]
    return used


def _write_state(state: dict, now: datetime | None = None):
    now = now or _now()
    days = dict(state.get("days", {}))
    for old in sorted(days)[:-BUDGET_KEEP_DAYS]:
        days.pop(old)
    event_cutoff = now - timedelta(hours=BUDGET_EVENT_KEEP_HOURS)
    events = []
    for event in state.get("events", []):
        at = _parse_event_time(event.get("at"), now.tzinfo)
        if at is not None and at > event_cutoff:
            events.append(event)
    target = _p.Path(BUDGET_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump({"schema_version": BUDGET_SCHEMA_VERSION,
                       "days": days, "events": events}, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, target)
    finally:
        _p.Path(tmp_name).unlink(missing_ok=True)


def _write_days(days: dict):
    """Совместимая запись календарной истории (rolling-событий ещё нет)."""
    _write_state({"schema_version": 1, "days": days, "events": []})


@contextmanager
def _budget_lock():
    """Межпроцессная блокировка общего счётчика parser/catch_up.

    Последовательность «прочитать → проверить → записать» должна быть одной
    операцией. Иначе два случайно параллельных запуска оба видят свободное
    место и вместе пробивают антибан-потолок.
    """
    lock_path = _p.Path(str(BUDGET_FILE) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_budget_used() -> dict:
    """Сколько запросов на хост потрачено за последние ровно 24 часа."""
    with _budget_lock():
        now = _now()
        state = _read_state()
        was_legacy = int(state.get("schema_version", 1)) < BUDGET_SCHEMA_VERSION
        state = _migrate_legacy_state(state, now)
        if was_legacy:
            _write_state(state, now)
        return _rolling_used(state, now)


def save_budget_used(used: dict):
    """Совместимый setter: заменить текущий rolling-расход (для recovery)."""
    with _budget_lock():
        now = _now()
        state = _migrate_legacy_state(_read_state(), now)
        state["events"] = [
            {"at": now.isoformat(), "host": host, "cost": int(used[host])}
            for host in ("kolesa", "cdn") if int(used[host]) > 0
        ]
        state["days"][now.date().isoformat()] = {
            "kolesa": int(used["kolesa"]), "cdn": int(used["cdn"])}
        _write_state(state, now)


def charge_budget(host: str, cost: int) -> dict:
    """Списать cost запросов и вернуть расход за скользящие 24 часа."""
    with _budget_lock():
        now = _now()
        state = _migrate_legacy_state(_read_state(), now)
        key = now.date().isoformat()
        cur = state["days"].setdefault(key, {"kolesa": 0, "cdn": 0})
        cur[host] = int(cur.get(host, 0)) + int(cost)
        state["events"].append({"at": now.isoformat(), "host": host,
                                "cost": int(cost)})
        _write_state(state, now)
        return _rolling_used(state, now)


def reserve_budget(host: str, cost: int, limit: int) -> dict | None:
    """Атомарно проверить потолок и заранее зарезервировать запросы.

    `None` означает, что порция уже не помещается. Резерв консервативный:
    таймаут, 429 или авария после старта не возвращают квоту, потому что запрос
    уже мог дойти до сайта.
    """
    with _budget_lock():
        now = _now()
        state = _migrate_legacy_state(_read_state(), now)
        used = _rolling_used(state, now)
        if int(used.get(host, 0)) + int(cost) > int(limit):
            return None
        key = now.date().isoformat()
        cur = state["days"].setdefault(key, {"kolesa": 0, "cdn": 0})
        cur[host] = int(cur.get(host, 0)) + int(cost)
        state["events"].append({"at": now.isoformat(), "host": host,
                                "cost": int(cost)})
        _write_state(state, now)
        used[host] += int(cost)
        return used


def compute_gaps() -> dict:
    """Считает пробелы по каждому сетевому джобу (сентинел-aware)."""
    eng = get_engine()
    g = {}

    # 1) статусы: пропали из листинга >=STALE_DAYS, НЕ терминальны И давно
    #    (>=RECHECK_DAYS) не проверялись напрямую. Та же логика, что
    #    needs_status_check() в check_status.py (пороги синхронны, см. тест).
    last_seen = pd.read_sql(
        "SELECT ad_id, MAX(seen_date) AS seen FROM sightings GROUP BY ad_id",
        eng, dtype={"ad_id": str})
    st = pd.read_sql("SELECT ad_id, status, checked_at FROM ad_status", eng,
                     dtype={"ad_id": str})
    ls = last_seen.merge(st, on="ad_id", how="left")
    today = pd.Timestamp.today().normalize()
    seen_days = (today - pd.to_datetime(ls["seen"])).dt.days
    checked_days = (today - pd.to_datetime(ls["checked_at"])).dt.days   # NaN = не проверяли
    terminal = ls["status"].isin(["archived", "deleted"])
    recently_checked = checked_days < STATUS_RECHECK_DAYS               # NaN<7 → False
    g["status"] = int(((~terminal) & (seen_days >= STATUS_STALE_DAYS)
                       & (~recently_checked)).sum())

    # 2) обогащение: объявления из clean_data, которых нет в enriched
    clean_ids = set(pd.read_sql("SELECT ad_id FROM clean_data", eng,
                                dtype={"ad_id": str})["ad_id"])
    enr = pd.read_sql("SELECT ad_id, kolesa_avg_price, page_status_badge, http_status "
                      "FROM enriched", eng, dtype={"ad_id": str})
    g["enrich"] = len(clean_ids - set(enr["ad_id"]))

    # 3) avgPrice/бейдж: enriched-строки (http 200), где NULL хотя бы одно
    #    (сентинелы -1/"-" уже НЕ NULL → не считаются пробелом)
    ok = enr[enr["http_status"] == 200]
    g["backfill"] = int((ok["kolesa_avg_price"].isna()
                         | ok["page_status_badge"].isna()).sum())
    g["enriched_total"] = int(len(ok))   # знаменатель: сколько всего обогащено (http 200)

    # 4) фото-хэши: реальные (http) фото, которых нет в photo_hashes
    photos = pd.read_sql("SELECT url FROM photos", eng)
    photos = photos[photos["url"].fillna("").str.startswith("http")]
    hashed = set(pd.read_sql("SELECT url FROM photo_hashes", eng)["url"])
    g["photo"] = int((~photos["url"].isin(hashed)).sum())

    return g


# джоб → (человекочитаемое имя, скрипт, ключ пробела, хост)
KOLESA = [
    ("статусы (check_status)",   "kz.collect.check_status",     "status"),
    ("обогащение (enrich)",      "kz.collect.enrich",           "enrich"),
    ("avgPrice+бейдж (backfill)", "kz.collect.backfill_avgprice", "backfill"),
]
CDN = [("фото-хэши (photo_dedup)", "kz.collect.photo_dedup", "photo")]   # другой хост
OFFLINE = [("чистка (clean)", "kz.transform.clean"), ("отчёт (explore)", "kz.report.explore")]

# Приоритетный набор для --values: джобы, заполняющие ЦЕННЫЕ для ОПРАВДАНИЯ
# (exculpation) поля. backfill добирает avgPrice + бейдж у старых строк;
# enrich даёт их же у новых объявлений ПЛЮС цвет/damage/растаможку/коммент —
# всё, на что смотрит exculpate() в clean.py, снимая ложные подозрения.
# Статусы (liveness) сюда не входят, фото — тем более (оно ДОБАВляет
# подозрение shared_photo, а не снимает). Чем быстрее заполнены эти поля,
# тем быстрее чистится список подозрительных под разметку.
VALUE_JOBS = [j for j in KOLESA if j[2] in ("enrich", "backfill")]

# Ещё уже: только backfill — чистый добор avgPrice+бейджа у УЖЕ обогащённых
# строк (пропускает заполненные, целится в 554 из 1089 http-200). НЕ трогает
# enrich (новые объявления). Для --backfill.
BACKFILL_JOBS = [j for j in KOLESA if j[2] == "backfill"]


def is_429_line(line: str) -> bool:
    """Настоящее rate-limit-событие, а НЕ подстрока '429' в ad_id/цене/
    таймстемпе («наблюдений: 429», «,429»). Все джобы логируют реальный
    429 вместе со словом 'пауза' (backoff) или 'подряд' (стоп)."""
    return "429" in line and ("пауза" in line or "подряд" in line)


def count_429() -> int:
    """Число настоящих 429-событий в логах — для детекта новых между джобами."""
    n = 0
    for f in glob.glob("logs/*.log"):
        try:
            n += sum(is_429_line(ln) for ln in
                     _p.Path(f).read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            pass
    return n


def run(script: str) -> int:
    print(f"\n{'═'*60}\n▶ {script}\n{'═'*60}")
    return subprocess.run([sys.executable, "-m", script]).returncode


def next_action(gap_before: int, gap_after: int, rc: int, saw_new_429: bool) -> str:
    """Чистое решение «что делать после одной порции джоба» — вынесено
    отдельно от сети/subprocess, чтобы тестировалось без запусков.

    Порядок проверок важен:
      done         — пробел закрылся, дальше крутить нечего;
      rate_limited — в логах появился НОВЫЙ 429 → сайт просит стоп,
                     прерываем цепочку (важнее всего для анти-бана);
      breaker      — джоб вышел с ошибкой (rc!=0) = сработал его
                     внутренний предохранитель (N сбоев подряд);
      stuck        — порция отработала чисто, но пробел НЕ уменьшился:
                     остаток недозаполним (404/нет данных/сентинелы уже
                     проставлены) — иначе был бы вечный цикл на этих строках;
      continue     — прогресс есть, гоним следующую порцию.
    """
    if gap_after == 0:
        return "done"
    if saw_new_429:
        return "rate_limited"
    if rc != 0:
        return "breaker"
    if gap_after >= gap_before:
        return "stuck"
    return "continue"


def budget_allows(host: str, key: str, gap_before: int, used: dict,
                  run_spent: dict | None = None) -> bool:
    """Влезает ли ЕЩЁ ОДНА порция джоба в бюджет хоста. Оценка стоимости
    порции = min(MAX_PER_RUN, оставшийся пробел): для почти добитого джоба
    это его реальные несколько запросов, а не полный MAX_PER_RUN — иначе
    near-done джоб голодал бы у края квоты.

    Проверяются ДВА потолка, оба равны DAILY_BUDGET:
      used      — расход за последние скользящие 24 часа;
      run_spent — расход этого запуска.
    Второй — defence-in-depth: даже очень долгий процесс, из rolling-окна
    которого постепенно выпадают старые события, не получит больше одной
    полной квоты за один запуск.
    """
    cost = min(CHUNK_MAX[key], gap_before)
    if used[host] + cost > DAILY_BUDGET[host]:
        return False
    return run_spent is None or run_spent[host] + cost <= DAILY_BUDGET[host]


def run_one_chunk(name: str, script: str, key: str, host: str, used: dict,
                  run_spent: dict | None = None) -> str:
    """Одна порция джоба с учётом rolling-бюджета ХОСТА. Возвращает исход:
      done         — пробелов нет;
      budget       — не влезает в остаток 24-часовой квоты хоста;
      rate_limited — новый 429 (сайт просит стоп всей хостовой цепочки);
      breaker      — джоб вышел с ошибкой (внутренний предохранитель);
      stuck        — порция не сдвинула пробел (остаток недозаполним);
      progress     — есть прогресс, можно крутить дальше.
    `used` (host → потрачено за 24 часа) мутируется и сохраняется на диск."""
    gap_before = compute_gaps()[key]
    if gap_before == 0:
        return "done"
    # Между порциями старые события могли выйти из rolling-окна; перечитываем
    # атомарный файл перед решением. run_spent всё равно не даст одному
    # долгому процессу получить больше полной квоты.
    used.update(load_budget_used())
    if not budget_allows(host, key, gap_before, used, run_spent):
        return "budget"

    cost = min(CHUNK_MAX[key], gap_before)   # верхняя оценка запросов порции
    reserved = reserve_budget(host, cost, DAILY_BUDGET[host])
    if reserved is None:
        # Другой процесс мог занять остаток между первым чтением и этим местом.
        used.update(load_budget_used())
        return "budget"
    used.update(reserved)
    if run_spent is not None:
        run_spent[host] += cost
    before_429 = count_429()
    print(f"\n  {name}: осталось {gap_before}; бюджет {host} "
          f"{used[host]}/{DAILY_BUDGET[host]}; гоню порцию (≈{cost} запросов)…")
    rc = run(script)
    saw_429 = count_429() > before_429
    gap_after = compute_gaps()[key]
    action = next_action(gap_before, gap_after, rc, saw_429)
    print(f"  {name}: пробел {gap_before} → {gap_after}; "
          f"бюджет {host} {used[host]}/{DAILY_BUDGET[host]}")
    return "progress" if action == "continue" else action


def drain_host(jobs, host: str, used: dict, until_done: bool,
               run_spent: dict | None = None) -> bool:
    """Гоняет джобы ОДНОГО хоста, деля общий rolling-бюджет host.
      until_done=False: один проход — по одной порции на джоб.
      until_done=True: round-robin порциями, пока есть прогресс и бюджет
        (равномерно двигает все фронты, а не добивает первый джоб в ноль,
        оставляя остальные голодать у общей квоты).
    Джоб, вернувший done/stuck/budget, до конца этого запуска больше не
    трогаем (в blocked) — иначе stuck-джоб жёг бы бюджет каждый проход.
    Возвращает True, если надо ПРЕРВАТЬ весь запуск (429/предохранитель:
    это сигнал самого хоста, дальше по нему в этот раз не ходим)."""
    blocked = set()
    while True:
        progressed = False
        for name, script, key in jobs:
            if key in blocked:
                continue
            outcome = run_one_chunk(name, script, key, host, used, run_spent)
            if outcome in ("rate_limited", "breaker"):
                print(f"\n⚠ {name}: {outcome} — прерываю джобы хоста «{host}» "
                      "(один IP, бережём).")
                return True
            if outcome == "progress":
                progressed = True
                continue
            blocked.add(key)   # done | stuck | budget — до следующего запуска
            if outcome == "done":
                print(f"✓ {name}: пробелов нет")
            elif outcome == "stuck":
                print(f"⚠ {name}: порция не сдвинула пробел — остаток "
                      "недозаполним (404/нет данных), пропускаю")
            elif outcome == "budget":
                print(f"⏸ {name}: квота «{host}» за 24 часа почти выбрана "
                      f"({used[host]}/{DAILY_BUDGET[host]}) — жду освобождения окна")
        if not until_done or not progressed or len(blocked) == len(jobs):
            return False


def report(g: dict, title: str):
    print(f"\n{LINE}\n{title}\n{LINE}")
    labels = {"status": "статусы к проверке", "enrich": "не обогащено",
              "backfill": "avgPrice/бейдж не добраны", "photo": "фото не хэшировано"}
    for k in ["status", "enrich", "backfill", "photo"]:
        mark = "—" if g[k] == 0 else str(g[k])
        extra = (f"  (из {g['enriched_total']} обогащённых)"
                 if k == "backfill" and g.get("enriched_total") else "")
        print(f"  {labels[k]:<28} {mark}{extra}")
    print(LINE)


def run_gapped_jobs(until_done: bool = False, kolesa_jobs=None, do_cdn: bool = True):
    """Сетевые джобы под скользящим 24-часовым бюджетом на хост.

    until_done=False (по умолчанию): один проход — по одной порции на джоб
      (в пределах бюджета) — вежливо, резюмируемо, за пару минут.
    until_done=True: используем всю оставшуюся квоту окна (round-robin
      порциями), потом встаём до освобождения старых событий.
    kolesa_jobs: какой набор kolesa-джобов гнать (KOLESA / VALUE_JOBS /
      BACKFILL_JOBS — выбирается флагами в main). do_cdn: трогать ли фото.
    Хосты идут по очереди (kolesa → CDN), у каждого свой бюджет. 429/
    предохранитель на хосте прерывает только ЕГО цепочку."""
    used = load_budget_used()
    run_spent = {"kolesa": 0, "cdn": 0}   # потолок на сам запуск, см. budget_allows
    t0 = time.time()

    kolesa_jobs = KOLESA if kolesa_jobs is None else kolesa_jobs
    kolesa_aborted = drain_host(kolesa_jobs, "kolesa", used, until_done, run_spent)
    if do_cdn:
        drain_host(CDN, "cdn", used, until_done, run_spent)
    if kolesa_aborted:
        print("\n(kolesa прерван по сигналу сайта; CDN — отдельный хост, "
              "его добор это не затрагивает.)")

    # офлайн-пересборка ВСЕГДА (влить то, что успели добрать, во флаги)
    for _name, script in OFFLINE:
        run(script)

    print(f"\n✔ catch_up завершён за {(time.time()-t0)/60:.1f} мин")
    print(f"  бюджет за 24 часа: kolesa {used['kolesa']}/{DAILY_BUDGET['kolesa']}, "
          f"CDN {used['cdn']}/{DAILY_BUDGET['cdn']}")
    report(compute_gaps(), "ОСТАЛОСЬ ПОСЛЕ ПРОГОНА")


def main():
    until_done = "--until-done" in sys.argv
    backfill_only = "--backfill" in sys.argv     # уже некуда: только avgPrice+бейдж
    values = "--values" in sys.argv and not backfill_only   # backfill приоритетнее

    # Бюджет настраиваемый: --budget N важнее env KOLESA_BUDGET важнее дефолта.
    cli_budget = parse_budget(sys.argv)
    if cli_budget is not None:
        DAILY_BUDGET["kolesa"] = cli_budget

    g = compute_gaps()
    report(g, "ПРОБЕЛЫ СЕЙЧАС (что можно добрать)")

    used = load_budget_used()
    label, note = risk_zone(DAILY_BUDGET["kolesa"])
    print(f"Скользящий бюджет запросов (израсходовано за 24 часа): "
          f"kolesa {used['kolesa']}/{DAILY_BUDGET['kolesa']} [{label}], "
          f"CDN {used['cdn']}/{DAILY_BUDGET['cdn']}")
    if label in ("риск", "высокий риск"):
        print(f"  ⚠ {label}: {note}")
    print_risk_help(DAILY_BUDGET["kolesa"])

    # набор джобов и «нечего делать» — по выбранному фокусу
    if backfill_only:
        kolesa_jobs, do_cdn, net = BACKFILL_JOBS, False, g["backfill"]
    elif values:
        kolesa_jobs, do_cdn, net = VALUE_JOBS, False, g["enrich"] + g["backfill"]
    else:
        kolesa_jobs, do_cdn = KOLESA, True
        net = g["status"] + g["enrich"] + g["backfill"] + g["photo"]
    if net == 0:
        print("\nНечего добирать в выбранном режиме.")
        return

    if backfill_only:
        print(f"\nРежим --backfill: добираю ТОЛЬКО avgPrice+бейдж у обогащённых "
              f"({g['backfill']} из {g['enriched_total']}), заполненные пропускаю.")
        print("enrich (новые), статусы и фото не трогаю.")
        doable = min(g["backfill"], max(0, DAILY_BUDGET["kolesa"] - used["kolesa"]))
        print(f"Влезает в остаток rolling-бюджета: {doable} "
              f"из {g['backfill']} (≈{eta_minutes(doable):.0f} мин).")
    elif values:
        print("\nРежим --values: обогащение + avgPrice/бейдж (enrich + backfill),")
        print("без статусов и фото — быстрая чистка подозрительных под разметку.")
    if until_done:
        print("--until-done: вся оставшаяся 24-часовая квота (round-robin; стоп при 429/бюджете).")
    elif not (values or backfill_only):
        print("\nОдин проход по всем джобам в пределах бюджета. Фокус: --values / --backfill.")

    flags = (" --until-done" if until_done else "") \
        + (" --backfill" if backfill_only else (" --values" if values else "")) \
        + (f" --budget {cli_budget}" if cli_budget is not None else "")
    if "--run" in sys.argv:
        run_gapped_jobs(until_done, kolesa_jobs, do_cdn)
        return
    if not sys.stdin.isatty():
        print(f"\nЗапустить: python -m kz.ops.catch_up --run{flags}")
        return
    ans = input("\nЗапустить догон сейчас? [y/N] ").strip().lower()
    if ans in ("y", "yes", "д", "да"):
        run_gapped_jobs(until_done, kolesa_jobs, do_cdn)
    else:
        print(f"Ок, не запускаю. Когда решишь: python -m kz.ops.catch_up --run{flags}")


if __name__ == "__main__":
    main()
