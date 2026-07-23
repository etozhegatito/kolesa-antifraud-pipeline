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

СУТОЧНЫЙ БЮДЖЕТ ЗАПРОСОВ НА ХОСТ (главный анти-бан-рычаг):
  Бан ловится по ОБЪЁМУ запросов с одного IP за сутки, а не по паузе
  между ними (паузы 4-8с уже внутри джобов). Поэтому поверх всего —
  общий на хост дневной потолок числа запросов (DAILY_BUDGET). Он
  ОБЩИЙ для трёх kolesa-джобов (это один IP!), у CDN — отдельный.
  Счётчик потраченного живёт в logs/.catch_up_budget.json и сбрасывается
  с началом новых суток. Как только квота хоста выбрана — джобы этого
  хоста встают до завтра (резюмируемо). Так НИ ОДИН запуск (даже
  случайный ручной, даже --until-done) не пробьёт суточный объём.

Сентинелы (важно для подсчёта «пробелов»): avgPrice = -1 и бейдж = "-"
означают «проверено, значения у объявления НЕТ» — это НЕ пробел, повторно
не качаем. Пробел = только NULL («ещё не смотрели»).

Джоб запускается, ТОЛЬКО если у него реально есть пробел (нечего качать —
не ходим в сеть зря). В конце — офлайн-пересборка clean+отчёт, чтобы
свежие данные попали во флаги.

Запуск: python catch_up.py             (отчёт + вопрос, запускать ли)
        python catch_up.py --run        (одна порция на джоб, без вопроса)
        python catch_up.py --run --until-done
                                        (использовать всю дневную квоту:
                                         крутит порциями, пока не выбран
                                         суточный бюджет хоста / не закрыты
                                         пробелы / не пришёл 429; резюмируемо
                                         назавтра — идеально под ежедневный крон)
        python catch_up.py --run --values
                                        (приоритетно ТОЛЬКО ценные-для-оправдания
                                         поля: enrich + backfill = avgPrice/бейдж/
                                         цвет/damage/растаможка. Статусы и фото
                                         пропускает. Быстро чистит подозрительных
                                         под разметку; сочетается с --until-done)
        python catch_up.py --run --backfill
                                        (ещё уже: ТОЛЬКО добор avgPrice+бейджа у
                                         УЖЕ обогащённых строк — целится в незапол-
                                         ненные, заполненные пропускает, новые
                                         объявления не обогащает; тоже с --until-done)
"""

import pathlib as _p
_expected = "catch_up.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import glob
import json
import subprocess
import sys
import time
from datetime import date

import pandas as pd

from db import get_engine

LINE = "─" * 64

# ─── Суточный бюджет запросов на ХОСТ (анти-бан, см. докстринг) ──────────────
# ОБЩИЙ для всех kolesa-джобов (один IP!); у CDN — свой.
# 2026-07-23: домашний IP словил ВРЕМЕННЫЙ бан kolesa (~270 запросов за день с
# одного IP: catch_up + run_all + ручной браузинг). Снизили kolesa 400→200 с
# запасом. ВАЖНО: бюджет видит только catch_up — run_all/parser и твой браузинг
# kolesa НЕ учитывает, а они бьют по тому же IP. В дни добора: run_all --light
# и не стакать всё разом. При budget 200 обычный --run делает ~одну порцию
# (статусы 150); точечный добор avgPrice/бейджа — --backfill (120).
DAILY_BUDGET = {"kolesa": 200, "cdn": 1200}
BUDGET_FILE  = "logs/.catch_up_budget.json"

# Верхняя оценка запросов за ОДНУ порцию джоба (= его MAX_PER_RUN). Держим
# копией здесь, чтобы не импортировать джобы (у них при импорте открываются
# лог-файлы и настраивается root logger). Синхронность с источником стережёт
# test_catch_up_chunk_sizes_match_jobs.
CHUNK_MAX = {"status": 150, "enrich": 120, "backfill": 120, "photo": 300}

# Пороги «нужен ли пере-запрос статуса» ДОЛЖНЫ совпадать с check_status.py,
# иначе счётчик пробелов разошёлся бы с реальной выборкой джоба. Синхронность
# стережёт test_catch_up_status_thresholds_match_check_status.
STATUS_STALE_DAYS   = 2     # пропал из листинга дольше → статус под сомнением
STATUS_RECHECK_DAYS = 7     # проверяли напрямую позже → не считаем пробелом


def load_budget_used() -> dict:
    """Сколько запросов на хост уже потрачено СЕГОДНЯ (файл-счётчик с датой;
    другой день → нули). Битый/отсутствующий файл трактуем как «ещё ноль»."""
    try:
        d = json.loads(_p.Path(BUDGET_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    if d.get("date") != date.today().isoformat():
        return {"kolesa": 0, "cdn": 0}
    return {"kolesa": int(d.get("kolesa", 0)), "cdn": int(d.get("cdn", 0))}


def save_budget_used(used: dict):
    _p.Path(BUDGET_FILE).write_text(json.dumps(
        {"date": date.today().isoformat(),
         "kolesa": used["kolesa"], "cdn": used["cdn"]}), encoding="utf-8")


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
    ("статусы (check_status)",   "check_status.py",     "status"),
    ("обогащение (enrich)",      "enrich.py",           "enrich"),
    ("avgPrice+бейдж (backfill)", "backfill_avgprice.py", "backfill"),
]
CDN = [("фото-хэши (photo_dedup)", "photo_dedup.py", "photo")]   # другой хост
OFFLINE = [("чистка (clean)", "clean.py"), ("отчёт (explore)", "explore.py")]

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
    return subprocess.run([sys.executable, script]).returncode


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


def budget_allows(host: str, key: str, gap_before: int, used: dict) -> bool:
    """Влезает ли ЕЩЁ ОДНА порция джоба в дневной бюджет хоста. Оценка
    стоимости порции = min(MAX_PER_RUN, оставшийся пробел): для почти
    добитого джоба это его реальные несколько запросов, а не полный
    MAX_PER_RUN — иначе near-done джоб голодал бы у края квоты."""
    cost = min(CHUNK_MAX[key], gap_before)
    return used[host] + cost <= DAILY_BUDGET[host]


def run_one_chunk(name: str, script: str, key: str, host: str, used: dict) -> str:
    """Одна порция джоба с учётом дневного бюджета ХОСТА. Возвращает исход:
      done         — пробелов нет;
      budget       — не влезает в остаток суточной квоты хоста (стоп до завтра);
      rate_limited — новый 429 (сайт просит стоп всей хостовой цепочки);
      breaker      — джоб вышел с ошибкой (внутренний предохранитель);
      stuck        — порция не сдвинула пробел (остаток недозаполним);
      progress     — есть прогресс, можно крутить дальше.
    `used` (host → потрачено сегодня) мутируется и сохраняется на диск."""
    gap_before = compute_gaps()[key]
    if gap_before == 0:
        return "done"
    if not budget_allows(host, key, gap_before, used):
        return "budget"

    cost = min(CHUNK_MAX[key], gap_before)   # верхняя оценка запросов порции
    before_429 = count_429()
    print(f"\n  {name}: осталось {gap_before}; бюджет {host} "
          f"{used[host]}/{DAILY_BUDGET[host]}; гоню порцию (≈{cost} запросов)…")
    rc = run(script)
    # Списываем сразу после запуска (консервативно: даже если внутри был
    # 429/сбой и реальных запросов меньше — бюджет только НЕ пробьём).
    used[host] += cost
    save_budget_used(used)

    saw_429 = count_429() > before_429
    gap_after = compute_gaps()[key]
    action = next_action(gap_before, gap_after, rc, saw_429)
    print(f"  {name}: пробел {gap_before} → {gap_after}; "
          f"бюджет {host} {used[host]}/{DAILY_BUDGET[host]}")
    return "progress" if action == "continue" else action


def drain_host(jobs, host: str, used: dict, until_done: bool) -> bool:
    """Гоняет джобы ОДНОГО хоста, деля общий дневной бюджет host.
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
            outcome = run_one_chunk(name, script, key, host, used)
            if outcome in ("rate_limited", "breaker"):
                print(f"\n⚠ {name}: {outcome} — прерываю джобы хоста «{host}» "
                      "(один IP, бережём).")
                return True
            if outcome == "progress":
                progressed = True
                continue
            blocked.add(key)   # done | stuck | budget — до завтра
            if outcome == "done":
                print(f"✓ {name}: пробелов нет")
            elif outcome == "stuck":
                print(f"⚠ {name}: порция не сдвинула пробел — остаток "
                      "недозаполним (404/нет данных), пропускаю")
            elif outcome == "budget":
                print(f"⏸ {name}: дневная квота «{host}» почти выбрана "
                      f"({used[host]}/{DAILY_BUDGET[host]}) — добью завтра")
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
    """Сетевые джобы под дневным бюджетом на хост.

    until_done=False (по умолчанию): один проход — по одной порции на джоб
      (в пределах бюджета) — вежливо, резюмируемо, за пару минут.
    until_done=True: используем всю оставшуюся дневную квоту (round-robin
      порциями), потом встаём до завтра.
    kolesa_jobs: какой набор kolesa-джобов гнать (KOLESA / VALUE_JOBS /
      BACKFILL_JOBS — выбирается флагами в main). do_cdn: трогать ли фото.
    Хосты идут по очереди (kolesa → CDN), у каждого свой бюджет. 429/
    предохранитель на хосте прерывает только ЕГО цепочку."""
    used = load_budget_used()
    t0 = time.time()

    kolesa_jobs = KOLESA if kolesa_jobs is None else kolesa_jobs
    kolesa_aborted = drain_host(kolesa_jobs, "kolesa", used, until_done)
    if do_cdn:
        drain_host(CDN, "cdn", used, until_done)
    if kolesa_aborted:
        print("\n(kolesa прерван по сигналу сайта; CDN — отдельный хост, "
              "его добор это не затрагивает.)")

    # офлайн-пересборка ВСЕГДА (влить то, что успели добрать, во флаги)
    for name, script in OFFLINE:
        run(script)

    print(f"\n✔ catch_up завершён за {(time.time()-t0)/60:.1f} мин")
    print(f"  бюджет за сегодня: kolesa {used['kolesa']}/{DAILY_BUDGET['kolesa']}, "
          f"CDN {used['cdn']}/{DAILY_BUDGET['cdn']}")
    report(compute_gaps(), "ОСТАЛОСЬ ПОСЛЕ ПРОГОНА")


def main():
    until_done = "--until-done" in sys.argv
    backfill_only = "--backfill" in sys.argv     # уже некуда: только avgPrice+бейдж
    values = "--values" in sys.argv and not backfill_only   # backfill приоритетнее

    g = compute_gaps()
    report(g, "ПРОБЕЛЫ СЕЙЧАС (что можно добрать)")

    used = load_budget_used()
    print(f"Дневной бюджет запросов (израсходовано сегодня): "
          f"kolesa {used['kolesa']}/{DAILY_BUDGET['kolesa']}, "
          f"CDN {used['cdn']}/{DAILY_BUDGET['cdn']}")

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
    elif values:
        print("\nРежим --values: обогащение + avgPrice/бейдж (enrich + backfill),")
        print("без статусов и фото — быстрая чистка подозрительных под разметку.")
    if until_done:
        print("--until-done: вся оставшаяся дневная квота (round-robin; стоп при 429/бюджете).")
    elif not (values or backfill_only):
        print("\nОдин проход по всем джобам в пределах бюджета. Фокус: --values / --backfill.")

    flags = (" --until-done" if until_done else "") \
        + (" --backfill" if backfill_only else (" --values" if values else ""))
    if "--run" in sys.argv:
        run_gapped_jobs(until_done, kolesa_jobs, do_cdn)
        return
    if not sys.stdin.isatty():
        print(f"\nЗапустить: python catch_up.py --run{flags}")
        return
    ans = input("\nЗапустить догон сейчас? [y/N] ").strip().lower()
    if ans in ("y", "yes", "д", "да"):
        run_gapped_jobs(until_done, kolesa_jobs, do_cdn)
    else:
        print(f"Ок, не запускаю. Когда решишь: python catch_up.py --run{flags}")


if __name__ == "__main__":
    main()
