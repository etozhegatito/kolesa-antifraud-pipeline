"""
Job 1c: проверка статуса собранных объявлений (check_status.py).

Зачем: «исчезло из наших 10 страниц листинга» ≠ «снято с сайта» —
объявление могло просто уползти глубже. Если помечать такие «продано»,
получим label noise (ошибки в целевой метке) и модель выучит наши ошибки.
Этот джоб проверяет статус НАПРЯМУЮ, запрашивая страницу объявления.

Стейты (проверено на реальных страницах kolesa):
  active   — HTTP 200, обычная страница
  archived — HTTP 200 + «Объявление находится в архиве» (продано/истекло)
  deleted  — HTTP 404
archived и deleted — терминальные состояния: раз попав туда, объявление
не возвращается, поэтому их мы больше никогда не перепроверяем.

Кого проверяем (чтобы не жечь запросы зря):
  - только тех, кого листинг НЕ видел ≥ STALE_DAYS дней
    (кого видели сегодня — и так известно, что active);
  - не больше MAX_CHECKS_PER_RUN за запуск (вежливость к сайту).

Запуск: python check_status.py   (раз в день, после parser.py)
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "check_status.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import csv
import logging
import random
import sys
import time
from datetime import datetime, date
from pathlib import Path

import requests

from db import upsert

RAW_CSV       = "data/raw/raw_data.csv"
SIGHTINGS_CSV = "data/raw/sightings.csv"
STATUS_CSV    = "data/raw/ad_status.csv"
LOG_FILE      = "logs/status.log"

STALE_DAYS         = 2      # не видели в листинге столько дней → статус под
                            # сомнением (буфер против «мигания» пагинации/VIP-
                            # ротации: увидим завтра — сходили бы зря). Виден
                            # свежее — точно active, запрос не нужен.
RECHECK_DAYS       = 7      # проверяли напрямую не позже стольких дней → не
                            # долбим повторно, иначе один и тот же active
                            # перезапрашивался бы каждый прогон, а база не
                            # обходилась бы вширь. Реальный охват всё равно
                            # ограничен MAX_CHECKS_PER_RUN.
MAX_CHECKS_PER_RUN = 150    # дневной лимит запросов этого джоба
DELAY_RANGE        = (4.0, 8.0)
MAX_CONSECUTIVE_FAILS = 3

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}
# В сыром HTML пробел закодирован сущностью: «в&nbsp;архиве».
# Проверяем несколько вариантов написания + бейдж «В архиве».
ARCHIVE_MARKERS = ["в&nbsp;архиве", "в\u00a0архиве", "в архиве", ">В архиве<"]

STATUS_FIELDS = ["ad_id", "status", "checked_at"]


def infer_active_from_listing(cur_status, seen_days, seen_after_check) -> bool:
    """Ставить ли active БЕЗ запроса, потому что объявление показалось в
    листинге (присутствие в листинге = доказательство, что оно живо).
    Пишем ТОЛЬКО изменения — уже-active не переписываем каждый прогон:
      • новый ad_id (статуса ещё не было) → active;
      • терминальный (archived/deleted), но увиденный в листинге ПОЗЖЕ
        последней проверки → реактивация (продавец продлил) → active.
    Чистая функция (без сети/I/O) — тестируется отдельно."""
    if seen_days is None or seen_days >= STALE_DAYS:
        return False                          # не свежий в листинге
    if cur_status is None:
        return True                           # новый — статуса ещё не было
    if cur_status in ("archived", "deleted") and seen_after_check:
        return True                           # реактивация
    return False                              # уже active — писать нечего


def needs_status_check(cur_status, seen_days, checked_days) -> bool:
    """Нужен ли СЕТЕВОЙ пере-запрос статуса. seen_days/checked_days — сколько
    дней назад видели в листинге / проверяли напрямую (None = никогда).
      • терминальный — нет (archived/deleted не воскресают на том же ad_id);
      • свежий в листинге (<STALE_DAYS) — нет, и так знаем что active;
      • проверяли недавно (<RECHECK_DAYS) — нет, остынь;
      • иначе (пропал из листинга и давно не проверяли) — да."""
    if cur_status in ("archived", "deleted"):
        return False
    if seen_days is not None and seen_days < STALE_DAYS:
        return False
    if checked_days is not None and checked_days < RECHECK_DAYS:
        return False
    return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_last_seen() -> dict:
    """ad_id → дата последнего появления в листинге."""
    last = {}
    if not Path(SIGHTINGS_CSV).exists():
        return last
    with open(SIGHTINGS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["seen_date"]
            if r["ad_id"] not in last or d > last[r["ad_id"]]:
                last[r["ad_id"]] = d
    return last


def load_status_rows() -> dict:
    """ad_id → (status, last_checked: date|None). Файл append-only:
    последняя запись за ad_id побеждает."""
    rows = {}
    if not Path(STATUS_CSV).exists():
        with open(STATUS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=STATUS_FIELDS).writeheader()
        return rows
    with open(STATUS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            checked = (r.get("checked_at") or "")[:10]   # ISO: берём дату
            try:
                d = date.fromisoformat(checked) if checked else None
            except ValueError:
                d = None
            rows[r["ad_id"]] = (r["status"], d)
    return rows


def append_status(ad_id: str, status: str):
    checked_at = datetime.now().isoformat(timespec="seconds")
    with open(STATUS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=STATUS_FIELDS).writerow({
            "ad_id": ad_id, "status": status, "checked_at": checked_at,
        })

    # Двойная запись (пилот миграции на Postgres, см. план): CSV остаётся
    # источником истины, сбой записи в БД не должен ронять прогон.
    # DO UPDATE (не DO NOTHING) — та же семантика "последняя запись
    # побеждает", что и в CSV-версии этой функции.
    try:
        upsert("ad_status", [{"ad_id": ad_id, "status": status, "checked_at": checked_at}],
               ["ad_id"], update_cols=["status", "checked_at"])
    except Exception as e:
        log.warning(f"Postgres upsert не удался для {ad_id}: {e}")


def check_ad(ad_id: str, session: requests.Session) -> str | None:
    """Возвращает active / archived / deleted, либо None при сетевой ошибке."""
    url = f"https://kolesa.kz/a/show/{ad_id}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        log.warning(f"{ad_id}: сетевая ошибка {e}")
        return None
    if resp.status_code == 404:
        return "deleted"
    if resp.status_code == 200:
        archived = any(m in resp.text for m in ARCHIVE_MARKERS)
        return "archived" if archived else "active"
    if resp.status_code == 429:
        # сайт просит притормозить — тормозим НАДОЛГО. Вернём None: в main
        # это считается сбоем, и предохранитель (3 подряд) остановит джоб.
        log.warning(f"{ad_id}: 429 — пауза 120с")
        time.sleep(120)
        return None
    log.warning(f"{ad_id}: неожиданный HTTP {resp.status_code}")
    return None


def main():
    last_seen = load_last_seen()
    rows = load_status_rows()
    today = date.today()

    # ── Шаг A: «показался в листинге ⇒ active» — без единого запроса.
    # Присутствие в листинге само доказывает, что объявление живо; заодно
    # чинит реактивацию (archived→active, если продавец продлил). Пишем
    # только изменения — уже-active не трогаем (иначе раздули бы журнал).
    marked = 0
    for ad_id, seen in last_seen.items():
        seen_d = date.fromisoformat(seen)
        seen_days = (today - seen_d).days
        cur_status, cur_checked = rows.get(ad_id, (None, None))
        seen_after_check = cur_checked is None or seen_d > cur_checked
        if infer_active_from_listing(cur_status, seen_days, seen_after_check):
            append_status(ad_id, "active")
            marked += 1
    if marked:
        log.info(f"Помечено active из листинга (без запросов): {marked}")

    # ── Шаг B: сетевые кандидаты — пропали из листинга, не терминальны,
    # давно не проверялись напрямую. Самые «протухшие» первыми (по ним
    # метка нужнее всего).
    candidates = []
    for ad_id, seen in last_seen.items():
        cur_status, cur_checked = rows.get(ad_id, (None, None))
        seen_days = (today - date.fromisoformat(seen)).days
        checked_days = None if cur_checked is None else (today - cur_checked).days
        if needs_status_check(cur_status, seen_days, checked_days):
            candidates.append((seen_days, ad_id))
    candidates.sort(reverse=True)
    batch = [ad for _, ad in candidates[:MAX_CHECKS_PER_RUN]]
    log.info(f"Кандидатов: {len(candidates)}, проверяем: {len(batch)}")

    session = requests.Session()
    fails = 0
    counts = {"active": 0, "archived": 0, "deleted": 0}
    for i, ad_id in enumerate(batch, 1):
        status = check_ad(ad_id, session)
        if status is None:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Стоп: сбои подряд — продолжим завтра.")
                sys.exit(1)
            time.sleep(30)
            continue
        fails = 0
        append_status(ad_id, status)
        counts[status] += 1
        if i % 25 == 0:
            log.info(f"  {i}/{len(batch)}: {counts}")
        time.sleep(random.uniform(*DELAY_RANGE))

    log.info(f"Готово: {counts} → {STATUS_CSV}")


if __name__ == "__main__":
    main()