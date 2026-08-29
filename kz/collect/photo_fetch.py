# -*- coding: utf-8 -*-
"""Скачивание фотографий на диск — сырьё для работы с изображениями.

Зачем отдельно от photo_dedup: тот считает перцептивный хэш и картинку
выбрасывает, потому что для поиска дублей достаточно отпечатка. Для двух
следующих задач нужны сами пиксели:

  качество снимка — резкость, яркость, разрешение: отсюда рекомендации
                    продавцу «третье фото смазано», «нет снимка салона»;
  признаки цены   — эмбеддинг изображения как вход модели. Это прямая
                    проверка гипотезы, ради которой всё затевалось: на
                    дешёвых и старых машинах цену определяет состояние,
                    а в табличных признаках его нет.

ХОСТ И БЮДЖЕТ. Фото лежат на CDN kcdn.kz — это НЕ kolesa.kz, у него свой
суточный лимит (см. catch_up.DAILY_BUDGET). Скачивание фотографий не расходует
квоту, из-за которой берегли основной сайт. Расход всё равно записывается в
общий счётчик, иначе два потребителя CDN считали бы каждый своё.

ХРАНЕНИЕ. Картинка ужимается до MAX_SIDE по длинной стороне и пишется JPEG.
Оригиналы держать незачем: и для эмбеддингов, и для оценки резкости этого с
запасом, а место экономится в разы.

РЕЗЮМИРУЕМОСТЬ. Уже скачанное пропускается по наличию файла, неудачи
записываются в манифест, чтобы мёртвые ссылки не перекачивались вечно.

Запуск:
    python -m kz.collect.photo_fetch                 обложки, порция по умолчанию
    python -m kz.collect.photo_fetch --limit 500     сколько скачать за раз
    python -m kz.collect.photo_fetch --all-positions не только обложки
"""

import pathlib as _p
_expected = "photo_fetch.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import csv
import io
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps

from kz.collect.enrich import HEADERS
from kz.core import pacing
from kz.core.db import get_engine
from kz.ops.catch_up import DAILY_BUDGET, charge_budget, load_budget_used

PHOTO_DIR = Path("data/photos")
MANIFEST = PHOTO_DIR / "manifest.csv"
LOG_FILE = "logs/photo_fetch.log"

MAX_PER_RUN = 300            # порция; CDN терпимее сайта, но меру знаем
MAX_SIDE = 768               # длинная сторона после сжатия, пикселей
JPEG_QUALITY = 85
DELAY_RANGE = (0.8, 2.0)     # картинки легче страниц — пауза меньше
BREAK_EVERY = 120            # перерыв реже, чем у страниц: CDN раздаёт
                             # статику, и пауза каждые 15 файлов означала бы,
                             # что job три четверти времени просто спит
MAX_CONSECUTIVE_FAILS = 10

MANIFEST_COLS = ["ad_id", "position", "url", "path", "http_status",
                 "width", "height", "bytes", "fetched_at"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()])
log = logging.getLogger(__name__)


def local_path(ad_id: str, position: int) -> Path:
    """Путь к файлу. Раскладываем по подпапкам с двумя первыми символами
    ad_id: тысячи файлов в одном каталоге тормозят и файловую систему, и
    любой ls."""
    return PHOTO_DIR / str(ad_id)[:2] / f"{ad_id}_{position}.jpg"


def load_manifest() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST, dtype={"ad_id": str})
    return pd.DataFrame(columns=MANIFEST_COLS)


def append_manifest(rows: list[dict]) -> None:
    """Дописываем csv-модулем: pandas превращает целые в «50.0» при
    round-trip (правило проекта №4)."""
    if not rows:
        return
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fresh = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if fresh:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLS})


# Коды, после которых повторять бессмысленно: файла нет и не будет.
# Всё остальное (таймаут, обрыв, 5xx) — временное, и такие ссылки должны
# оставаться в очереди. Иначе одна сетевая икота навсегда теряла бы фото.
PERMANENT_STATUSES = {200, 404, 410}


def live_hosts(urls) -> set[str]:
    """Хосты, которые сейчас резолвятся.

    Нужно потому, что 2026-08-09 обнаружилось: kolesa вывел из эксплуатации
    один из двух CDN-хостов, и 37% наших ссылок стали недостижимы. Проверять
    DNS по одному разу на хост вместо тысячи одинаковых падений — и быстрее,
    и в логе видно причину. Мёртвые ссылки при этом НЕ помечаются навсегда:
    если хост вернётся, они снова попадут в очередь.
    """
    import socket
    hosts = {u.split("/")[2] for u in urls}
    alive = set()
    for h in sorted(hosts):
        try:
            socket.getaddrinfo(h, 443)
            alive.add(h)
        except OSError:
            log.warning(f"хост {h} не резолвится — пропускаю его ссылки")
    return alive


def pick_targets(limit: int, covers_only: bool = True,
                 complete_only: bool = False) -> pd.DataFrame:
    """Что качать: сначала обложки — они есть у всех объявлений, и по одной
    на каждое даёт максимальный охват на единицу трафика."""
    ph = pd.read_sql("SELECT ad_id, position, url FROM photos", get_engine(),
                     dtype={"ad_id": str})
    ph = ph[ph["url"].fillna("").str.startswith("http")]
    if covers_only:
        ph = ph[ph["position"] == ph.groupby("ad_id")["position"].transform("min")]

    man = load_manifest()
    if complete_only and len(man):
        started = set(man.loc[man["http_status"] == 200, "ad_id"])
        ph = ph[ph["ad_id"].isin(started)]
    if len(man):
        done = set(man.loc[man["http_status"].isin(PERMANENT_STATUSES), "url"])
        ph = ph[~ph["url"].isin(done)]
    ph = ph[~ph.apply(lambda r: local_path(r.ad_id, r.position).exists(), axis=1)]

    alive = live_hosts(ph["url"])
    ph = ph[ph["url"].str.split("/").str[2].isin(alive)]
    return ph.sort_values(["position", "ad_id"]).head(limit)


def save_image(content: bytes, dest: Path) -> tuple[int, int, int]:
    """Ужать и сохранить. Возвращает (ширина, высота, байт на диске).

    Приводим к RGB: часть картинок приходит в webp с альфа-каналом, а JPEG
    прозрачность не умеет и падает на сохранении.
    """
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(content)))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return img.width, img.height, dest.stat().st_size


def main():
    limit = MAX_PER_RUN
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    covers_only = "--all-positions" not in sys.argv
    # Дополнить уже начатые объявления: позиции 2-5 там, где обложка есть.
    # Полный комплект по части машин полезнее, чем по одной картинке у всех:
    # гипотеза о видимых повреждениях проверяется только на комплектах.
    complete_only = "--complete" in sys.argv
    if complete_only:
        covers_only = False

    used = load_budget_used()
    left = DAILY_BUDGET["cdn"] - used["cdn"]
    if left <= 0:
        log.info(f"Суточный лимит CDN выбран ({used['cdn']}/{DAILY_BUDGET['cdn']}), "
                 "продолжим завтра.")
        return
    limit = min(limit, left)

    targets = pick_targets(limit, covers_only, complete_only)
    if targets.empty:
        log.info("Нечего качать: всё скачано или помечено как недоступное.")
        return
    log.info(f"К скачиванию {len(targets)} "
             f"({'обложки' if covers_only else 'все позиции'}); "
             f"лимит CDN {used['cdn']}/{DAILY_BUDGET['cdn']}")

    session = requests.Session()
    rows, ok, fails, streak = [], 0, 0, 0
    for i, r in enumerate(targets.itertuples(index=False), 1):
        dest = local_path(r.ad_id, r.position)
        try:
            resp = session.get(r.url, headers=HEADERS, timeout=20)
            status = resp.status_code
            if status == 200:
                w, h, size = save_image(resp.content, dest)
                rows.append(dict(ad_id=r.ad_id, position=r.position, url=r.url,
                                 path=str(dest), http_status=200, width=w,
                                 height=h, bytes=size,
                                 fetched_at=datetime.now().isoformat(timespec="seconds")))
                ok += 1
                streak = 0
            else:
                rows.append(dict(ad_id=r.ad_id, position=r.position, url=r.url,
                                 path="", http_status=status, width="", height="",
                                 bytes="",
                                 fetched_at=datetime.now().isoformat(timespec="seconds")))
                fails += 1
                streak += 1
        except Exception as e:                    # noqa: BLE001 — сеть и битые файлы
            log.warning(f"{r.ad_id}/{r.position}: {e}")
            rows.append(dict(ad_id=r.ad_id, position=r.position, url=r.url,
                             path="", http_status=-1, width="", height="", bytes="",
                             fetched_at=datetime.now().isoformat(timespec="seconds")))
            fails += 1
            streak += 1

        if streak >= MAX_CONSECUTIVE_FAILS:
            log.error("Стоп: подряд слишком много сбоев — продолжим позже.")
            break
        if i % 50 == 0:
            append_manifest(rows); rows = []      # пишем порциями, чтобы не терять при обрыве
            log.info(f"  {i}/{len(targets)}: скачано {ok}, ошибок {fails}")
        pacing.polite_sleep(i, DELAY_RANGE, log, break_every=BREAK_EVERY)

    append_manifest(rows)
    charge_budget("cdn", ok + fails)
    mb = sum(int(x["bytes"]) for x in rows if x.get("bytes")) / 1e6
    log.info(f"Готово: скачано {ok} ({mb:.1f} МБ), ошибок {fails}. "
             f"Всего на диске: {sum(1 for _ in PHOTO_DIR.rglob('*.jpg'))} файлов")


if __name__ == "__main__":
    main()
