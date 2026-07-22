# -*- coding: utf-8 -*-
"""
Job: фото-дедуп (photo_dedup.py) — ищем ворованные/переиспользованные фото.

Зачем: детектор в clean.py видит только цену и текст. Он не отличает
честную развалюху (дёшево, потому что реально убитая) от мошенника,
укравшего чужое фото под другую машину. Разница видна ИМЕННО на
фото — значит, надо сравнивать фото между объявлениями.

Метод — perceptual hash (pHash), не ML: две визуально похожие
картинки дают близкие хэши, а не пиксель-в-пиксель разные числа
(в отличие от обычного md5/sha, где 1 бит разницы = случайный хэш).
Совпадение хэша у ДВУХ РАЗНЫХ объявлений само по себе не подозрительно
(дилер легитимно перезаливает те же фото под ту же машину — это уже
ловит add_duplicate_flags() в clean.py). Подозрительно, когда одно и
то же фото — под ЗАЯВЛЕННО РАЗНЫМИ машинами (марка/модель, год,
цена) — это и есть сигнал «фото одно, машины разные».

Приоритизация (как в enrich.py): сначала фото подозрительных
объявлений (из clean_data.csv пасс 1), затем — обложка (position=1)
у всех остальных раньше, чем более глубокие фото у кого-то одного:
на старте нужнее широкий охват, а не глубина по паре объявлений.

Запуск: python photo_dedup.py   (после enrich.py, перед clean.py пасс 2)
Выход:  photo_hashes.csv (append-only кэш, по строке на фото)
        photo_duplicates.csv (пересобирается каждый прогон, как
        suspicious_sorted.csv — список пар «одно фото, разные машины»)
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "photo_dedup.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import csv
import io
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image
from sqlalchemy import text

from db import get_engine, upsert

HASHES_CSV      = "data/enriched/photo_hashes.csv"
DUPLICATES_CSV  = "data/enriched/photo_duplicates.csv"
LOG_FILE        = "logs/photo_dedup.log"

MAX_PER_RUN            = 300    # дневной бюджет запросов к CDN
DELAY_RANGE            = (1.5, 3.0)   # CDN легче основного сайта, но не долбим без лимита
MAX_CONSECUTIVE_FAILS  = 5
MAX_IMAGE_BYTES        = 5_000_000    # защита от нештатно огромного файла

# Порог Хэмминга ОТКАЛИБРОВАН на реальных парах (2026-07-20): настоящий
# дубль (одни и те же файлы у двух объявлений OMODA) дал ровно 0, а ВСЕ
# пары с расстоянием 2–4 оказались «одна дилерская студия — разные
# машины»: поворотный круг/фон/ракурс одинаковые, и низкие частоты
# pHash (64 бит) забиты композицией, а не машиной. Причём ложные пары
# совпадали и по 2-3 фото сразу (студия снимает все машины с одних
# ракурсов) — так что «требовать несколько совпадений» не спасает,
# спасает только точное равенство хэша. Если появятся воры, которые
# пережимают/кадрируют фото (hamming 1-4 у НАСТОЯЩИХ дублей), путь
# апгрейда — доверификация кандидатов 256-битным хэшем по паре скачек,
# а не откат этого порога.
HAMMING_THRESHOLD  = 0
PRICE_DIFF_RATIO   = 0.15    # цены расходятся более чем на 15% — «разные деньги»
YEAR_DIFF_MIN      = 2       # год расходится на ≥2 — «разные машины»

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
}

HASH_FIELDS = ["ad_id", "position", "url", "phash", "fetched_at", "http_status"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_hashes() -> pd.DataFrame:
    """Кэш уже посчитанных хэшей — резюмируемо, тот же паттерн, что
    load_done() в enrich.py: набор URL, которые не нужно трогать снова."""
    if not Path(HASHES_CSV).exists():
        with open(HASHES_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HASH_FIELDS).writeheader()
        return pd.DataFrame(columns=HASH_FIELDS)
    return pd.read_csv(HASHES_CSV, dtype={"ad_id": str})


def append_hash(row: dict):
    with open(HASHES_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HASH_FIELDS, extrasaction="ignore").writerow(row)

    # Двойная запись (пилот миграции на Postgres, см. план): CSV остаётся
    # источником истины, сбой записи в БД не должен ронять прогон —
    # хэширование стоит дорого (сеть+CPU), терять его нельзя.
    try:
        # "" → NULL (см. аналогичный комментарий в enrich.py):
        # пустой phash у битой/несжатой картинки в БД должен быть NULL
        pg_row = {k: (v if v != "" else None) for k, v in row.items()}
        upsert("photo_hashes", [pg_row], ["ad_id", "position"])
    except Exception as e:
        log.warning(f"Postgres upsert не удался для {row.get('url')}: {e}")


def pick_targets(done_urls: set) -> pd.DataFrame:
    """Очередь на хэширование: подозрительные объявления — целиком (все
    их фото), у остальных — только обложка (position=1) сначала, глубже
    (position>=2) — уже по остаточному бюджету. Внутри uses MAX_PER_RUN."""
    engine = get_engine()
    photos = pd.read_sql("SELECT ad_id, position, url FROM photos", engine,
                          dtype={"ad_id": str})
    photos = photos[~photos["url"].isin(done_urls)]
    # «нет фото» отдаётся как заглушка-стаб (протокол-относительный URL
    # без реальной картинки, напр. .../stubs/noPhoto_160x120.svg) — если
    # хэшировать её как обычное фото, она будет ОДИНАКОВОЙ у всех таких
    # объявлений и завалит find_cross_car_duplicates ложными совпадениями.
    # Реальные фото у kolesa всегда абсолютные https-URL — фильтр по
    # префиксу отсекает заглушки, не завязываясь на конкретное имя файла.
    photos = photos[photos["url"].fillna("").str.startswith("http")]

    clean = pd.read_sql("SELECT ad_id, is_suspicious FROM clean_data", engine,
                         dtype={"ad_id": str})
    photos = photos.merge(clean, on="ad_id", how="left")
    photos["is_suspicious"] = photos["is_suspicious"].fillna(0)

    # ~96% фото лежат на CDN (kcdn.kz), но ~4% — на m.kolesa.kz, то есть
    # на инфраструктуре ОСНОВНОГО сайта. Этот джоб работает параллельно
    # с enrich.py (который стучится на kolesa.kz) — см. run_all.py, —
    # поэтому kolesa-хосты отодвигаем в конец очереди: CDN-бэклог
    # разгребается первым, и в параллельном окне на основной сайт
    # photo_dedup почти не ходит.
    photos["_kolesa_host"] = photos["url"].str.contains(r"//[^/]*kolesa\.kz/")

    # приоритет: подозрительные впереди; затем CDN раньше kolesa-хостов;
    # внутри группы — обложка (position=1) раньше
    photos = photos.sort_values(
        ["is_suspicious", "_kolesa_host", "position"],
        ascending=[False, True, True])
    return photos.head(MAX_PER_RUN)


def fetch_phash(url: str, session: requests.Session) -> tuple[str | None, int]:
    """Возвращает (phash-строка либо None, http_status). Чистого разделения
    сеть/вычисление здесь нет смысла делать — вся функция и так тонкая
    обёртка над двумя вызовами (requests + imagehash)."""
    resp = session.get(url, headers=HEADERS, timeout=15,
                        stream=True)
    if resp.status_code != 200:
        return None, resp.status_code
    content = resp.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    if len(content) > MAX_IMAGE_BYTES:
        log.warning(f"{url}: файл больше {MAX_IMAGE_BYTES} байт, пропуск")
        return None, resp.status_code
    img = Image.open(io.BytesIO(content))
    return str(imagehash.phash(img)), resp.status_code


def collect_hashes():
    done = load_hashes()
    done_urls = set(done["url"])
    targets = pick_targets(done_urls)
    log.info(f"К хэшированию: {len(targets)} (уже готово: {len(done_urls)})")

    session = requests.Session()
    fails = 0
    for i, row in enumerate(targets.itertuples(), 1):
        try:
            phash, http_status = fetch_phash(row.url, session)
        except requests.RequestException as e:
            log.warning(f"{row.url}: сетевая ошибка {e}")
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Стоп: сбои подряд — продолжим завтра.")
                sys.exit(1)
            time.sleep(30)
            continue
        except Exception as e:
            log.warning(f"{row.url}: не удалось разобрать изображение: {e}")
            phash, http_status = None, -1

        # 429 = CDN просит притормозить: раньше это НЕ считалось сбоем и
        # предохранитель на него не срабатывал — джоб долбил бы сквозь
        # rate-limit. Теперь тормозим и считаем к предохранителю.
        if http_status == 429:
            log.warning("429: пауза 120с")
            time.sleep(120)
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("Стоп: 429 подряд — CDN лимитирует, продолжим позже.")
                sys.exit(1)
            continue   # не записываем битый хэш, повторим в другой раз

        fails = 0
        append_hash({
            "ad_id": row.ad_id, "position": row.position, "url": row.url,
            "phash": phash or "", "http_status": http_status,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        })
        if i % 50 == 0:
            log.info(f"  {i}/{len(targets)}")
        time.sleep(random.uniform(*DELAY_RANGE))

    log.info(f"Готово → {HASHES_CSV}")


# ─── Группировка: чистая функция, без сети — тестируется без единого запроса ─
def find_cross_car_duplicates(hashes: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """
    Ищем пары объявлений с одинаковым/близким фото, но разными машинами.

    Шаг 1 — точные совпадения: группируем по хэшу (dict), это ловит
    подавляющее большинство реальных случаев (скопированный файл
    идентичен побитово, а не просто «похож»).

    Шаг 2 — почти-совпадения (пересжатое/чуть обрезанное фото): полный
    перебор всех пар был бы O(n²) и не нужен — бакетируем по первым
    16 битам хэша (LSH-приём: похожие хэши с большой вероятностью
    совпадают хотя бы в части бит) и сравниваем по Хэммингу только
    внутри бакета. Компромисс осознанный: часть почти-дублей, у
    которых различие пришлось именно на первые биты, не поймаем —
    это эвристика, не гарантия, и это ок для сигнала «стоит посмотреть
    глазами», а не для автоматического бана.

    «Разные машины» — исключаем легитимные дилерские перезаливы
    (та же марка+модель+год+похожая цена уже ловится
    add_duplicate_flags() в clean.py, задваивать сигнал не нужно):
    флагуем пару, только если отличается марка/модель, ИЛИ год
    разошёлся на ≥2, ИЛИ цена — более чем на 15%.
    """
    valid = hashes[hashes["phash"].notna() & (hashes["phash"] != "")].copy()
    if valid.empty:
        return pd.DataFrame(columns=[
            "ad_id_a", "ad_id_b", "hamming_distance",
            "model_key_a", "price_a", "year_a",
            "model_key_b", "price_b", "year_b"])

    keep = ["brand", "model", "year", "price_tenge"]
    for extra in ("condition", "labels"):
        if extra in clean.columns:
            keep.append(extra)
    cars = clean.set_index("ad_id")[keep]
    cars["model_key"] = (cars["brand"].fillna("") + " " + cars["model"].fillna("")).str.strip()

    # Дилерский иммунитет (тот же, что у possible_repost в clean.py):
    # официальный дилер вешает ОДНО пресс-фото производителя на разные
    # комплектации одной модели (GS3 GB/GL, S5 Life/Prestige) — точное
    # совпадение фото, но это НЕ кража. Исключаем пару, только если ОБА —
    # новый/дилер. Кражей остаётся дилер↔частник (украли пресс-фото под
    # фейковый б/у) — там хотя бы одна сторона не дилер, флаг сохраняется.
    _cond = cars["condition"] if "condition" in cars.columns else pd.Series("", index=cars.index)
    _lab = cars["labels"] if "labels" in cars.columns else pd.Series("", index=cars.index)
    cars["dealer_new"] = (_cond.eq("новый").fillna(False)
                          | _lab.fillna("").str.contains("дилер|Новая", case=False))

    def both_dealer_new(a: str, b: str) -> bool:
        if a not in cars.index or b not in cars.index:
            return False
        return bool(cars.loc[a, "dealer_new"]) and bool(cars.loc[b, "dealer_new"])

    def different_cars(a: str, b: str) -> bool:
        if a not in cars.index or b not in cars.index:
            return False
        ca, cb = cars.loc[a], cars.loc[b]
        if ca["model_key"] != cb["model_key"]:
            return True
        if pd.notna(ca["year"]) and pd.notna(cb["year"]) and abs(ca["year"] - cb["year"]) >= YEAR_DIFF_MIN:
            return True
        if pd.notna(ca["price_tenge"]) and pd.notna(cb["price_tenge"]):
            hi = max(ca["price_tenge"], cb["price_tenge"])
            if hi > 0 and abs(ca["price_tenge"] - cb["price_tenge"]) / hi > PRICE_DIFF_RATIO:
                return True
        return False

    pairs = {}   # (ad_id_a, ad_id_b) -> min hamming_distance найденный

    def consider(a: str, b: str, dist: int):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if key not in pairs or dist < pairs[key]:
            pairs[key] = dist

    # Шаг 1: точные совпадения
    for _, ad_ids in valid.groupby("phash")["ad_id"]:
        uniq = sorted(set(ad_ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                consider(uniq[i], uniq[j], 0)

    # Шаг 2: почти-совпадения внутри бакетов по первым 16 битам хэша
    valid["_bucket"] = valid["phash"].str[:4]   # 4 hex-символа = 16 бит
    for _, bucket in valid.groupby("_bucket"):
        rows = list(bucket[["ad_id", "phash"]].itertuples(index=False))
        for i in range(len(rows)):
            hi = imagehash.hex_to_hash(rows[i].phash)
            for j in range(i + 1, len(rows)):
                if rows[i].ad_id == rows[j].ad_id:
                    continue
                hj = imagehash.hex_to_hash(rows[j].phash)
                dist = hi - hj
                if dist <= HAMMING_THRESHOLD:
                    consider(rows[i].ad_id, rows[j].ad_id, dist)

    records = []
    for (a, b), dist in pairs.items():
        if both_dealer_new(a, b):
            continue   # дилер↔дилер: пресс-фото между комплектациями, не кража
        if not different_cars(a, b):
            continue
        ca = cars.loc[a] if a in cars.index else None
        cb = cars.loc[b] if b in cars.index else None
        records.append({
            "ad_id_a": a, "ad_id_b": b, "hamming_distance": dist,
            "model_key_a": ca["model_key"] if ca is not None else "",
            "price_a": ca["price_tenge"] if ca is not None else None,
            "year_a": ca["year"] if ca is not None else None,
            "model_key_b": cb["model_key"] if cb is not None else "",
            "price_b": cb["price_tenge"] if cb is not None else None,
            "year_b": cb["year"] if cb is not None else None,
        })

    cols = ["ad_id_a", "ad_id_b", "hamming_distance",
            "model_key_a", "price_a", "year_a",
            "model_key_b", "price_b", "year_b"]
    # from_records([]) на пустом списке даёт DataFrame БЕЗ колонок вообще
    # (не с нужной схемой, а с нулём столбцов) — это ломает и to_sql,
    # и последующий SELECT ad_id_a, ... в clean.py. Явно фиксируем schema.
    out = pd.DataFrame.from_records(records, columns=cols) if records \
        else pd.DataFrame(columns=cols)
    if not out.empty:
        out = out.sort_values("hamming_distance")
    return out


def main():
    collect_hashes()

    engine = get_engine()
    with engine.begin() as conn:
        has_clean = conn.execute(text("SELECT to_regclass('public.clean_data')")).scalar()
    if not has_clean:
        log.warning("Таблица clean_data не найдена — группировку по машинам пропускаю.")
        pd.DataFrame(columns=["ad_id_a", "ad_id_b"]).to_csv(DUPLICATES_CSV, index=False)
        return

    # hashes — из CSV (тот же "дорогой сырой слой на время пилота", что и
    # enriched.csv в enrich.py); clean_data — disposable, только Postgres.
    hashes = pd.read_csv(HASHES_CSV, dtype={"ad_id": str})
    clean = pd.read_sql("SELECT ad_id, brand, model, year, price_tenge, condition, labels "
                        "FROM clean_data", engine, dtype={"ad_id": str})
    dups = find_cross_car_duplicates(hashes, clean)
    dups.to_csv(DUPLICATES_CSV, index=False)
    # disposable, как clean_data: TRUNCATE+заливка в одной транзакции
    with engine.begin() as conn:
        dups.to_sql("photo_duplicates", conn, if_exists="replace", index=False)
    log.info(f"Найдено пар «одно фото — разные машины»: {len(dups)} → {DUPLICATES_CSV} "
             f"и в таблицу photo_duplicates")


if __name__ == "__main__":
    main()
