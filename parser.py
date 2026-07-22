"""
Kolesa.kz parser — Job 1 (сбор RAW-данных, город Алматы).

Принципы (это важно понимать, а не просто копировать):
  1. RAW-слой ничего не фильтрует и не чистит. Пишем ВСЁ как есть.
     Чистка — это Job 2, отдельный скрипт. Так данные воспроизводимы:
     ошиблись в чистке → перезапустили Job 2, ничего не потеряли.
  2. Дедупликация только по ad_id — единственная логика, которая
     допустима в RAW-слое (иначе файл раздуется повторами).
  3. Вежливый скрейпинг = защита от бана. Мы не «атакуем» сайт,
     мы имитируем одного неторопливого человека: паузы, перерывы,
     один запуск в день, остановка при первых признаках блокировки.

Запуск:  python parser.py
Зависимости:  pip install playwright beautifulsoup4
              playwright install chromium
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "parser.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import asyncio
import csv
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from db import upsert

# ─── Настройки ────────────────────────────────────────────────────────────────
OUTPUT_CSV    = "data/raw/raw_data.csv"    # «паспорт» объявления: первая встреча
SIGHTINGS_CSV = "data/raw/sightings.csv"   # журнал наблюдений: каждая встреча каждый день
LOG_FILE      = "logs/parser.log"
STATE_FILE = "browser_state.json"   # cookies между запусками → выглядим как
                                    # постоянный посетитель, а не новый бот

BASE_URL = "https://kolesa.kz"

# Категории = сегменты листинга. Зачем: kolesa показывает ограниченное число
# страниц в одном листинге, а сортировка «по дате» плывёт между запросами.
# Разбив листинг на непересекающиеся сегменты (по цене), мы:
#   а) достаём больше объявлений, чем даёт одна «лента»;
#   б) каждый сегмент короче → меньше дублей и пропусков при листании.
# Сегменты НЕ пересекаются (price[to] включительно, следующий from = to+1).
CATEGORIES = [
    ("almaty_do_3m",   "/cars/almaty/?price[to]=3000000"),
    ("almaty_3_7m",    "/cars/almaty/?price[from]=3000001&price[to]=7000000"),
    ("almaty_7_15m",   "/cars/almaty/?price[from]=7000001&price[to]=15000000"),
    ("almaty_15m_up",  "/cars/almaty/?price[from]=15000001"),
]

# Глубина листания на сегмент. ×4 сегмента ×~20 карточек = объём/день:
#   10 → ~800 (полный сбор), 3 → ~240, 2 → ~160.
# Настраивается через .env (KOLESA_MAX_PAGES) без правки кода — удобно
# временно снизить нагрузку/риск бана, пока догоняем backlog обогащения.
# Дефолт 10 (публичный репо не меняется); .env грузится через db→config.
# ВАЖНО: новые объявления кучкуются на первых страницах, так что 2-3
# страницы ловят почти весь свежак; глубокие страницы — это в основном
# ПОВТОРНЫЕ встречи уже собранного (sightings), не новьё.
MAX_PAGES_PER_CATEGORY = int(os.getenv("KOLESA_MAX_PAGES", "10"))
DELAY_MIN, DELAY_MAX   = 3.0, 7.0    # пауза между страницами, сек
COFFEE_BREAK_EVERY     = 5           # каждые N страниц — длинная пауза
COFFEE_BREAK_RANGE     = (20, 45)    # длительность длинной паузы, сек
MAX_CONSECUTIVE_FAILS  = 3           # предохранитель: 3 сбоя подряд → стоп
HEADLESS = True
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)

FIELDS = [
    # идентификация
    "ad_id", "url", "title", "brand", "model",
    # цена и характеристики машины
    "price_tenge", "year", "mileage_km", "engine_volume", "engine_type",
    "transmission", "body_type", "condition",
    # география и текст
    "city", "description",
    # сигналы для антифрода и анализа
    "photos_count", "photo_url", "views_count", "posted_date",
    "labels", "is_vip", "has_monthly_price",
    # техническое
    "category", "scraped_at",
]

# Журнал наблюдений: только то, что МЕНЯЕТСЯ день ото дня.
# Статичные характеристики (год, кузов, КПП) лежат в raw_data.csv —
# дублировать их каждый день = раздувать файл без пользы (нормализация).
SIGHTING_FIELDS = ["ad_id", "seen_date", "price_tenge", "views_count",
                   "is_vip", "category"]

# Таблица фото: строка на КАЖДОЕ фото (long format). Число фото у объявлений
# разное, поэтому колонки photo1..photoN — плохая схема (пустые ячейки,
# ломается при выходе за N), а склейка URL в одну ячейку — боль при разборе.
PHOTOS_CSV = "data/raw/photos.csv"
PHOTO_FIELDS = ["ad_id", "position", "url"]

# Замена суффикса размера превью на -full даёт полноразмерное фото
# (проверено запросами к CDN: -full существует, без суффикса — 404).
_SIZE_SUFFIX = re.compile(r"-\d+x\d+\.(jpg|webp)$")


def to_full_size(url: str) -> str:
    return _SIZE_SUFFIX.sub(r"-full.\1", url or "")


def extract_photo_urls(card) -> list[str]:
    """Все превью карточки (главное фото + подгружаемые <template>),
    приведённые к полному размеру, без дублей, в исходном порядке.

    Фильтр /static/: карточка содержит не только фото машины, но и
    служебные картинки вёрстки — иконку-бейдж (m.kolesa.kz/static/
    mobile/images/app/report/advert/badge.png) и заглушки «нет фото»
    (/static/frontend/images/stubs/noPhoto_*.svg). Это НЕ фото
    объявления: бейдж одинаковый у всех карточек и, попав в
    photo_dedup, «совпал» бы у сотен объявлений, завалив детектор
    ложными парами. Реальные фото живут на CDN (kcdn.kz) и путь
    /static/ не содержат — отсекаем по нему, не завязываясь на хост."""
    urls, seen = [], set()
    for img in card.select("img[src]"):
        src = img["src"] or ""
        if "/static/" in src:
            continue
        u = to_full_size(src)
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def init_photos(path: str):
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PHOTO_FIELDS).writeheader()


def append_photos(path: str, ad_id: str, urls: list[str]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PHOTO_FIELDS)
        for i, u in enumerate(urls, 1):
            w.writerow({"ad_id": ad_id, "position": i, "url": u})


def init_sightings(path: str):
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SIGHTING_FIELDS).writeheader()


def append_sighting(path: str, row: dict, today: str):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=SIGHTING_FIELDS).writerow({
            "ad_id": row["ad_id"],
            "seen_date": today,
            "price_tenge": row["price_tenge"],
            "views_count": row["views_count"],
            "is_vip": row["is_vip"],
            "category": row["category"],
        })


def load_today_sightings(path: str, today: str) -> set:
    """Чтобы при повторном запуске в тот же день не задвоить наблюдения
    (идемпотентность в пределах дня)."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["ad_id"] for r in csv.DictReader(f)
                if r.get("seen_date") == today}


# Известные бренды из 2+ слов — чтобы правильно разрезать title на brand/model.
MULTIWORD_BRANDS = [
    "Mercedes-Benz", "Land Rover", "Alfa Romeo", "Great Wall", "Aston Martin",
    "Rolls-Royce", "SsangYong", "ВАЗ (Lada)", "Иж",
]


# ─── Работа с CSV ─────────────────────────────────────────────────────────────
def init_csv(path: str):
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        return
    # Защита от дрейфа схемы: если шапка существующего файла не совпадает
    # с текущим FIELDS, дописывание строк МОЛЧА сдвинет значения по
    # колонкам (цена окажется в столбце года). Тихая порча данных хуже
    # падения — поэтому падаем громко и объясняем, что делать.
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != FIELDS:
        raise SystemExit(
            f"СХЕМА ИЗМЕНИЛАСЬ: шапка {path} не совпадает с текущим FIELDS.\n"
            f"  в файле : {header}\n"
            f"  ожидаем : {FIELDS}\n"
            f"Переименуй старый файл (например, {path}.old) и запусти снова.")


def append_row(path: str, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def load_passports(path: str) -> dict:
    """Загружаем ВСЕ паспорта в память (dict: ad_id → строка).
    Зачем целиком, а не только id: паспорта, собранные из VIP-карточек,
    неполные (нет пробега/кузова). Когда VIP-статус у объявления кончается,
    оно приходит обычной карточкой со всеми полями — и мы ДОЗАПОЛНЯЕМ дыры."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        passports = {row["ad_id"]: row for row in csv.DictReader(f)}
    log.info(f"Уже собрано {len(passports)} объявлений")
    return passports


_MISSING = {None, "", "nan", "None"}


def upgrade_passport(stored: dict, fresh: dict) -> bool:
    """Заполняет пустые поля паспорта данными из свежей карточки.
    Возвращает True, если что-то дозаполнили. Существующие значения
    НЕ перезаписываем (кроме случая VIP→обычная для photo/vip-полей):
    паспорт — про неизменные свойства машины."""
    changed = False
    for f in ["mileage_km", "engine_volume", "engine_type", "transmission",
              "body_type", "condition", "description", "labels", "city"]:
        old = str(stored.get(f, "")).strip()
        new = fresh.get(f)
        if old in _MISSING and new not in (None, ""):
            stored[f] = new
            changed = True
    # больше фоток / полноразмерный url — берём лучшее
    old_cnt = clean_int(str(stored.get("photos_count") or "")) or 0
    if fresh["photos_count"] > old_cnt:
        stored["photos_count"] = fresh["photos_count"]
        changed = True
    if "-full." in str(fresh.get("photo_url", "")) and \
            "-full." not in str(stored.get("photo_url", "")):
        stored["photo_url"] = fresh["photo_url"]
        changed = True
    return changed


def rewrite_passports(path: str, passports: dict):
    """Атомарная перезапись: пишем во временный файл, потом переименовываем.
    Если скрипт умрёт посреди записи — старый файл останется целым
    (rename на одном диске атомарен: файл либо старый, либо новый,
    «наполовину записанного» состояния не бывает)."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in passports.values():
            w.writerow(row)
    Path(tmp).replace(path)


# Поля паспорта, которые upgrade_passport() реально может дозаполнить
# (см. саму функцию) — используются как update_cols при батч-UPDATE в Postgres.
UPGRADE_FIELDS = ["mileage_km", "engine_volume", "engine_type", "transmission",
                  "body_type", "condition", "description", "labels", "city",
                  "photos_count", "photo_url"]

# Колонки, которые в Postgres целочисленные (INTEGER/BIGINT/SMALLINT).
# Паспорта из CSV приходят строками, и после pandas-round-trip в CSV
# целые пишутся как "50.0" — Postgres такую строку в INTEGER не примет.
# Приводим через int(float(...)), чтобы и "50", и "50.0", и 50 легли.
PG_INT_FIELDS = {"price_tenge", "year", "mileage_km", "photos_count",
                 "views_count", "is_vip", "has_monthly_price"}


def _pg_value(col: str, v):
    """Готовит значение к вставке в Postgres: пусто → NULL; целочисленные
    колонки → int (терпит '50', '50.0', 50, 50.0)."""
    if v is None or v == "":
        return None
    if col in PG_INT_FIELDS:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    return v


def flush_postgres(pg_new_ads: list[dict], pg_new_photos: list[dict],
                    pg_sightings: list[dict], pg_upgraded: list[dict]):
    """Двойная запись (пилот миграции на Postgres, см. план): CSV остаётся
    источником истины весь пилот, сбой записи в БД не должен ронять
    прогон — это дорогие (сетевые) данные, терять их нельзя. Батчами, а
    не построчно — иначе на каждую карточку листинга уходил бы отдельный
    round-trip к БД поверх и так небыстрого похода за страницей."""
    # Готовим значения к вставке: пусто → NULL, целочисленные колонки →
    # int (см. _pg_value). pg_upgraded приходит из CSV-паспортов строками,
    # где после pandas-round-trip целые записаны как "50.0" — без
    # приведения Postgres роняет батч с invalid input syntax.
    def clean_rows(rows):
        return [{k: _pg_value(k, v) for k, v in row.items()} for row in rows]

    try:
        upsert("raw_ads", clean_rows(pg_new_ads), ["ad_id"])
        upsert("photos", pg_new_photos, ["ad_id", "position"])
        upsert("sightings", clean_rows(pg_sightings), ["ad_id", "seen_date"])
        upsert("raw_ads", clean_rows(pg_upgraded), ["ad_id"],
               update_cols=UPGRADE_FIELDS)
    except Exception as e:
        log.warning(f"Postgres dual-write не удался: {e}")


# ─── Извлечение чисел и характеристик из текста ──────────────────────────────
def clean_int(raw: str) -> int | None:
    digits = re.sub(r"\D", "", raw or "")
    return int(digits) if digits else None


def split_brand_model(title: str) -> tuple[str | None, str | None]:
    """'Kia K7' -> ('Kia', 'K7'); 'Mercedes-Benz GLS 450' -> ('Mercedes-Benz', 'GLS 450')."""
    if not title:
        return None, None
    for b in MULTIWORD_BRANDS:
        if title.startswith(b):
            return b, title[len(b):].strip() or None
    parts = title.split(maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def parse_spec_line(text: str) -> dict:
    """
    Разбирает строку характеристик вида:
    '2014 г., Б/у седан, 3 л, бензин, КПП автомат, с пробегом 170 000 км, <текст продавца>'
    У VIP-карточек она короче: '2026 г., 1.5 л, гибрид, КПП автомат' (без пробега).
    """
    r = {"year": None, "engine_volume": None, "engine_type": None,
         "transmission": None, "body_type": None, "condition": None,
         "mileage_km": None, "description": ""}
    if not text:
        return r
    low = text.lower()

    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        r["year"] = int(m.group())

    # Пробег: сначала строгий формат «с пробегом 170 000 км», затем
    # fallback — просто «260 000 км» (kolesa использует оба варианта;
    # привязка только к первому теряла ~40% пробегов).
    m = re.search(r"с пробегом\s+([\d\s\u00a0]+)\s*км", low)
    if not m:
        m = re.search(r"(?:^|,)\s*([\d][\d\s\u00a0]*)\s*км\b", low)
    if m:
        r["mileage_km"] = int(re.sub(r"[\s\u00a0]", "", m.group(1)))
    elif "без пробега" in low:
        r["mileage_km"] = 0

    m = re.search(r"([\d]+(?:[.,]\d+)?)\s*л\b", text)
    if m:
        r["engine_volume"] = float(m.group(1).replace(",", "."))

    # ВАЖНО: порядок — от специфичного к общему. Иначе подстрока «бензин»
    # внутри «газ-бензин» сработает первой и мы получим неверную метку.
    for fuel in ["газ-бензин", "гибрид", "электричество", "электро",
                 "дизель", "газ", "бензин"]:
        if fuel in low:
            r["engine_type"] = "электро" if fuel == "электричество" else fuel
            break

    for kpp, canon in [("автомат", "автомат"), ("вариатор", "вариатор"),
                       ("робот", "робот"), ("механи", "механика")]:
        if kpp in low:
            r["transmission"] = canon
            break

    # «кроссовер» — самый частый кузов на kolesa, в старом списке его не было!
    for body in ["кроссовер", "внедорожник", "седан", "хэтчбек", "лифтбек",
                 "универсал", "минивэн", "микроавтобус", "купе", "пикап",
                 "кабриолет", "родстер", "лимузин", "фургон"]:
        if body in low:
            r["body_type"] = body
            break

    if "б/у" in low:
        r["condition"] = "б/у"
    elif "нов" in low.split(",")[0] or "новый" in low or "новая" in low:
        r["condition"] = "новый"

    # Текст продавца. Строка устроена как «спеки, ..., текст», поэтому
    # ищем последний надёжный якорь и берём всё после него. Цепочка
    # запасных якорей (fallback): пробег → КПП → топливо. Раньше якорь
    # был только один («км,») — и у строк без пробега текст продавца
    # выбрасывался целиком (~360 потерянных описаний).
    for anchor in (r"с пробегом[\d\s\u00a0]+км,?\s*",
                   r"\bкм,?\s*",
                   r"КПП\s+\S+,\s*",
                   r"(?:газ-бензин|бензин|дизель|гибрид|электро|газ),\s*"):
        m = re.search(anchor + r"(.+)$", text, re.IGNORECASE)
        if not m:
            continue
        candidate = m.group(1).strip()
        # ранний якорь мог захватить хвост спеков («КПП автомат») —
        # такое не текст продавца, отвергаем и НЕ пробуем более
        # поздние якоря (текста в строке просто нет)
        if re.match(r"^(КПП\b|с пробегом\b)", candidate, re.IGNORECASE):
            break
        r["description"] = candidate[:500]
        break

    return r


# ─── Разбор карточек листинга ────────────────────────────────────────────────
def txt(node) -> str:
    return node.get_text(" ", strip=True) if node else ""


def parse_cards(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for card in soup.select(".js__a-card"):
        ad_id = card.get("data-id", "")
        if not ad_id:
            continue

        is_vip = "vip-card" in " ".join(card.get("class", []))
        prefix = "vip-card" if is_vip else "a-card"

        # URL без трекинговых параметров (?search_id=...)
        link = card.select_one("a[href*='/a/show/']")
        href = re.sub(r"\?.*", "", link["href"]) if link else f"/a/show/{ad_id}"
        url = BASE_URL + href

        title = txt(card.select_one(f".{prefix}__title")).replace(
            "Добавить в избранное", "").strip() or None
        brand, model = split_brand_model(title or "")

        price = clean_int(txt(card.select_one(f".{prefix}__price")))

        # У VIP-карточек характеристики лежат в .vip-card__description —
        # раньше мы брали только alt картинки и теряли КПП/топливо/объём.
        spec = txt(card.select_one(f".{prefix}__description"))
        if not spec:  # запасной вариант — alt картинки
            img_alt = card.select_one("img[alt]")
            spec = img_alt["alt"] if img_alt else ""
        parsed = parse_spec_line(spec)

        city = txt(card.select_one(f".{prefix}__region")
                   or card.select_one("[data-test='region']")) or None

        # ── Новые поля-сигналы ────────────────────────────────────────────
        views = clean_int(txt(card.select_one(".nb-views")))          # просмотры
        posted = txt(card.select_one(f".{prefix}__date")
                     or card.select_one(".a-card__param--date")) or None
        labels = "|".join(txt(x) for x in card.select(".a-label__text")) or None
        has_monthly = "/мес" in card.get_text()   # рассрочка/кредит от дилера

        # Фото: собираем ВСЕ превью карточки и переводим в полный размер.
        # Счётчик из листинга — нижняя граница: сайт кладёт в карточку
        # максимум ~5 превью, даже если фоток больше.
        photo_urls = extract_photo_urls(card)
        cnt = clean_int(txt(card.select_one(".thumb-gallery__count")))
        photos_count = max(cnt or 0, len(photo_urls), 1)
        photo_url = photo_urls[0] if photo_urls else ""

        if not title or not price:
            continue  # карточка-заглушка/реклама

        results.append({
            "ad_id": ad_id, "url": url, "title": title,
            "brand": brand, "model": model,
            "price_tenge": price,
            "year": parsed["year"],
            "mileage_km": parsed["mileage_km"],
            "engine_volume": parsed["engine_volume"],
            "engine_type": parsed["engine_type"],
            "transmission": parsed["transmission"],
            "body_type": parsed["body_type"],
            "condition": parsed["condition"],
            "city": city,
            "description": parsed["description"],
            "photos_count": photos_count,
            "photo_url": photo_url,
            "_photo_urls": photo_urls,  # служебное поле: уйдёт в photos.csv
            "views_count": views,
            "posted_date": posted,
            "labels": labels,
            "is_vip": int(is_vip),
            "has_monthly_price": int(has_monthly),
            "category": category,
            "scraped_at": datetime.now().isoformat(timespec="seconds"),
        })

    return results


# ─── Сетевая часть: вежливо и с предохранителями ─────────────────────────────
async def human_pause(page):
    """Имитация чтения страницы: скролл + случайная пауза."""
    await asyncio.sleep(random.uniform(0.8, 1.6))
    await page.evaluate(f"window.scrollBy(0, {random.randint(400, 900)})")
    await asyncio.sleep(random.uniform(0.5, 1.2))


async def get_html(page, url: str, retries: int = 3) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await human_pause(page)
            return await page.content()
        except PWTimeout:
            log.warning(f"Таймаут [{attempt}/{retries}]: {url}")
        except Exception as e:
            log.error(f"Ошибка [{attempt}/{retries}]: {e}")
        if attempt < retries:
            # Экспоненциальный backoff: 5с, 10с, 20с — каждый повтор ждём
            # вдвое дольше. Если сайт «устал» от нас, частые ретраи только
            # ухудшают ситуацию; растущая пауза даёт ему остыть.
            await asyncio.sleep(5 * (2 ** (attempt - 1)))
    return None


def looks_blocked(html: str) -> bool:
    """
    ВАЖНО: маркеры должны встречаться ТОЛЬКО на странице блокировки.
    Слово «captcha» есть в футере каждой обычной страницы kolesa
    («Защищено reCAPTCHA») — из-за него детектор ловил ложные
    срабатывания на здоровых страницах. Поэтому:
      1) маркеры — только специфичные фразы страниц блокировки/логина;
      2) страница с карточками объявлений блокировкой не считается
         в принципе (структурная проверка сильнее текстовой).
    """
    if "js__a-card" in html:          # есть карточки → точно не блокировка
        return False
    markers = ["Вход в личный кабинет", "passport/login",
               "Доступ ограничен", "Too Many Requests",
               "Подтвердите, что вы не робот"]
    return any(m in html for m in markers)


async def run():
    init_csv(OUTPUT_CSV)
    init_sightings(SIGHTINGS_CSV)
    init_photos(PHOTOS_CSV)
    passports = load_passports(OUTPUT_CSV)
    today = datetime.now().date().isoformat()
    seen_today = load_today_sightings(SIGHTINGS_CSV, today)
    total_saved = 0
    total_upgraded = 0
    total_sightings = 0
    consecutive_fails = 0
    pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded = [], [], [], []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        # storage_state: сохраняем cookies между запусками. Для сайта мы —
        # один и тот же посетитель, который заходит раз в день. Это гораздо
        # менее подозрительно, чем «свежий» браузер без истории каждый раз.
        state = STATE_FILE if Path(STATE_FILE).exists() else None
        context = await browser.new_context(
            storage_state=state,
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
            locale="ru-RU",
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = await context.new_page()

        # Прогрев: заходим на главную, как обычный человек
        log.info("Прогрев сессии...")
        try:
            await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            await human_pause(page)
        except Exception as e:
            log.warning(f"Прогрев не удался: {e}")

        pages_done = 0
        for cat_name, cat_path in CATEGORIES:
            log.info(f"── Категория: {cat_name} ──")
            for page_num in range(1, MAX_PAGES_PER_CATEGORY + 1):
                # kolesa канонизирует URL: «?page=1» редиректится на адрес
                # БЕЗ параметра page → петля редиректов (ERR_TOO_MANY_REDIRECTS).
                # Поэтому первую страницу запрашиваем без page.
                if page_num == 1:
                    url = f"{BASE_URL}{cat_path}"
                else:
                    sep = "&" if "?" in cat_path else "?"
                    url = f"{BASE_URL}{cat_path}{sep}page={page_num}"
                log.info(f"[{cat_name}] стр. {page_num}: {url}")

                html = await get_html(page, url)

                if html is None or looks_blocked(html or ""):
                    consecutive_fails += 1
                    log.error(f"Сбой/блокировка ({consecutive_fails}/"
                              f"{MAX_CONSECUTIVE_FAILS})")
                    if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                        # Предохранитель (circuit breaker): лучше потерять
                        # один день сбора, чем IP на неделю.
                        log.error("Стоп: слишком много сбоев подряд. "
                                  "Завтра запуск продолжит с того же места.")
                        if total_upgraded:
                            rewrite_passports(OUTPUT_CSV, passports)
                        flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)
                        await context.storage_state(path=STATE_FILE)
                        await browser.close()
                        sys.exit(1)
                    await asyncio.sleep(60)
                    continue
                consecutive_fails = 0

                cards = parse_cards(html, cat_name)
                if not cards:
                    log.info(f"[{cat_name}] карточек нет — конец сегмента")
                    break

                new = 0
                for row in cards:
                    photo_urls = row.pop("_photo_urls", [])

                    # 1) Журнал наблюдений — пишем ВСЕГДА: из повторов
                    #    Job 2 соберёт историю цены и days_on_market.
                    if row["ad_id"] not in seen_today:
                        append_sighting(SIGHTINGS_CSV, row, today)
                        pg_sightings.append({
                            "ad_id": row["ad_id"], "seen_date": today,
                            "price_tenge": row["price_tenge"],
                            "views_count": row["views_count"],
                            "is_vip": row["is_vip"], "category": row["category"],
                        })
                        seen_today.add(row["ad_id"])
                        total_sightings += 1

                    # 2) Паспорт: новый — добавляем; известный — пробуем
                    #    дозаполнить дыры (VIP-карточки дают неполные
                    #    паспорта, обычные — полные).
                    if row["ad_id"] in passports:
                        if upgrade_passport(passports[row["ad_id"]], row):
                            total_upgraded += 1
                            pg_upgraded.append(dict(passports[row["ad_id"]]))
                        continue
                    row = {k: v for k, v in row.items() if k in FIELDS}
                    passports[row["ad_id"]] = row
                    append_row(OUTPUT_CSV, row)
                    append_photos(PHOTOS_CSV, row["ad_id"], photo_urls)
                    pg_new_ads.append(dict(row))
                    pg_new_photos.extend({"ad_id": row["ad_id"], "position": i, "url": u}
                                         for i, u in enumerate(photo_urls, 1))
                    total_saved += 1
                    new += 1
                log.info(f"  карточек: {len(cards)}, новых: {new}, "
                         f"наблюдений сегодня: {total_sightings}, "
                         f"всего объявлений: {total_saved}")

                pages_done += 1
                if pages_done % COFFEE_BREAK_EVERY == 0:
                    brk = random.uniform(*COFFEE_BREAK_RANGE)
                    log.info(f"  ☕ длинная пауза {brk:.0f}s")
                    await asyncio.sleep(brk)
                else:
                    await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        await context.storage_state(path=STATE_FILE)
        await browser.close()

    if total_upgraded:
        rewrite_passports(OUTPUT_CSV, passports)
        log.info(f"Дозаполнено паспортов: {total_upgraded}")
    flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)

    log.info(f"\n{'=' * 50}\nГотово! Новых: {total_saved}, "
             f"дозаполнено: {total_upgraded}, "
             f"наблюдений: {total_sightings} → {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(run())