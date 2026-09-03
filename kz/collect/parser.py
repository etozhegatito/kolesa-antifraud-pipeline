"""Implementation for the `kz.collect.parser` module."""

import pathlib as _p

_expected = "parser.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from kz.core.db import upsert
from kz.ops import catch_up as request_budget


OUTPUT_CSV = "data/raw/raw_data.csv"
SIGHTINGS_CSV = "data/raw/sightings.csv"
LOG_FILE = "logs/parser.log"
RUN_STATUS_FILE = "logs/parser_last_run.json"
STATE_FILE = "browser_state.json"


BASE_URL = "https://kolesa.kz"


CATEGORIES = [
    ("almaty_do_3m", "/cars/almaty/?price[to]=3000000"),
    ("almaty_3_7m", "/cars/almaty/?price[from]=3000001&price[to]=7000000"),
    ("almaty_7_15m", "/cars/almaty/?price[from]=7000001&price[to]=15000000"),
    ("almaty_15m_up", "/cars/almaty/?price[from]=15000001"),
]


MAX_PAGES_PER_CATEGORY = int(os.getenv("KOLESA_MAX_PAGES", "3"))


#


START_PAGE = int(os.getenv("KOLESA_START_PAGE", "1"))


MAX_CARDS_PER_RUN = int(os.getenv("KOLESA_MAX_CARDS", "0"))
if MAX_CARDS_PER_RUN < 0:
    raise SystemExit("KOLESA_MAX_CARDS must be >= 0")
DELAY_MIN, DELAY_MAX = 3.0, 7.0
COFFEE_BREAK_EVERY = 5
COFFEE_BREAK_RANGE = (20, 45)
MAX_CONSECUTIVE_FAILS = 3


LISTING_HEALTH_FIRST_PAGES = 3
LISTING_HEALTH_MIN_RAW_CARDS = 5
HEADLESS = True
# ──────────────────────────────────────────────────────────────────────────────


class DailyBudgetExhausted(RuntimeError):
    """Implementation of `DailyBudgetExhausted`."""


class ListingSchemaError(RuntimeError):
    """Implementation of `ListingSchemaError`."""


_run_kolesa_requests = 0


def reserve_kolesa_request() -> dict:
    """Implement `reserve_kolesa_request`."""
    global _run_kolesa_requests
    limit = request_budget.DAILY_BUDGET["kolesa"]
    if _run_kolesa_requests + 1 > limit:
        raise DailyBudgetExhausted(
            f"The current-run Kolesa limit is exhausted: {limit}/{limit}; "
            "listing collection will resume on the next run"
        )
    current = request_budget.reserve_budget("kolesa", 1, limit)
    if current is None:
        used = request_budget.load_budget_used()
        raise DailyBudgetExhausted(
            f"The rolling 24-hour Kolesa budget is exhausted: {used['kolesa']}/{limit}; "
            "listing collection will resume when the window frees up"
        )
    _run_kolesa_requests += 1
    return current


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

FIELDS = [
    "ad_id",
    "url",
    "title",
    "brand",
    "model",
    "price_tenge",
    "year",
    "mileage_km",
    "engine_volume",
    "engine_type",
    "transmission",
    "body_type",
    "condition",
    "city",
    "description",
    "photos_count",
    "photo_url",
    "views_count",
    "posted_date",
    "labels",
    "is_vip",
    "has_monthly_price",
    "category",
    "scraped_at",
]


SIGHTING_FIELDS = ["ad_id", "seen_date", "price_tenge", "views_count", "is_vip", "category"]


PHOTOS_CSV = "data/raw/photos.csv"
PHOTO_FIELDS = ["ad_id", "position", "url"]


_SIZE_SUFFIX = re.compile(r"-\d+x\d+\.(jpg|webp)$")


def to_full_size(url: str) -> str:
    return _SIZE_SUFFIX.sub(r"-full.\1", url or "")


def extract_photo_urls(card) -> list[str]:
    """Implement `extract_photo_urls`."""
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
        csv.DictWriter(f, fieldnames=SIGHTING_FIELDS).writerow(
            {
                "ad_id": row["ad_id"],
                "seen_date": today,
                "price_tenge": row["price_tenge"],
                "views_count": row["views_count"],
                "is_vip": row["is_vip"],
                "category": row["category"],
            }
        )


def load_today_sightings(path: str, today: str) -> set:
    """Implement `load_today_sightings`."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["ad_id"] for r in csv.DictReader(f) if r.get("seen_date") == today}


MULTIWORD_BRANDS = [
    "Mercedes-Benz",
    "Land Rover",
    "Alfa Romeo",
    "Great Wall",
    "Aston Martin",
    "Rolls-Royce",
    "SsangYong",
    "ВАЗ (Lada)",
    "Иж",
]


def init_csv(path: str):
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        return

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != FIELDS:
        raise SystemExit(
            f"SCHEMA CHANGED: the header in {path} does not match the current FIELDS.\n"
            f"  file     : {header}\n"
            f"  expected : {FIELDS}\n"
            f"Rename the old file (for example, {path}.old) and run again."
        )


def append_row(path: str, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def load_passports(path: str) -> dict:
    """Implement `load_passports`."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        passports = {row["ad_id"]: row for row in csv.DictReader(f)}
    log.info(f"Already collected {len(passports)} listings")
    return passports


_MISSING = {None, "", "nan", "None"}


def upgrade_passport(stored: dict, fresh: dict) -> bool:
    """Implement `upgrade_passport`."""
    changed = False
    for f in [
        "mileage_km",
        "engine_volume",
        "engine_type",
        "transmission",
        "body_type",
        "condition",
        "description",
        "labels",
        "city",
    ]:
        old = str(stored.get(f, "")).strip()
        new = fresh.get(f)
        if old in _MISSING and new not in (None, ""):
            stored[f] = new
            changed = True

    old_cnt = clean_int(str(stored.get("photos_count") or "")) or 0
    if fresh["photos_count"] > old_cnt:
        stored["photos_count"] = fresh["photos_count"]
        changed = True
    if "-full." in str(fresh.get("photo_url", "")) and "-full." not in str(
        stored.get("photo_url", "")
    ):
        stored["photo_url"] = fresh["photo_url"]
        changed = True
    return changed


def rewrite_passports(path: str, passports: dict):
    """Implement `rewrite_passports`."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in passports.values():
            w.writerow(row)
    Path(tmp).replace(path)


UPGRADE_FIELDS = [
    "mileage_km",
    "engine_volume",
    "engine_type",
    "transmission",
    "body_type",
    "condition",
    "description",
    "labels",
    "city",
    "photos_count",
    "photo_url",
]


PG_INT_FIELDS = {
    "price_tenge",
    "year",
    "mileage_km",
    "photos_count",
    "views_count",
    "is_vip",
    "has_monthly_price",
}


def _pg_value(col: str, v):
    """Implement `_pg_value`."""
    if v is None or v == "":
        return None
    if col in PG_INT_FIELDS:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    return v


def flush_postgres(
    pg_new_ads: list[dict],
    pg_new_photos: list[dict],
    pg_sightings: list[dict],
    pg_upgraded: list[dict],
):
    """Implement `flush_postgres`."""

    def clean_rows(rows):
        return [{k: _pg_value(k, v) for k, v in row.items()} for row in rows]

    try:
        upsert("raw_ads", clean_rows(pg_new_ads), ["ad_id"])
        upsert("photos", pg_new_photos, ["ad_id", "position"])
        upsert("sightings", clean_rows(pg_sightings), ["ad_id", "seen_date"])
        upsert("raw_ads", clean_rows(pg_upgraded), ["ad_id"], update_cols=UPGRADE_FIELDS)
    except Exception as e:
        log.warning(f"PostgreSQL dual-write failed: {e}")


def clean_int(raw: str) -> int | None:
    digits = re.sub(r"\D", "", raw or "")
    return int(digits) if digits else None


def split_brand_model(title: str) -> tuple[str | None, str | None]:
    """'Kia K7' -> ('Kia', 'K7'); 'Mercedes-Benz GLS 450' -> ('Mercedes-Benz', 'GLS 450')."""
    if not title:
        return None, None
    for b in MULTIWORD_BRANDS:
        if title.startswith(b):
            return b, title[len(b) :].strip() or None
    parts = title.split(maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def parse_spec_line(text: str) -> dict:
    """Implement `parse_spec_line`."""
    r = {
        "year": None,
        "engine_volume": None,
        "engine_type": None,
        "transmission": None,
        "body_type": None,
        "condition": None,
        "mileage_km": None,
        "description": "",
    }
    if not text:
        return r
    low = text.lower()

    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        r["year"] = int(m.group())

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

    for fuel in ["газ-бензин", "гибрид", "электричество", "электро", "дизель", "газ", "бензин"]:
        if fuel in low:
            r["engine_type"] = "электро" if fuel == "электричество" else fuel
            break

    for kpp, canon in [
        ("автомат", "автомат"),
        ("вариатор", "вариатор"),
        ("робот", "робот"),
        ("механи", "механика"),
    ]:
        if kpp in low:
            r["transmission"] = canon
            break

    for body in [
        "кроссовер",
        "внедорожник",
        "седан",
        "хэтчбек",
        "лифтбек",
        "универсал",
        "минивэн",
        "микроавтобус",
        "купе",
        "пикап",
        "кабриолет",
        "родстер",
        "лимузин",
        "фургон",
    ]:
        if body in low:
            r["body_type"] = body
            break

    if "б/у" in low:
        r["condition"] = "б/у"
    elif "нов" in low.split(",")[0] or "новый" in low or "новая" in low:
        r["condition"] = "новый"

    for anchor in (
        r"с пробегом[\d\s\u00a0]+км,?\s*",
        r"\bкм,?\s*",
        r"КПП\s+\S+,\s*",
        r"(?:газ-бензин|бензин|дизель|гибрид|электро|газ),\s*",
    ):
        m = re.search(anchor + r"(.+)$", text, re.IGNORECASE)
        if not m:
            continue
        candidate = m.group(1).strip()

        if re.match(r"^(КПП\b|с пробегом\b)", candidate, re.IGNORECASE):
            break
        r["description"] = candidate[:500]
        break

    return r


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

        link = card.select_one("a[href*='/a/show/']")
        href = re.sub(r"\?.*", "", link["href"]) if link else f"/a/show/{ad_id}"
        url = BASE_URL + href

        title = (
            txt(card.select_one(f".{prefix}__title")).replace("Добавить в избранное", "").strip()
            or None
        )
        brand, model = split_brand_model(title or "")

        price = clean_int(txt(card.select_one(f".{prefix}__price")))

        spec = txt(card.select_one(f".{prefix}__description"))
        if not spec:
            img_alt = card.select_one("img[alt]")
            spec = img_alt["alt"] if img_alt else ""
        parsed = parse_spec_line(spec)

        city = (
            txt(card.select_one(f".{prefix}__region") or card.select_one("[data-test='region']"))
            or None
        )

        views = clean_int(txt(card.select_one(".nb-views")))
        posted = (
            txt(card.select_one(f".{prefix}__date") or card.select_one(".a-card__param--date"))
            or None
        )
        labels = "|".join(txt(x) for x in card.select(".a-label__text")) or None
        has_monthly = "/мес" in card.get_text()

        photo_urls = extract_photo_urls(card)
        cnt = clean_int(txt(card.select_one(".thumb-gallery__count")))
        photos_count = max(cnt or 0, len(photo_urls), 1)
        photo_url = photo_urls[0] if photo_urls else ""

        if not title or not price:
            continue

        results.append(
            {
                "ad_id": ad_id,
                "url": url,
                "title": title,
                "brand": brand,
                "model": model,
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
                "_photo_urls": photo_urls,
                "views_count": views,
                "posted_date": posted,
                "labels": labels,
                "is_vip": int(is_vip),
                "has_monthly_price": int(has_monthly),
                "category": category,
                "scraped_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    return results


async def human_pause(page):
    """Implement `human_pause`."""
    await asyncio.sleep(random.uniform(0.8, 1.6))
    await page.evaluate(f"window.scrollBy(0, {random.randint(400, 900)})")
    await asyncio.sleep(random.uniform(0.5, 1.2))


async def get_html(page, url: str, retries: int = 3) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            reserve_kolesa_request()
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await human_pause(page)
            return await page.content()
        except DailyBudgetExhausted:
            raise
        except request_budget.BudgetStateError:
            raise
        except PWTimeout:
            log.warning(f"Timeout [{attempt}/{retries}]: {url}")
        except Exception as e:
            log.error(f"Error [{attempt}/{retries}]: {e}")
        if attempt < retries:
            await asyncio.sleep(5 * (2 ** (attempt - 1)))
    return None


def looks_blocked(html: str) -> bool:
    """Implement `looks_blocked`."""
    if "js__a-card" in html:
        return False
    markers = [
        "Вход в личный кабинет",
        "passport/login",
        "Доступ ограничен",
        "Too Many Requests",
        "Подтвердите, что вы не робот",
    ]
    return any(m in html for m in markers)


def validate_listing_page(html: str, cards: list[dict], page_num: int, category: str) -> int:
    """Implement `validate_listing_page`."""
    raw_count = len(BeautifulSoup(html, "html.parser").select(".js__a-card"))
    if page_num <= LISTING_HEALTH_FIRST_PAGES and raw_count < LISTING_HEALTH_MIN_RAW_CARDS:
        raise ListingSchemaError(
            f"[{category}] page {page_num}: found only {raw_count} raw cards; "
            "Kolesa may have changed its HTML or selectors"
        )
    min_parsed = max(1, raw_count // 2)
    if raw_count >= LISTING_HEALTH_MIN_RAW_CARDS and len(cards) < min_parsed:
        raise ListingSchemaError(
            f"[{category}] page {page_num}: parsed {len(cards)}/{raw_count}; "
            "listing-card fields or selectors may have changed"
        )
    return raw_count


def page_limit_has_unseen(
    page_num: int, unseen: int, card_count: int, start_page: int, max_pages: int
) -> bool:
    """Implement `page_limit_has_unseen`."""
    return start_page == 1 and page_num == max_pages and card_count > 0 and unseen > 0


def cap_cards_for_run(
    cards: list[dict], already_processed: int, limit: int
) -> tuple[list[dict], bool]:
    """Implement `cap_cards_for_run`."""
    if limit <= 0:
        return cards, False
    remaining = max(0, limit - already_processed)
    selected = cards[:remaining]
    return selected, already_processed + len(selected) >= limit


def write_run_status(report: dict):
    """Implement `write_run_status`."""
    target = Path(RUN_STATUS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def mark_unhandled_failure(error: Exception):
    """Implement `mark_unhandled_failure`."""
    try:
        report = json.loads(Path(RUN_STATUS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {"schema_version": 1, "started_at": None, "segments": {}, "totals": {}}
    if report.get("status") != "running":
        return
    report.update(
        {
            "status": "failed",
            "message": f"{type(error).__name__}: {error}",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    write_run_status(report)


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
    total_cards_processed = 0
    card_limit_reached = False
    consecutive_fails = 0
    pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded = [], [], [], []
    run_report = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": None,
        "status": "running",
        "message": None,
        "config": {
            "start_page": START_PAGE,
            "max_pages_per_category": MAX_PAGES_PER_CATEGORY,
            "max_cards_per_run": MAX_CARDS_PER_RUN,
            "categories": [name for name, _ in CATEGORIES],
        },
        "segments": {},
        "freshness_truncated_segments": [],
        "totals": {},
    }

    def save_status(status: str, message: str | None = None):
        run_report["status"] = status
        run_report["message"] = message
        run_report["totals"] = {
            "new_ads": total_saved,
            "upgraded_passports": total_upgraded,
            "sightings": total_sightings,
            "cards_processed": total_cards_processed,
            "kolesa_requests_reserved": _run_kolesa_requests,
        }
        run_report["card_limit_reached"] = card_limit_reached
        run_report["freshness_truncated_segments"] = sorted(
            name
            for name, state in run_report["segments"].items()
            if state.get("page_limit_has_unseen")
        )
        if status != "running":
            run_report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_run_status(run_report)

    save_status("running")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )

        state = STATE_FILE if Path(STATE_FILE).exists() else None
        context = await browser.new_context(
            storage_state=state,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="ru-RU",
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = await context.new_page()

        log.info("Warming up the session...")
        try:
            reserve_kolesa_request()
            await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            await human_pause(page)
        except DailyBudgetExhausted as e:
            log.warning(f"Stopped before network access: {e}")
            save_status("budget_exhausted", str(e))
            await browser.close()
            return
        except request_budget.BudgetStateError as e:
            log.error(f"Stopped before network access: {e}")
            save_status("budget_error", str(e))
            await browser.close()
            return
        except Exception as e:
            log.warning(f"Session warm-up failed: {e}")

        pages_done = 0
        for cat_name, cat_path in CATEGORIES:
            log.info(f"── Category: {cat_name} ──")
            for page_num in range(START_PAGE, MAX_PAGES_PER_CATEGORY + 1):
                if page_num == 1:
                    url = f"{BASE_URL}{cat_path}"
                else:
                    sep = "&" if "?" in cat_path else "?"
                    url = f"{BASE_URL}{cat_path}{sep}page={page_num}"
                log.info(f"[{cat_name}] page {page_num}: {url}")

                try:
                    html = await get_html(page, url)
                except DailyBudgetExhausted as e:
                    log.warning(f"Stopped by shared request budget: {e}")
                    if total_upgraded:
                        rewrite_passports(OUTPUT_CSV, passports)
                    flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)
                    save_status("budget_exhausted", str(e))
                    await context.storage_state(path=STATE_FILE)
                    await browser.close()
                    return
                except request_budget.BudgetStateError as e:
                    log.error(f"Stopped: {e}")
                    if total_upgraded:
                        rewrite_passports(OUTPUT_CSV, passports)
                    flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)
                    save_status("budget_error", str(e))
                    await context.storage_state(path=STATE_FILE)
                    await browser.close()
                    return

                if html is None or looks_blocked(html or ""):
                    consecutive_fails += 1
                    log.error(
                        f"Failure/block response ({consecutive_fails}/{MAX_CONSECUTIVE_FAILS})"
                    )
                    if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                        log.error(
                            "Stopped after too many consecutive failures. "
                            "The next run will resume from the same position."
                        )
                        if total_upgraded:
                            rewrite_passports(OUTPUT_CSV, passports)
                        flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)
                        save_status("blocked", "three consecutive network or block responses")
                        await context.storage_state(path=STATE_FILE)
                        await browser.close()
                        sys.exit(1)
                    await asyncio.sleep(60)
                    continue
                consecutive_fails = 0

                cards = parse_cards(html, cat_name)
                try:
                    raw_card_count = validate_listing_page(html, cards, page_num, cat_name)
                except ListingSchemaError as e:
                    log.error(f"SCHEMA_ERROR {e}")
                    if total_upgraded:
                        rewrite_passports(OUTPUT_CSV, passports)
                    flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)
                    save_status("schema_error", str(e))
                    await context.storage_state(path=STATE_FILE)
                    await browser.close()
                    sys.exit(1)
                if not cards:
                    run_report["segments"][cat_name] = {
                        "last_page": page_num,
                        "raw_cards": raw_card_count,
                        "parsed_cards": 0,
                        "unseen_ads": 0,
                        "unseen_share": 0.0,
                        "page_limit_has_unseen": False,
                    }
                    save_status("running")
                    log.info(f"[{cat_name}] no listing cards; segment complete")
                    break

                run_cards, stop_after_page = cap_cards_for_run(
                    cards, total_cards_processed, MAX_CARDS_PER_RUN
                )
                total_cards_processed += len(run_cards)
                new = 0
                for row in run_cards:
                    photo_urls = row.pop("_photo_urls", [])

                    if row["ad_id"] not in seen_today:
                        append_sighting(SIGHTINGS_CSV, row, today)
                        pg_sightings.append(
                            {
                                "ad_id": row["ad_id"],
                                "seen_date": today,
                                "price_tenge": row["price_tenge"],
                                "views_count": row["views_count"],
                                "is_vip": row["is_vip"],
                                "category": row["category"],
                            }
                        )
                        seen_today.add(row["ad_id"])
                        total_sightings += 1

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
                    pg_new_photos.extend(
                        {"ad_id": row["ad_id"], "position": i, "url": u}
                        for i, u in enumerate(photo_urls, 1)
                    )
                    total_saved += 1
                    new += 1
                log.info(
                    f"  cards: {len(cards)}, new: {new}, "
                    f"processed this run: {total_cards_processed}, "
                    f"observations today: {total_sightings}, "
                    f"total new listings: {total_saved}"
                )

                boundary_open = page_limit_has_unseen(
                    page_num, new, len(cards), START_PAGE, MAX_PAGES_PER_CATEGORY
                )
                run_report["segments"][cat_name] = {
                    "last_page": page_num,
                    "raw_cards": raw_card_count,
                    "parsed_cards": len(cards),
                    "unseen_ads": new,
                    "processed_cards": len(run_cards),
                    "unseen_share": round(new / len(cards), 4),
                    "page_limit_has_unseen": boundary_open,
                }
                save_status("running")
                if boundary_open:
                    log.warning(
                        "FRESHNESS_TRUNCATED category=%s page=%d unseen=%d "
                        "cards=%d unseen_share=%.1f%%",
                        cat_name,
                        page_num,
                        new,
                        len(cards),
                        100 * new / len(cards),
                    )

                if stop_after_page:
                    card_limit_reached = True
                    save_status("running")
                    log.info("MICRO_LIMIT_REACHED cards=%d", total_cards_processed)
                    break

                pages_done += 1
                if pages_done % COFFEE_BREAK_EVERY == 0:
                    brk = random.uniform(*COFFEE_BREAK_RANGE)
                    log.info(f"  Long pacing break: {brk:.0f}s")
                    await asyncio.sleep(brk)
                else:
                    await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

            if card_limit_reached:
                break

        await context.storage_state(path=STATE_FILE)
        await browser.close()

    if total_upgraded:
        rewrite_passports(OUTPUT_CSV, passports)
        log.info(f"Backfilled listing records: {total_upgraded}")
    flush_postgres(pg_new_ads, pg_new_photos, pg_sightings, pg_upgraded)

    save_status("success")

    log.info(
        f"\n{'=' * 50}\nCompleted. New: {total_saved}, "
        f"backfilled: {total_upgraded}, "
        f"observations: {total_sightings}, "
        f"cards processed: {total_cards_processed} → {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as error:
        mark_unhandled_failure(error)
        raise
