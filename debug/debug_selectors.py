"""
Запускай это ПЕРВЫМ — он откроет браузер в видимом режиме,
зайдёт на сайт и сохранит HTML чтобы ты мог посмотреть реальные классы.

python debug/debug_selectors.py
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).parent


TEST_LISTING_URL = "https://kolesa.kz/cars/"
TEST_AD_URL      = None  # будет взят первый найденный

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,   # ВИДИМЫЙ браузер для отладки
            slow_mo=500,      # замедление для наглядности
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="ru-RU",
            viewport={"width": 1440, "height": 900},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        page = await context.new_page()

        print(f"\n[1] Открываем листинг: {TEST_LISTING_URL}")
        await page.goto(TEST_LISTING_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)

        html = await page.content()
        (HERE / "debug_listing.html").write_text(html, encoding="utf-8")
        print("  → Сохранён debug_listing.html")

        # Пробуем найти ссылки на объявления
        selectors_to_try = [
            "a.card__link",
            "a[href*='/a/']",
            ".list-item a",
            ".card a",
            "article a",
        ]
        for sel in selectors_to_try:
            found = await page.query_selector_all(sel)
            if found:
                href = await found[0].get_attribute("href")
                print(f"  ✓ Рабочий селектор: '{sel}' — нашёл {len(found)} элементов, первый href: {href}")
                # Запоминаем первый URL объявления
                if "/a/" in (href or "") or "/cars/" in (href or ""):
                    global_ad_url = "https://kolesa.kz" + href if href.startswith("/") else href
            else:
                print(f"  ✗ Не нашёл: '{sel}'")

        # Пробуем зайти на первое объявление
        first_link = await page.query_selector("a[href*='/a/']") or await page.query_selector("a.card__link")
        if first_link:
            href = await first_link.get_attribute("href")
            ad_url = "https://kolesa.kz" + href if href.startswith("/") else href
            print(f"\n[2] Открываем объявление: {ad_url}")
            await page.goto(ad_url, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)

            ad_html = await page.content()
            (HERE / "debug_ad.html").write_text(ad_html, encoding="utf-8")
            print("  → Сохранён debug_ad.html")

            # Проверяем селекторы для объявления
            ad_selectors = {
                "Заголовок":    ["h1.offer__title", "h1"],
                "Цена":         ["span.price__current", ".offer-price__value", "[class*='price']"],
                "Параметры":    ["dl.offer-params__list", ".offer-params", ".offer-params li"],
                "Город":        [".offer__location", "[class*='location']"],
                "Описание":     [".offer__description-text", ".offer-description__text", "[class*='description']"],
                "Фото":         ["div.gallery__image img", ".offer-gallery img", "[class*='gallery'] img"],
            }
            print("\n  Проверяем селекторы объявления:")
            for name, sels in ad_selectors.items():
                for sel in sels:
                    found = await page.query_selector_all(sel)
                    if found:
                        text = await found[0].text_content()
                        print(f"  ✓ {name}: '{sel}' ({len(found)} шт.) → '{text[:60].strip()}'")
                        break
                else:
                    print(f"  ✗ {name}: ни один селектор не сработал!")

        print("\n[!] Открой debug_listing.html и debug_ad.html в браузере")
        print("[!] Если какие-то селекторы не сработали — открой DevTools (F12)")
        print("[!] и найди правильные классы, затем обнови parser.py\n")

        await asyncio.sleep(5)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())