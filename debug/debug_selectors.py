"""Open live Kolesa pages and report which parser selectors still work."""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


HERE = Path(__file__).parent
TEST_LISTING_URL = "https://kolesa.kz/cars/"


async def main():
    """Inspect listing and detail-page selectors in a visible browser."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=500)
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

        print(f"\n[1] Opening listing page: {TEST_LISTING_URL}")
        await page.goto(
            TEST_LISTING_URL,
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        await asyncio.sleep(3)
        (HERE / "debug_listing.html").write_text(await page.content(), encoding="utf-8")
        print("  → Saved debug_listing.html")

        selectors_to_try = [
            "a.card__link",
            "a[href*='/a/']",
            ".list-item a",
            ".card a",
            "article a",
        ]
        for selector in selectors_to_try:
            found = await page.query_selector_all(selector)
            if found:
                href = await found[0].get_attribute("href")
                print(
                    f"  ✓ Working selector: '{selector}' — "
                    f"found {len(found)} elements; first href: {href}"
                )
            else:
                print(f"  ✗ No match: '{selector}'")

        first_link = await page.query_selector("a[href*='/a/']") or await page.query_selector(
            "a.card__link"
        )
        if first_link:
            href = await first_link.get_attribute("href")
            if not href:
                raise RuntimeError("The first listing link has no href attribute")
            ad_url = "https://kolesa.kz" + href if href.startswith("/") else href
            print(f"\n[2] Opening listing detail: {ad_url}")
            await page.goto(ad_url, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)
            (HERE / "debug_ad.html").write_text(await page.content(), encoding="utf-8")
            print("  → Saved debug_ad.html")

            ad_selectors = {
                "Title": ["h1.offer__title", "h1"],
                "Price": [
                    "span.price__current",
                    ".offer-price__value",
                    "[class*='price']",
                ],
                "Parameters": [
                    "dl.offer-params__list",
                    ".offer-params",
                    ".offer-params li",
                ],
                "City": [".offer__location", "[class*='location']"],
                "Description": [
                    ".offer__description-text",
                    ".offer-description__text",
                    "[class*='description']",
                ],
                "Photos": [
                    "div.gallery__image img",
                    ".offer-gallery img",
                    "[class*='gallery'] img",
                ],
            }
            print("\n  Checking listing-detail selectors:")
            for name, selectors in ad_selectors.items():
                for selector in selectors:
                    found = await page.query_selector_all(selector)
                    if found:
                        value = (await found[0].text_content() or "").strip()
                        print(f"  ✓ {name}: '{selector}' ({len(found)} matches) → '{value[:60]}'")
                        break
                else:
                    print(f"  ✗ {name}: no selector matched")

        print("\n[!] Open debug_listing.html and debug_ad.html in a browser")
        print("[!] If a selector failed, open DevTools (F12), inspect the current")
        print("[!] classes, and update parser.py accordingly.\n")
        await asyncio.sleep(5)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
