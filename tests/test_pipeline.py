# -*- coding: utf-8 -*-
"""
Тесты чистых функций пайплайна. Запуск: pytest test_pipeline.py -v

Философия: тестируем ЛОГИКУ без сети и файлов. Все проверяемые функции
чистые (вход → выход), поэтому тесты выполняются за миллисекунды и их
можно гонять при каждом изменении. Сетевые джобы тестируются через эти
же функции на сохранённых HTML-страницах (fixtures).

Каждый тест здесь — это застывший баг или инвариант, который мы уже
ловили руками в процессе разработки. Тест = страховка от регрессии
(regression — возвращение старого бага после новых правок).
"""

import pandas as pd

from kz.collect import parser as listing_parser
from kz.transform import clean
from kz.collect import enrich
from kz.collect import photo_dedup
from kz.report import evaluate_detector


# ─── parse_spec_line: оба формата пробега (реальный баг №1) ──────────────────
def test_mileage_with_probegom():
    r = listing_parser.parse_spec_line(
        "2014 г., Б/у седан, 3 л, бензин, КПП автомат, с пробегом 170 000 км, срочно")
    assert r["mileage_km"] == 170000


def test_mileage_bare_km():
    """Формат без слов «с пробегом» — терялось 40% пробегов."""
    r = listing_parser.parse_spec_line(
        "1994 г., Б/у седан, 2 л, бензин, КПП механика, 260 000 км, серебристый")
    assert r["mileage_km"] == 260000


def test_vip_line_without_mileage():
    r = listing_parser.parse_spec_line("2026 г., 1.5 л, гибрид, КПП автомат")
    assert r["mileage_km"] is None
    assert r["year"] == 2026
    assert r["transmission"] == "автомат"


# ─── топливо: порядок специфичности (реальный баг №2) ────────────────────────
def test_gas_petrol_not_mislabeled():
    """«газ-бензин» не должен определяться как «бензин»."""
    r = listing_parser.parse_spec_line("2010 г., Б/у седан, 2 л, газ-бензин, КПП автомат")
    assert r["engine_type"] == "газ-бензин"


def test_crossover_body_detected():
    """«кроссовер» отсутствовал в списке кузовов — 34% пропусков."""
    r = listing_parser.parse_spec_line("2020 г., Б/у кроссовер, 2 л, бензин, КПП автомат")
    assert r["body_type"] == "кроссовер"


# ─── фото: превью → полный размер ────────────────────────────────────────────
def test_full_size_url():
    assert listing_parser.to_full_size(
        "https://x.kcdn.kz/webp/aa/bb/13-255x138.jpg").endswith("13-full.jpg")
    assert listing_parser.to_full_size(
        "https://x.kcdn.kz/webp/aa/bb/9-160x120.webp").endswith("9-full.webp")


def test_static_ui_images_not_collected_as_photos():
    """Реальный баг: карточка содержит служебные картинки вёрстки
    (badge.png — 603 штуки попали в photos, заглушки noPhoto_*.svg) —
    это не фото машины. Бейдж одинаков у всех карточек: попав в
    photo_dedup, он «совпал» бы у сотен объявлений (ложные пары)."""
    from bs4 import BeautifulSoup
    html = '''<div class="js__a-card" data-id="1">
        <img src="https://m.kolesa.kz/static/mobile/images/app/report/advert/badge.png"/>
        <img src="//kolesa.kz/static/frontend/images/stubs/noPhoto_160x120.svg"/>
        <img src="https://alakt-photos-kl.kcdn.kz/webp/aa/bb/1-255x138.jpg"/>
    </div>'''
    card = BeautifulSoup(html, "html.parser").select_one(".js__a-card")
    urls = listing_parser.extract_photo_urls(card)
    assert urls == ["https://alakt-photos-kl.kcdn.kz/webp/aa/bb/1-full.jpg"]


# ─── бренд/модель ────────────────────────────────────────────────────────────
def test_split_brand_model():
    assert listing_parser.split_brand_model("Kia K7") == ("Kia", "K7")
    assert listing_parser.split_brand_model("Mercedes-Benz GLS 450") == \
        ("Mercedes-Benz", "GLS 450")


# ─── детектор блокировки (реальный баг №3: ложное срабатывание) ─────────────
def test_normal_page_with_recaptcha_footer_not_blocked():
    """Слово captcha в футере — НЕ блокировка (нас на этом словили)."""
    html = '<div class="js__a-card">...</div><p>Защищено reCAPTCHA</p>'
    assert listing_parser.looks_blocked(html) is False


def test_login_page_is_blocked():
    assert listing_parser.looks_blocked(
        "<title>Вход в личный кабинет</title>") is True


# ─── комментарий продавца из embedded JSON (unicode-escape) ──────────────────
def test_seller_comment_unicode_escape():
    html = ('{"descriptionText":"\\u041f\\u0440\\u043e\\u0434\\u0430\\u043c '
            '\\u0430\\u0432\\u0442\\u043e<br />\\u0442\\u043e\\u0440\\u0433"}')
    text = enrich.extract_seller_comment(html)
    assert "Продам авто" in text
    assert "<br" not in text          # html-теги вычищены


# ─── лексикон убитости: хитрое прошедшее время ───────────────────────────────
def test_damage_past_tense_running():
    """«был находу» = сейчас НЕ на ходу (реальный кейс Пассата за 200к)."""
    searchable = "продам пассат был находу двиготель коробка есть"
    hits = [p for p in enrich.DAMAGE_PATTERNS if p in searchable]
    assert "был находу" in hits


# ─── ТЕСТ-СТОРОЖ: лексиконы в clean и enrich не должны разъехаться ───────────
def test_damage_patterns_in_sync():
    """Раньше DAMAGE_PATTERNS дублировался в clean.py и enrich.py и этот
    тест сторожил синхронность копий. Теперь источник один — damage.py,
    а тест сторожит, что оба файла реально импортируют ЕГО (а не завели
    свою копию заново)."""
    from kz.transform import damage
    assert clean.DAMAGE_PATTERNS is damage.DAMAGE_PATTERNS
    assert enrich.DAMAGE_PATTERNS is damage.DAMAGE_PATTERNS


# ─── отрицания: «нет гнили» — это НЕ повреждение (реальные кейсы из базы) ────
def test_negated_damage_not_detected():
    """Три реальных объявления, ложно получавших damage_keywords:
    225209601 («На 99% нету никаких гнилей» → 'гнил'),
    225770229 («Вложения не требует» → 'вложения'),
    225838706 («не требует вложений» → 'требует вложений')."""
    from kz.transform.damage import find_damage_keywords
    assert find_damage_keywords(
        "Машина в идеальном состоянии. На 99% нету никаких гнилей.") == []
    assert find_damage_keywords("Вложения не требует. Обмен не интересует!") == []
    assert find_damage_keywords(
        "В хорошем состоянии, не требует вложений, ТО пройдено.") == []


def test_real_damage_still_detected():
    """Позитивный контроль: настоящая убитость не должна потеряться
    из-за окна отрицаний."""
    from kz.transform.damage import find_damage_keywords
    assert "гнил" in find_damage_keywords("кузов гнилой, пороги под замену")
    assert "требует вложений" in find_damage_keywords("машина требует вложений")
    assert "не на ходу" in find_damage_keywords("стоит в гараже, не на ходу")
    assert "после дтп" in find_damage_keywords("продаю после дтп, на запчасти")
    # отрицание есть, но НЕ относится к хиту — хит должен выжить
    assert "ржавчин" in find_damage_keywords("салон не прокурен, есть ржавчина по аркам")
    # паттерны, сами начинающиеся с отрицания: соседняя фраза «без матора,»
    # не должна гасить следующий за ней хит «без коробки» (кейс Delica)
    kws = find_damage_keywords("Машина без матора, без коробки, остальное на месте")
    assert "без матора" in kws and "без коробки" in kws


def test_damage_disclosed_rust_and_gearbox():
    """Раскрытые дефекты, которые старый лексикон пропускал (реальный ad
    225502216 Chevrolet Aveo): «рыжики» (сленг про ржавчину) и «не
    включается 5-я передача» (общий дефект, брат «не работает»). Обычные
    отрицания ПЕРЕД словом по-прежнему гасят хит (окно 2 токена)."""
    from kz.transform.damage import find_damage_keywords, has_damage
    assert has_damage("есть классические рыжики на порогах")
    assert has_damage("не включается 5-я передача")
    assert has_damage("не включается кондиционер")
    assert not has_damage("без рыжиков, кузов идеальный")
    assert not has_damage("нет рыжиков")


# ─── статистика: модифицированный z-score робастен к выбросу ─────────────────
def test_robust_z_ignores_single_outlier():
    import pandas as pd
    import numpy as np
    # 20 «нормальных» цен + один дикий выброс
    s = pd.Series(np.log([5e6] * 10 + [6e6] * 10 + [200e6]))
    med = s.median()
    mad = (s - med).abs().median()
    z_outlier = 0.6745 * (s.iloc[-1] - med) / mad
    z_normal = 0.6745 * (s.iloc[0] - med) / mad
    assert abs(z_outlier) > 3.5      # выброс пойман
    assert abs(z_normal) < 3.5       # нормальные не задеты (нет masking)


# ─── описание: цепочка якорей (реальный баг №4: терялись 360 описаний) ───────
def test_description_after_km():
    r = listing_parser.parse_spec_line(
        "2014 г., Б/у седан, 3 л, бензин, КПП автомат, с пробегом 170 000 км, Срочно нужны деньги")
    assert r["description"] == "Срочно нужны деньги"


def test_description_without_km_after_kpp():
    """Нет пробега в строке → текст продавца всё равно извлекается."""
    r = listing_parser.parse_spec_line(
        "2008 г., Б/у седан, 1.6 л, бензин, КПП механика, Авто в хорошем состояний сел поехал")
    assert r["description"] == "Авто в хорошем состояний сел поехал"


def test_description_without_km_and_kpp():
    r = listing_parser.parse_spec_line(
        "1997 г., Б/у минивэн, 2 л, бензин, синий, литые диски")
    assert "синий" in r["description"]


def test_description_empty_when_no_seller_text():
    r = listing_parser.parse_spec_line("2026 г., 1.5 л, гибрид, КПП автомат")
    assert r["description"] == ""


# ─── перезаливы: совпадение title+год+цена требует второго фактора ──────────
def _dup_df(rows):
    cols = ["ad_id", "title", "year", "price_tenge", "mileage_km",
            "description", "condition", "labels"]
    d = pd.DataFrame(rows, columns=cols)
    d["info_flags"] = ""
    return d


def test_repost_confirmed_by_mileage_is_flagged():
    d = _dup_df([
        ("1", "Kia Rio", 2015, 5_000_000, 120_000, "продам", "б/у", None),
        ("2", "Kia Rio", 2015, 5_000_000, 120_000, None,     "б/у", None),
    ])
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "possible_repost").all()


def test_repost_unconfirmed_goes_to_info_only():
    """Калибровка 2026-07-20: 69 из 96 групп совпадали ТОЛЬКО по
    title+год+цена при разных пробегах (три разных BMW X5 по круглой
    рыночной цене) — совпадение, а не перезалив. Такие — только
    информационная пометка, не is_suspicious."""
    d = _dup_df([
        ("1", "BMW X5", 2016, 17_500_000, 210_000, None, "б/у", None),
        ("2", "BMW X5", 2016, 17_500_000, 241_200, None, "б/у", None),
    ])
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "").all()
    assert (out["info_flags"].str.contains("repost_unconfirmed")).all()


def _dup_df_color(rows):
    cols = ["ad_id", "title", "year", "price_tenge", "mileage_km",
            "description", "condition", "labels", "color"]
    d = pd.DataFrame(rows, columns=cols)
    d["info_flags"] = ""
    return d


def test_repost_different_base_color_not_confirmed():
    """Цвет не меняется у той же машины: разный БАЗОВЫЙ цвет при совпавших
    title+год+цена+пробег = РАЗНЫЕ авто (совпал круглый пробег), не перезалив.
    Реальный кейс: белая и чёрная Hyundai Sonata 2023, обе 100000 км, 10.5М."""
    d = _dup_df_color([
        ("1", "Hyundai Sonata", 2023, 10_500_000, 100_000, None, "б/у", None, "белый"),
        ("2", "Hyundai Sonata", 2023, 10_500_000, 100_000, None, "б/у", None, "черный металлик"),
    ])
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "").all()                       # разный цвет → не перезалив
    assert out["info_flags"].str.contains("repost_unconfirmed").all()


def test_repost_same_base_color_metallic_still_confirmed():
    """«белый» vs «белый металлик» — тот же базовый цвет (суффикс финиша не
    различает машину) → остаётся possible_repost (реальный кейс Camry 2021)."""
    d = _dup_df_color([
        ("1", "Toyota Camry", 2021, 14_500_000, 110_000, None, "б/у", None, "белый"),
        ("2", "Toyota Camry", 2021, 14_500_000, 110_000, None, "б/у", None, "белый металлик"),
    ])
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "possible_repost").all()


# ─── photo_dedup: фото одно — машины разные (стоит подозрения) ───────────────
def _cars(rows):
    return pd.DataFrame(rows, columns=["ad_id", "brand", "model", "year", "price_tenge"])


def test_exact_hash_diff_model_is_flagged():
    hashes = pd.DataFrame([
        {"ad_id": "1", "position": 1, "phash": "a" * 16},
        {"ad_id": "2", "position": 1, "phash": "a" * 16},
    ])
    cars = _cars([
        ("1", "Toyota", "Camry", 2015, 5_000_000),
        ("2", "Honda", "Civic", 2015, 5_000_000),
    ])
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert {"1", "2"} == {out.iloc[0]["ad_id_a"], out.iloc[0]["ad_id_b"]}


def test_exact_hash_same_car_not_flagged_dealer_repost():
    """Тот же дилер перезалил те же фото под ту же машину — не мошенничество,
    это уже ловит add_duplicate_flags() в clean.py, задваивать не надо."""
    hashes = pd.DataFrame([
        {"ad_id": "1", "position": 1, "phash": "a" * 16},
        {"ad_id": "2", "position": 1, "phash": "a" * 16},
    ])
    cars = _cars([
        ("1", "Kia", "Rio", 2020, 6_000_000),
        ("2", "Kia", "Rio", 2020, 6_050_000),   # цена почти та же (<15%)
    ])
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert out.empty


def test_dealer_press_photo_across_trims_not_flagged():
    """Реальный класс ложняков (2026-07): официальный дилер вешает ОДНО
    пресс-фото на разные комплектации одной модели (OMODA S5 Life/Prestige,
    GAC GS3 GB/GL) — фото совпадает точно, но это не кража. Дилер↔дилер
    исключаем; дилер↔частник (украли пресс-фото под чужой б/у) — остаётся."""
    hashes = pd.DataFrame([
        {"ad_id": "1", "position": 1, "phash": "a" * 16},
        {"ad_id": "2", "position": 1, "phash": "a" * 16},
        {"ad_id": "3", "position": 1, "phash": "a" * 16},
    ])
    cars = pd.DataFrame([
        ("1", "OMODA", "S5 Life",     2025, 7_490_000, "новый", "Новая|Официальный дилер"),
        ("2", "OMODA", "S5 Prestige", 2025, 7_990_000, "новый", "Новая|Официальный дилер"),
        ("3", "Chevrolet", "Nexia",   2015, 3_000_000, "б/у",   ""),   # частник, украл фото
    ], columns=["ad_id", "brand", "model", "year", "price_tenge", "condition", "labels"])
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    pairs = {frozenset((r.ad_id_a, r.ad_id_b)) for r in out.itertuples()}
    assert frozenset(("1", "2")) not in pairs      # дилер↔дилер — исключены
    assert frozenset(("1", "3")) in pairs           # дилер↔частник — кража, флаг остаётся
    assert frozenset(("2", "3")) in pairs


def test_near_hash_not_flagged_studio_lookalike():
    """Калибровка на реальных данных (2026-07-20): ВСЕ пары с hamming
    2-4 оказались «одна дилерская студия — разные машины» (Lexus RX vs
    Chery Tiggo: тот же поворотный круг, фон, ракурс), а настоящий
    дубль (OMODA) дал ровно 0. Порог ужесточён до точного равенства —
    «почти совпадение» на 64-битном pHash в этом домене означает
    одинаковую композицию, а не одинаковую машину."""
    hashes = pd.DataFrame([
        {"ad_id": "1", "position": 1, "phash": "0000000000000000"},
        {"ad_id": "2", "position": 1, "phash": "0000000000000001"},
    ])
    cars = _cars([
        ("1", "BMW", "X5", 2010, 8_000_000),
        ("2", "BMW", "X5", 2020, 8_000_000),
    ])
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert out.empty


def test_single_photo_no_match_not_flagged():
    """Реальный баг: pd.DataFrame.from_records([]) на пустом списке даёт
    DataFrame БЕЗ колонок вообще, а не с ожидаемой схемой — это ломало
    to_sql("photo_duplicates", ...) и последующий SELECT ad_id_a в
    clean.py. Пустой результат должен сохранять правильные колонки."""
    hashes = pd.DataFrame([{"ad_id": "1", "position": 1, "phash": "a" * 16}])
    cars = _cars([("1", "Toyota", "Camry", 2015, 5_000_000)])
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert out.empty
    assert list(out.columns) == [
        "ad_id_a", "ad_id_b", "hamming_distance",
        "model_key_a", "price_a", "year_a",
        "model_key_b", "price_b", "year_b"]


# ─── evaluate_detector: матрица ошибок считается верно ──────────────────────
def test_confusion_matrix_counts():
    """Харнесс метрик (precision/recall) должен верно раскладывать
    предсказания детектора против ручных вердиктов по 4 клеткам."""
    df = pd.DataFrame({
        "ad_id": list("123456"),
        "is_suspicious": [1, 1, 0, 1, 0, 0],
        "verdict": ["fraud", "fraud", "fraud", "legit", "legit", "legit"],
    })
    c = evaluate_detector.confusion(df)
    assert c == {"TP": 2, "FP": 1, "FN": 1, "TN": 2}


def test_weighted_confusion_uses_sampling_probability():
    """Один fraud в контроле 1/100 представляет 100 строк населения."""
    df = pd.DataFrame({
        "is_suspicious": [1, 1, 0],
        "verdict": ["fraud", "legit", "fraud"],
        "stratum_population": [2, 2, 100],
        "stratum_sample_size": [2, 2, 1],
    })
    c = evaluate_detector.weighted_confusion(df)
    assert c == {"TP": 1.0, "FP": 1.0, "FN": 100.0, "TN": 0.0}
    assert evaluate_detector._prf({"TP": 0, "FP": 2, "FN": 3, "TN": 0}) == (
        0.0, 0.0, 0.0
    )


# ─── dual-write: "50.0" из CSV не должно ронять запись в Postgres INTEGER ────
def test_pg_value_coerces_float_strings():
    """Реальный баг: после pandas-round-trip пробег в CSV = '50.0';
    Postgres-колонка INTEGER такую строку отвергает и роняет весь батч
    dual-write. _pg_value приводит '50'/'50.0'/50/50.0 → int, пусто → None."""
    assert listing_parser._pg_value("mileage_km", "50.0") == 50
    assert listing_parser._pg_value("mileage_km", "50") == 50
    assert listing_parser._pg_value("mileage_km", 50.0) == 50
    assert listing_parser._pg_value("mileage_km", "") is None
    assert listing_parser._pg_value("mileage_km", None) is None
    # текстовую колонку не трогаем
    assert listing_parser._pg_value("title", "Toyota Camry") == "Toyota Camry"


# ─── kolesa avgPrice: извлечение + кросс-чек детектора ──────────────────────
def test_extract_avg_price():
    """avgPrice лежит в embedded JSON страницы объявления."""
    assert enrich.extract_avg_price('..."brand":"BYD","avgPrice":23608000},...') == 23608000
    assert enrich.extract_avg_price("нет такого ключа") is None


def test_kolesa_cross_check_downgrades_false_positive():
    """Реальный кейс BYD Leopard 5 2024: наш z=-7.6 (корзина смешала
    2024 с 2026), но цена 22.5М в пределах рыночной по kolesa (23.6М) →
    ложное срабатывание снимается в info-пометку kolesa_price_ok.
    А реально дешёвое (5М при рынке 20М) — флаг остаётся."""
    d = pd.DataFrame({
        "price_tenge": [22_500_000, 5_000_000],
        "kolesa_avg_price": [23_608_000, 20_000_000],
        "stat_reasons": ["price_anomaly_low", "price_anomaly_low"],
        "info_flags": ["", ""],
        "text_full": ["", ""],
        "condition": ["б/у", "б/у"],
        "labels": ["", ""],
    })
    out = clean.exculpate(d)
    assert out.iloc[0]["stat_reasons"] == ""                       # BYD снят
    assert "kolesa_price_ok" in out.iloc[0]["info_flags"]
    assert out.iloc[1]["stat_reasons"] == "price_anomaly_low"      # реальный дешёвый остался


def test_kolesa_sentinel_and_missing_do_not_exculpate():
    """-1 (у модели нет эталона kolesa) и NaN (не обогащено) НЕ должны
    оправдывать — иначе price >= 0.80*(-1) ложно снял бы любой флаг."""
    d = pd.DataFrame({
        "price_tenge": [5_000_000, 5_000_000],
        "kolesa_avg_price": [-1, None],
        "stat_reasons": ["price_anomaly_low", "price_anomaly_low"],
        "info_flags": ["", ""], "text_full": ["", ""],
        "condition": ["б/у", "б/у"], "labels": ["", ""],
    })
    out = clean.exculpate(d)
    assert (out["stat_reasons"] == "price_anomaly_low").all()


# ─── статус-бейдж сайта: извлечение + оправдание young_car_cheap ────────────
def test_extract_status_badge():
    """Бейдж «Аварийная/Не на ходу» — div.offer__parameters-mortgaged,
    отдельный от dt/dd (раньше не собирался вообще)."""
    html = ('<div class="offer__parameters-mortgaged" '
            'data-test="offer-parameters">Аварийная/Не на ходу</div>')
    assert enrich.parse_ad_page(html)["page_status_badge"] == "Аварийная/Не на ходу"
    # нет бейджа → маркер "-" («проверено, бейджа нет»), НЕ None
    assert enrich.parse_ad_page("<div>нет бейджа</div>")["page_status_badge"] == "-"


def test_used_zero_mileage_excludes_current_year_new():
    """б/у + 0 км = сокрытие пробега ТОЛЬКО у не-новой машины. У текущего
    модельного года 0 км — правда «новая со склада» (реальный кейс Changan
    X5 Plus 2026: condition криво распарсился как «б/у», лейбл «первый взнос»,
    коммент «машина новая без пробега»)."""
    from kz.transform import clean
    cy = clean.CURRENT_YEAR
    df = pd.DataFrame([
        {"year": cy,     "condition": "б/у", "mileage_km": 0, "price_tenge": 7_600_000},
        {"year": cy - 5, "condition": "б/у", "mileage_km": 0, "price_tenge": 3_000_000},
    ])
    df["age"] = cy - df["year"] + 1
    out = clean.apply_hard_rules(df)
    assert "used_but_zero_mileage" not in out.iloc[0]["rule_reasons"]   # новая — не флаг
    assert "used_but_zero_mileage" in out.iloc[1]["rule_reasons"]       # старая 0 км — флаг


def test_young_car_cheap_cleared_when_declared_wreck():
    """Честно битый молодой дешёвый (сайт: «Аварийная») — НЕ приманка,
    young_car_cheap снимается. Кейс Chevrolet Onix 2023 за 1.7М.
    А молодой дешёвый БЕЗ объяснения — флаг остаётся (на разметку)."""
    d = pd.DataFrame({
        "price_tenge": [1_700_000, 3_500_000],
        "rule_reasons": ["young_car_cheap", "young_car_cheap"],
        "stat_reasons": ["", ""],
        "info_flags": ["", ""],
        "text_full": ["", ""],
        "damage_keywords": ["", ""],
        "condition": ["б/у", "б/у"],
        "labels": ["", ""],
        "page_status_badge": ["Аварийная/Не на ходу", None],   # 1-я битая, 2-я нет
    })
    out = clean.exculpate(d)
    assert "young_car_cheap" not in out.iloc[0]["rule_reasons"]   # битая — снято
    assert "low_price_explained" in out.iloc[0]["info_flags"]
    assert "young_car_cheap" in out.iloc[1]["rule_reasons"]       # без объяснения — осталось


# ─── catch_up: оркестратор ссылается на реально существующие скрипты ────────
def test_catch_up_references_real_modules():
    """Защита от опечатки в имени джоба — оркестратор упал бы в рантайме.

    Джобы запускаются как `python -m <модуль>`, поэтому проверяем именно
    разрешимость модуля, а не наличие файла: путь к файлу опечатку в
    пакетной части имени не поймал бы."""
    import importlib.util
    from kz.ops import catch_up
    mods = ([s for _, s, _ in catch_up.KOLESA]
            + [s for _, s, _ in catch_up.CDN]
            + [s for _, s in catch_up.OFFLINE])
    for m in mods:
        assert importlib.util.find_spec(m), f"catch_up ссылается на несуществующий {m}"


# ─── data-quality: плейсхолдер-пробег (777777) занулить перед моделью ───────
def test_junk_mileage_placeholder_detection():
    """Репдигит >300k («забитое» поле 777777) — junk; реальные 99999/111111/
    150000 и 0/None — НЕ junk."""
    from kz.transform.data_quality import is_junk_mileage
    assert is_junk_mileage(777777)          # 777k репдигит → плейсхолдер
    assert is_junk_mileage(999999)
    assert is_junk_mileage(888888)
    assert not is_junk_mileage(99999)       # 99k — правдоподобно
    assert not is_junk_mileage(111111)      # 111k < 300k — не трогаем
    assert not is_junk_mileage(150000)      # обычный пробег
    assert not is_junk_mileage(0)           # отдельная история (used_but_zero)
    assert not is_junk_mileage(None)
    assert not is_junk_mileage(float("nan"))


# ─── time-to-sell (уровень 2): парсинг даты публикации из карточки ──────────
def test_parse_posted_date():
    from datetime import date
    from kz.ml.time_to_sell import parse_posted
    cy = date.today().year
    assert parse_posted("18 июля") == date(cy, 7, 18)
    assert parse_posted("18 июл.") == date(cy, 7, 18)      # сокращение
    assert parse_posted("5 мая") == date(cy, 5, 5)
    assert parse_posted("сегодня") is None                 # относительная — не дата
    assert parse_posted(None) is None
    assert parse_posted("99 июля") is None                 # невалидный день


# ─── квантильный residual-детектор: конфиг осмыслен, фичи без утечки ────────
def test_residual_detector_config():
    from kz.ml import residual_detector as r
    from kz.ml.train_price_model import FEATURES
    assert 0 < r.ALPHA < 0.5              # нижний квантиль (пол цены)
    assert r.MIN_SUPPORT >= 1 and r.AGE_MAX >= 1
    assert r.FEATURES is FEATURES         # те же фичи модели → та же анти-утечка


# ─── текстовые фичи для модели цены (интерпретируемые keyword-сигналы) ──────
def test_text_features_extract():
    from kz.transform.text_features import text_features
    f = text_features("Максимальная комплектация, кожа, панорама, камера. "
                      "Не бит не крашен, один хозяин.")
    assert f["txt_opt_count"] >= 3        # макс.компл + кожа + панорама + камера
    assert f["txt_positive"] == 1
    assert f["txt_damage"] == 0
    f2 = text_features("требует ремонта, рыжики на порогах, продаю срочно")
    assert f2["txt_damage"] == 1          # damage-лексикон (вкл. рыжики)
    assert f2["txt_urgency"] == 1
    assert f2["txt_opt_count"] == 0
    assert text_features(None)["txt_len"] == 0


# ─── модель цены: НЕ должна видеть признаки-утечки цели ─────────────────────
def test_price_model_features_no_leakage():
    """Модель цены не должна учиться на признаках, производных ОТ цены или на
    чужой оценке цены (target leakage, правило №6): kolesa_avg_price, price_z,
    сама цена, is_suspicious. city — константа (Алматы), views_count —
    пост-фактум. Иначе модель «списывала» бы, а не оценивала."""
    from kz.ml import train_price_model as m
    banned = {"price_tenge", "log_price", "price_z", "kolesa_avg_price",
              "is_suspicious", "suspicion_reasons", "city", "views_count"}
    leak = set(m.FEATURES) & banned
    assert not leak, f"утечка цели в фичах модели: {leak}"


# ─── catch_up --values: приоритет ценных-для-оправдания джобов ──────────────
def test_catch_up_value_jobs_are_exculpation_fillers():
    """--values гоняет ТОЛЬКО enrich+backfill (заполняют avgPrice/бейдж/цвет/
    damage — поля exculpation), пропуская статусы и фото. Подмножество KOLESA."""
    import importlib.util
    from kz.ops import catch_up
    keys = [k for _, _, k in catch_up.VALUE_JOBS]
    assert keys == ["enrich", "backfill"]
    assert all(j in catch_up.KOLESA for j in catch_up.VALUE_JOBS)   # подмножество KOLESA
    assert "status" not in keys and "photo" not in keys            # не liveness/фото
    for _, mod, _ in catch_up.VALUE_JOBS:
        assert importlib.util.find_spec(mod)
    # --backfill ещё уже: только backfill (чистый добор avgPrice+бейджа),
    # подмножество VALUE_JOBS, без enrich
    bkeys = [k for _, _, k in catch_up.BACKFILL_JOBS]
    assert bkeys == ["backfill"]
    assert all(j in catch_up.VALUE_JOBS for j in catch_up.BACKFILL_JOBS)


# ─── catch_up: детект 429 не должен ложно срабатывать на числах ─────────────
def test_catch_up_429_detection_not_fooled_by_numbers():
    """Реальный баг моего же кода: count_429 считал подстроку '429', а она
    есть в ad_id/ценах/таймстемпах («наблюдений: 429») → catch_up ложно
    обрывал бы джобы. Считаем только настоящие rate-limit-строки."""
    from kz.ops import catch_up
    assert catch_up.is_429_line("2026-01-01 12:00:00  INFO  429: пауза 120с")
    assert catch_up.is_429_line("Стоп: 429 подряд — сайт лимитирует")
    assert not catch_up.is_429_line("наблюдений сегодня: 429, всего: 429")
    assert not catch_up.is_429_line("2026-07-18 20:15:23,429  INFO  карточек: 23")
    assert not catch_up.is_429_line("ad_id 224290000 обработан")


# ─── catch_up --until-done: решение цикла после каждой порции ────────────────
def test_catch_up_until_done_next_action():
    """Чистая логика режима «добить до конца». Критично, что нет вечного
    цикла: если порция отработала чисто (rc=0, без 429), но пробел НЕ
    уменьшился — это 'stuck' (остаток недозаполним: 404/нет данных/сентинелы),
    а не бесконечный повтор тех же строк."""
    from kz.ops.catch_up import next_action
    # прогресс есть → крутим дальше
    assert next_action(500, 380, 0, False) == "continue"
    # пробел закрыт → готово (даже если формально был 429 на последнем запросе)
    assert next_action(120, 0, 0, False) == "done"
    assert next_action(120, 0, 0, True) == "done"
    # новый 429 при незакрытом пробеле → стоп цепочки (важнее предохранителя)
    assert next_action(500, 450, 0, True) == "rate_limited"
    # джоб вышел с ошибкой (внутренний предохранитель) → стоп
    assert next_action(500, 480, 1, False) == "breaker"
    # порция не сдвинула пробел без 429/ошибки → недозаполнимо, не зациклиться
    assert next_action(30, 30, 0, False) == "stuck"
    assert next_action(30, 31, 0, False) == "stuck"


# ─── catch_up: дневной бюджет запросов на хост (анти-бан) ────────────────────
def test_catch_up_chunk_sizes_match_jobs():
    """CHUNK_MAX в catch_up — копия MAX_PER_RUN самих джобов (импорт джобов
    там избегаем ради их import-side-effects). Если в джобе поменяли лимит,
    а тут забыли — бюджет считался бы по устаревшей цифре. Этот тест ловит дрейф."""
    from kz.ops import catch_up
    from kz.collect import check_status, enrich, backfill_avgprice, photo_dedup
    assert catch_up.CHUNK_MAX["status"]   == check_status.MAX_CHECKS_PER_RUN
    assert catch_up.CHUNK_MAX["enrich"]   == enrich.MAX_PER_RUN
    assert catch_up.CHUNK_MAX["backfill"] == backfill_avgprice.MAX_PER_RUN
    assert catch_up.CHUNK_MAX["photo"]    == photo_dedup.MAX_PER_RUN


def test_catch_up_budget_allows_near_done_at_edge():
    """Оценка стоимости порции = min(MAX_PER_RUN, пробел): полная порция у
    края квоты НЕ влезает, но почти добитый джоб (маленький пробел) — влезает
    в тот же остаток. Иначе near-done джоб голодал бы у границы бюджета."""
    from kz.ops import catch_up
    B = catch_up.DAILY_BUDGET["kolesa"]
    cm = catch_up.CHUNK_MAX["enrich"]
    assert catch_up.budget_allows("kolesa", "status", 10**6, {"kolesa": 0, "cdn": 0})
    edge = B - (cm - 1)                       # остаток = cm-1 < полной порции
    assert not catch_up.budget_allows("kolesa", "enrich", 10**6, {"kolesa": edge, "cdn": 0})
    assert catch_up.budget_allows("kolesa", "enrich", 1, {"kolesa": edge, "cdn": 0})


def test_catch_up_status_thresholds_match_check_status():
    """Пороги staleness/recheck в catch_up.compute_gaps должны совпадать с
    check_status — иначе счётчик пробелов разошёлся бы с реальной выборкой
    джоба (показывал бы «есть что добрать», а джоб ничего бы не брал)."""
    from kz.ops import catch_up
    from kz.collect import check_status
    assert catch_up.STATUS_STALE_DAYS   == check_status.STALE_DAYS
    assert catch_up.STATUS_RECHECK_DAYS == check_status.RECHECK_DAYS


def test_status_recheck_and_listing_inference():
    """Логика статус-джоба (чистые предикаты, без сети):
      needs_status_check — терминал не трогаем; свежий в листинге не требует
        запроса; недавно проверенный остывает; пропал+давно не проверяли → да.
      infer_active_from_listing — показавшийся в листинге active без запроса;
        уже-active не переписываем; терминал реактивируем ТОЛЬКО если увиден
        ПОСЛЕ пометки терминальным."""
    from kz.collect.check_status import needs_status_check, infer_active_from_listing
    # needs_status_check(cur_status, seen_days, checked_days)
    assert not needs_status_check("archived", 30, None)     # терминал
    assert not needs_status_check("deleted", 30, 30)        # терминал
    assert not needs_status_check("active", 0, None)        # свежий в листинге
    assert not needs_status_check(None, 5, 1)               # проверяли вчера (<RECHECK)
    assert needs_status_check("active", 5, 10)              # пропал + давно не проверяли
    assert needs_status_check(None, 5, None)                # пропал, ни разу не проверяли
    # infer_active_from_listing(cur_status, seen_days, seen_after_check)
    assert infer_active_from_listing(None, 0, True)         # новый, свежий в листинге
    assert not infer_active_from_listing("active", 0, True) # уже active — не переписываем
    assert infer_active_from_listing("archived", 0, True)   # реактивация (виден ПОСЛЕ архива)
    assert not infer_active_from_listing("archived", 0, False)  # виден, но ДО архивации
    assert not infer_active_from_listing(None, 5, True)     # не свежий в листинге


def test_catch_up_budget_resets_next_day(tmp_path, monkeypatch):
    """Счётчик бюджета сбрасывается с новыми сутками, битый файл = ноль (не
    падаем), сегодняшняя запись читается как есть."""
    from kz.ops import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    f.write_text('{"date":"2000-01-01","kolesa":399,"cdn":5}', encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 0, "cdn": 0}   # старый день → сброс
    catch_up.save_budget_used({"kolesa": 150, "cdn": 300})
    assert catch_up.load_budget_used() == {"kolesa": 150, "cdn": 300}  # сегодня → как есть
    f.write_text("{ битый json", encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 0, "cdn": 0}   # не падаем


# ─── ML validation: дубли, время, калибровка и train/inference schema ───────
def test_duplicate_groups_keep_repost_in_one_fold():
    """Цена может поменяться у перезалива, но это всё ещё одна CV-группа."""
    import pandas as pd
    from kz.ml.train_price_model import duplicate_groups
    d = pd.DataFrame([
        {"ad_id": "1", "brand": "Toyota", "model": "Camry", "year": 2020,
         "mileage_km": 80000, "engine_volume": 2.5, "body_type": "седан",
         "text_full": "один хозяин, родной окрас, зимняя резина",
         "price_tenge": 10_000_000},
        {"ad_id": "2", "brand": "Toyota", "model": "Camry", "year": 2020,
         "mileage_km": 80000, "engine_volume": 2.5, "body_type": "седан",
         "text_full": "один хозяин, родной окрас, зимняя резина",
         "price_tenge": 9_700_000},
        # Без содержательного текста круглого пробега недостаточно для склейки.
        {"ad_id": "3", "brand": "Toyota", "model": "Camry", "year": 2020,
         "mileage_km": 80000, "engine_volume": 2.5, "body_type": "седан",
         "text_full": "", "price_tenge": 10_000_000},
    ])
    g = duplicate_groups(d)
    assert g.iloc[0] == g.iloc[1]
    assert g.iloc[2] != g.iloc[0]


def test_temporal_holdout_is_future_and_removes_group_overlap():
    import pandas as pd
    from kz.ml.train_price_model import duplicate_groups, temporal_holdout
    rows = []
    for i in range(120):
        rows.append({
            "ad_id": str(i), "scraped_at": pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=i), "brand": "B", "model": f"M{i}",
            "year": 2020, "mileage_km": i + 1000, "engine_volume": 2.0,
            "body_type": "седан", "text_full": f"уникальное описание машины номер {i}",
        })
    # Последняя строка — перезалив первой; первая должна быть удалена из train.
    rows[-1].update({
        "brand": rows[0]["brand"], "model": rows[0]["model"],
        "mileage_km": rows[0]["mileage_km"], "text_full": rows[0]["text_full"],
    })
    d = pd.DataFrame(rows)
    tr, te = temporal_holdout(d)
    assert d.loc[tr, "scraped_at"].max() < d.loc[te, "scraped_at"].min()
    assert set(duplicate_groups(d.loc[tr])).isdisjoint(duplicate_groups(d.loc[te]))


def test_residual_calibration_hits_requested_fraction():
    import numpy as np
    from kz.ml.residual_detector import calibration_offset
    y = np.linspace(-2, 2, 1001)
    raw = np.zeros_like(y)
    offset = calibration_offset(y, raw, alpha=0.10)
    frac = float((y < raw + offset).mean())
    assert abs(frac - 0.10) <= 1 / len(y)


def test_predict_row_matches_training_schema_and_zero_is_not_missing():
    from kz.ml.predict_price import make_row
    from kz.ml.train_price_model import FEATURES
    row = make_row(
        brand="Toyota", model="Camry", year=2020, mileage_km=0,
        engine_volume=2.5,
    )
    assert list(row.columns) == FEATURES
    assert row.loc[0, "is_mileage_missing"] == 0
    assert row.loc[0, "brand"] == "Toyota"


def test_model_artifacts_are_runtime_data_not_git_payload():
    from pathlib import Path
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/models/*.cbm" in ignore
    assert "data/models/*.json" in ignore


def test_labeling_queue_contains_positive_residual_and_control_strata():
    """Без random_control очередь не способна находить false negatives."""
    import pandas as pd
    from kz.report.explore import select_labeling_rows
    n = 27
    d = pd.DataFrame({
        "ad_id": [str(i) for i in range(n)],
        "is_suspicious": [1] * 3 + [0] * (n - 3),
        "both_detectors_low": [0] * n,
        "price_z": list(range(n)),
        "residual_gap": [0.0] * n,
    })
    residual = pd.Series([False] * 3 + [True] * 4 + [False] * 20)
    q = select_labeling_rows(d, residual, control_n=5)
    counts = q["sampling_stratum"].value_counts().to_dict()
    assert counts == {
        "random_control": 5,
        "residual_candidate": 4,
        "rule_positive": 3,
    }
    assert set(q.loc[q["sampling_stratum"] == "random_control",
                     "stratum_population"]) == {20}


# ─── pacing.py: вежливый ритм (politeness, не маскировка) ────────────────────

def test_pacing_never_faster_than_base_range():
    """Главная гарантия: пауза НИКОГДА не короче нижней границы — иначе
    «человечность» тайком повысила бы частоту запросов, а цель обратная."""
    import random as _r
    from kz.core import pacing
    lo, hi = 4.0, 8.0
    rng = _r.Random(0)
    pauses = [pacing.human_pause(lo, hi, rng=rng) for _ in range(2000)]
    assert min(pauses) >= lo
    assert max(pauses) <= hi * pacing.LONG_TAIL_MULT
    # и в среднем строго медленнее плоского uniform
    assert sum(pauses) / len(pauses) > (lo + hi) / 2


def test_pacing_long_break_cadence():
    """Перерыв ровно каждые BREAK_EVERY запросов, а не когда попало."""
    import random as _r
    from kz.core import pacing
    rng = _r.Random(1)
    hits = [i for i in range(1, 61) if pacing.long_break(i, rng=rng) is not None]
    assert hits == list(range(pacing.BREAK_EVERY, 61, pacing.BREAK_EVERY))
    assert pacing.long_break(0, rng=rng) is None       # i с 1, не с 0


def test_pacing_mean_pause_accounts_for_breaks():
    """mean_pause честно учитывает хвост и перерывы (иначе ETA врёт)."""
    from kz.core import pacing
    assert pacing.mean_pause(4.0, 8.0) > 6.0          # больше плоского среднего


def test_kolesa_jobs_use_shared_pacing():
    """Все три kolesa-джоба ходят через pacing, а не через свой time.sleep(
    random.uniform(...)) — иначе политика ритма разъедется по файлам."""
    from pathlib import Path
    for f in ["kz/collect/enrich.py", "kz/collect/check_status.py", "kz/collect/backfill_avgprice.py"]:
        src = Path(f).read_text(encoding="utf-8")
        assert "pacing.polite_sleep" in src, f
        assert "time.sleep(random.uniform" not in src, f


# ─── catch_up: настраиваемый бюджет и зоны риска ─────────────────────────────

def test_catch_up_parse_budget_forms():
    import pytest as _pt
    from kz.ops import catch_up
    assert catch_up.parse_budget(["kz/ops/catch_up.py"]) is None
    assert catch_up.parse_budget(["x", "--budget", "300"]) == 300
    assert catch_up.parse_budget(["x", "--budget=450"]) == 450
    for bad in (["x", "--budget", "abc"], ["x", "--budget=0"], ["x", "--budget=-5"]):
        with _pt.raises(SystemExit):
            catch_up.parse_budget(bad)


def test_catch_up_risk_zones_match_observed_ban():
    """Зоны откалиброваны на реальном факте: ~270 запросов = бан 2026-07-23.
    Дефолт обязан лежать в безопасной зоне."""
    from kz.ops import catch_up
    assert catch_up.risk_zone(50)[0] == "спокойно"
    assert catch_up.risk_zone(catch_up.DEFAULT_KOLESA_BUDGET)[0] == "безопасно"
    assert catch_up.risk_zone(270)[0] == "риск"
    assert catch_up.risk_zone(500)[0] == "высокий риск"
    # монотонность: больше запросов не может быть «менее рискованно»
    order = [z[1] for z in catch_up.RISK_ZONES]
    seen = [catch_up.risk_zone(n)[0] for n in (1, 100, 101, 200, 201, 270, 271, 10**6)]
    assert [order.index(s) for s in seen] == sorted(order.index(s) for s in seen)


def test_catch_up_eta_grows_with_volume():
    from kz.ops import catch_up
    assert catch_up.eta_minutes(0) == 0
    assert catch_up.eta_minutes(540) > catch_up.eta_minutes(200) > 0


# ─── label_cards.py: офлайн-разметка (мёртвые страницы + не жжёт лимит) ──────

def test_label_cards_never_requests_kolesa():
    """Карточки — офлайн-инструмент: генератор не делает HTTP-запросов
    вообще (фото подставляются как URL и грузятся браузером с CDN)."""
    from pathlib import Path
    src = Path("kz/report/label_cards.py").read_text(encoding="utf-8")
    for bad in ("requests.get", "requests.head", "urlopen", "httpx"):
        assert bad not in src, bad


def test_label_cards_help_covers_real_flags():
    """У каждого флага, который детектор реально ставит, должна быть
    подсказка «как решать» — иначе разметчик остаётся без критерия."""
    from kz.report import label_cards
    from kz.transform import clean
    from pathlib import Path
    src = Path("kz/transform/clean.py").read_text(encoding="utf-8")
    # флаги подозрения, реально встречающиеся в коде детектора
    for flag in ["price_anomaly_low", "young_car_cheap", "possible_repost",
                 "shared_photo_diff_car", "used_but_zero_mileage",
                 "cheap_and_urgent"]:
        assert flag in src, f"{flag} исчез из clean.py — обнови FLAG_HELP"
        assert flag in label_cards.FLAG_HELP, f"нет подсказки для {flag}"
    # подсказка обязана различать fraud/legit, а не просто описывать флаг
    for flag, (what, fr, lg) in label_cards.FLAG_HELP.items():
        assert what and fr and lg, flag
        assert "fraud" in fr and "legit" in lg, flag


def test_label_cards_csv_line_matches_labels_schema():
    """Строка, которую собирает страница (ad_id + 7 пустых + verdict,comment),
    должна ложиться ровно в схему manual_labels.csv — иначе clean.py не
    прочитает вердикт."""
    import csv
    from io import StringIO
    from kz.report import label_cards as lc
    # Схему берём из кода, а не из data/manual_labels.csv: тот в .gitignore,
    # и в CI на чистом клоне его нет — тест падал FileNotFoundError.
    header = lc.journal_header()
    assert header[0] == "ad_id"
    assert header[-2:] == ["verdict", "comment"]
    # шаблон из JS: id + ',,,,,,,,' + verdict + ',' + comment
    line = "123" + "," * 8 + "legit,причина"
    assert len(next(csv.reader(StringIO(line)))) == len(header)


def test_label_cards_money_and_fmt_handle_missing():
    """Пропуски — норма в этих данных (37% без пробега): формат не должен
    печатать 'nan' в карточке."""
    from kz.report import label_cards
    assert label_cards.money(None) == "—"
    assert label_cards.money(float("nan")) == "—"
    assert label_cards.fmt(None) == "—"
    assert label_cards.fmt(float("nan")) == "—"
    assert label_cards.fmt("") == "—"
    assert label_cards.fmt(2007.0) == "2007"      # не 2007.0


def test_label_cards_money_reads_naturally():
    """Миллионы — только от миллиона: «0.24М ₸» для 240 000 читается хуже,
    а дешёвых объявлений среди подозрительных больше всего."""
    from kz.report import label_cards as lc
    assert lc.money(240000) == "240 000 ₸"
    assert lc.money(95000) == "95 000 ₸"
    assert lc.money(1_000_000) == "1М ₸"
    assert lc.money(4_900_000) == "4.9М ₸"
    assert lc.money(12_000_000) == "12М ₸"


def test_label_cards_price_bands_are_monotonic():
    """Полосы не должны спорить с процентом (был баг: «60% от среднего —
    цена в норме»). Чем дешевле относительно рынка, тем «ниже» ярлык."""
    from kz.report import label_cards as lc
    labels = [lab for _, lab in lc.PRICE_BANDS]
    seen = [lc.price_band(r) for r in (0.2, 0.59, 0.61, 0.84, 0.9, 1.2, 1.5, 9.0)]
    idx = [labels.index(s) for s in seen]
    assert idx == sorted(idx)
    assert lc.price_band(0.59) == "сильно дешевле рынка"
    assert lc.price_band(1.0) == "в пределах среднего"
    # границы включаются в верхнюю полосу, а не выпадают
    assert lc.price_band(0.60) == "заметно ниже среднего"
    assert lc.price_band(1.40).startswith("существенно выше")


def test_label_cards_gallery_and_keyboard_present():
    """Ключевая эргономика: крупное фото + миниатюры + лайтбокс + шорткаты.
    Раньше были только 190px-миниатюры, по которым состояние не оценить."""
    from pathlib import Path
    src = Path("kz/report/label_cards.py").read_text(encoding="utf-8")
    for token in ['class="hero"', 'class="thumb', 'id="box"', "openBox",
                  "setVerdict", "focusCard"]:
        assert token in src, token
    # шаблон должен быть СЫРОЙ строкой, иначе \n в JS сломается
    assert 'TEMPLATE = r"""' in src


def test_catch_up_budget_charges_to_calendar_day(tmp_path, monkeypatch):
    """Расход списывается на текущие сутки, а не на день старта прогона.
    Реальный случай 2026-07-30: --until-done пересёк полночь, и 400
    вчерашних запросов записались сегодняшним числом, съев новую квоту."""
    from kz.ops import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    assert catch_up.charge_budget("kolesa", 20) == {"kolesa": 20, "cdn": 0}
    assert catch_up.charge_budget("kolesa", 20) == {"kolesa": 40, "cdn": 0}
    assert catch_up.charge_budget("cdn", 300)["cdn"] == 300
    assert catch_up.load_budget_used() == {"kolesa": 40, "cdn": 300}
    # вчерашняя запись не влияет на сегодняшний расход
    import json
    days = json.loads(f.read_text(encoding="utf-8"))["days"]
    days["2000-01-01"] = {"kolesa": 999, "cdn": 999}
    f.write_text(json.dumps({"days": days}), encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 40, "cdn": 300}


def test_catch_up_budget_file_reads_old_format(tmp_path, monkeypatch):
    """Файл прежнего формата не должен терять сегодняшний расход при
    обновлении кода — иначе квота молча удвоится."""
    from kz.ops import catch_up
    from datetime import date
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    f.write_text('{"date":"%s","kolesa":180,"cdn":7}' % date.today().isoformat(),
                 encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 180, "cdn": 7}
    assert catch_up.charge_budget("kolesa", 20)["kolesa"] == 200


def test_catch_up_budget_file_keeps_history_bounded(tmp_path, monkeypatch):
    import json
    from kz.ops import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    days = {f"2026-01-{d:02d}": {"kolesa": 1, "cdn": 0} for d in range(1, 21)}
    f.write_text(json.dumps({"days": days}), encoding="utf-8")
    catch_up.charge_budget("kolesa", 1)
    kept = json.loads(f.read_text(encoding="utf-8"))["days"]
    assert len(kept) <= catch_up.BUDGET_KEEP_DAYS


def test_catch_up_per_run_cap_blocks_midnight_burst():
    """Сутки честно обнуляются в полночь, поэтому нужен второй потолок — на
    сам запуск. Иначе прогон, начатый в 23:50, выдал бы двойную квоту
    всплеском за двадцать минут, а банят именно за всплеск объёма."""
    from kz.ops import catch_up
    B = catch_up.DAILY_BUDGET["kolesa"]
    fresh_day = {"kolesa": 0, "cdn": 0}          # после полуночи расход суток 0
    spent_run = {"kolesa": B, "cdn": 0}          # но прогон уже выбрал квоту
    assert not catch_up.budget_allows("kolesa", "enrich", 10**6,
                                     fresh_day, spent_run)
    # без учёта прогона (обычный одиночный запуск) — разрешено
    assert catch_up.budget_allows("kolesa", "enrich", 10**6, fresh_day)
    assert catch_up.budget_allows("kolesa", "enrich", 10**6, fresh_day,
                                 {"kolesa": 0, "cdn": 0})


# ─── Сентинелы avgPrice=-1 и бейдж="-" не должны считаться значениями ────────

def test_avgprice_sentinel_never_acts_as_price():
    """-1 в kolesa_avg_price = «у модели нет эталона», НЕ цена. Кросс-чек
    обязан его игнорировать: иначе -1 сравнивалось бы с ценой объявления и
    снимало флаг у всех подряд (price >= 0.80 * -1 верно всегда)."""
    import numpy as np
    import pandas as pd
    from kz.transform.clean import exculpate
    # exculpate() читает весь набор колонок clean-слоя; наполняем нейтрально,
    # чтобы проверялся ровно один фактор — сентинел.
    base = dict(stat_reasons="price_anomaly_low", rule_reasons="",
                info_flags="", suspicion_reasons="", is_suspicious=1,
                price_tenge=1_000_000, text_full="", condition="",
                labels="", customs_cleared="Да")
    df = pd.DataFrame([
        {**base, "kolesa_avg_price": -1},          # сентинел → флаг остаётся
        {**base, "kolesa_avg_price": np.nan},      # не обогащено → остаётся
        {**base, "kolesa_avg_price": 1_100_000},   # реальный эталон → снимается
        {**base, "kolesa_avg_price": 5_000_000},   # сильно дешевле → остаётся
    ])
    out = exculpate(df.copy())
    assert list(out["stat_reasons"]) == [
        "price_anomaly_low", "price_anomaly_low", "", "price_anomaly_low"]
    assert "kolesa_price_ok" in out.loc[2, "info_flags"]
    assert "kolesa_price_ok" not in out.loc[0, "info_flags"]


def test_badge_sentinel_never_exculpates():
    """Бейдж "-" = «проверено, бейджа нет». Он не должен попадать под
    «аварийная/не на ходу» и снимать подозрение."""
    import numpy as np
    import pandas as pd
    from kz.transform.clean import exculpate
    # exculpate() читает весь набор колонок clean-слоя; наполняем нейтрально,
    # чтобы проверялся ровно один фактор — сентинел.
    base = dict(stat_reasons="price_anomaly_low", rule_reasons="",
                info_flags="", suspicion_reasons="", is_suspicious=1,
                price_tenge=1_000_000, text_full="", condition="",
                labels="", customs_cleared="Да")
    df = pd.DataFrame([
        {**base, "page_status_badge": "-"},
        {**base, "page_status_badge": np.nan},
        {**base, "page_status_badge": "Аварийная/Не на ходу"},
    ])
    out = exculpate(df.copy())
    assert list(out["stat_reasons"]) == ["price_anomaly_low", "price_anomaly_low", ""]


def test_avgprice_and_badge_stay_out_of_model():
    """Оценка kolesa — валидатор детектора, а НЕ признак модели цены:
    иначе модель просто копировала бы kolesa (target leakage)."""
    from kz.ml.train_price_model import FEATURES
    assert "kolesa_avg_price" not in FEATURES
    assert "page_status_badge" not in FEATURES


def test_label_cards_hint_ignores_sentinel():
    """В карточке разметки сентинел не должен показываться как «средняя цена»."""
    from kz.report import label_cards as lc
    assert lc.price_verdict_hint({"kolesa_avg_price": -1,
                                  "price_tenge": 1_000_000}) == ""
    assert lc.price_verdict_hint({"kolesa_avg_price": 2_000_000,
                                  "price_tenge": 1_000_000}) != ""


# ─── Airflow DAG'и: сетевой не должен запускаться сам ────────────────────────

def test_network_dag_is_paused_and_single_run():
    """DAG, ходящий на kolesa.kz, обязан создаваться ВЫКЛЮЧЕННЫМ: иначе
    scheduler запустит его по расписанию и пойдёт скрейпить без присмотра
    (домашний IP уже ловил блокировку за объём 2026-07-23). Проверка по
    исходнику: airflow не установлен в основной venv."""
    from pathlib import Path
    src = Path("airflow/dags/kolesa_pipeline_dag.py").read_text(encoding="utf-8")
    assert "is_paused_upon_creation=True" in src
    assert "max_active_runs=1" in src        # два прогона = двойная частота
    assert "catchup=False" in src            # иначе досчитает все пропущенные дни


def test_offline_dag_has_no_network_jobs():
    """Офлайн-DAG безопасно запускать в любой момент. Если в него заедет
    сетевой модуль, «безобидный» пересчёт начнёт тратить суточный лимит."""
    from pathlib import Path
    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    for net in ["kz.collect.parser", "kz.collect.enrich", "kz.collect.check_status",
                "kz.collect.photo_dedup", "kz.collect.backfill_avgprice",
                "kz.ops.catch_up"]:
        assert net not in src, f"{net} — сетевой, ему не место в офлайн-DAG"
    assert "schedule=None" in src                    # сам не стартует
    assert "is_paused_upon_creation=False" in src    # но ручной запуск исполнится


def test_offline_dag_covers_whole_ml_chain():
    """DAG и оркестратор не должны разъезжаться: если в run_all появился шаг,
    а в DAG его нет, Airflow-прогон молча посчитает не всё."""
    from pathlib import Path
    from kz.ops.run_all import ML_CHAIN, OFFLINE_CHAIN
    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    for _, cmd in ML_CHAIN + OFFLINE_CHAIN:
        assert cmd[-1] in src, f"{cmd[-1]} есть в run_all, но нет в офлайн-DAG"


def test_offline_dag_dependencies_respect_artifacts():
    """Граф DAG'а обязан уважать зависимости по артефактам: графики читают
    модель цены, отчёт — модель И ценовой пол. Иначе таск упадёт в проде на
    отсутствующем файле, хотя в UI выглядел независимым."""
    from pathlib import Path
    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    assert "clean >> train >> dashboard" in src
    assert "train >> residual >> report" in src
    assert "clean >> explore >> cards" in src


def test_collect_dag_delegates_budget_to_catch_up():
    """Сетевой добор должен идти ОДНИМ таском через catch_up, а не отдельными
    тасками на джоб: иначе Airflow запустил бы их параллельно, и суточный
    лимит на хост перестал бы соблюдаться — все стучатся в kolesa с одного IP."""
    from pathlib import Path
    src = Path("airflow/dags/kolesa_pipeline_dag.py").read_text(encoding="utf-8")
    assert "kz.ops.catch_up" in src
    for direct in ["kz.collect.enrich", "kz.collect.check_status",
                   "kz.collect.photo_dedup", "kz.collect.backfill_avgprice"]:
        assert direct not in src, (
            f"{direct} вызван напрямую — обойдёт суточный лимит catch_up")


# ─── label_cards: сохранение вердиктов в журнал ──────────────────────────────

def _tmp_journal(tmp_path, monkeypatch):
    """Синтетический журнал в tmp: тесты не касаются настоящего (правило №1)
    и не зависят от него.

    Раньше здесь копировался реальный data/manual_labels.csv, и в CI тесты
    падали FileNotFoundError: data/ в .gitignore, в чистом клоне файла нет.
    Тест, который проверяет логику, не должен требовать чужих данных.
    """
    import csv
    from kz.report import label_cards as lc
    dst = tmp_path / "manual_labels.csv"
    header = ["ad_id", "url", "title", "year", "price_tenge", "mileage_km",
              "suspicion_reasons", "seller_comment", "verdict", "comment"]
    rows = [
        # заполненный вердикт и пустые заглушки из очереди — как в жизни
        ["225936503", "https://kolesa.kz/a/show/225936503", "Chevrolet Onix",
         "2023", "1700000", "100000", "young_car_cheap", "Оникс аварийный",
         "legit", "честно битая"],
        ["225480956", "https://kolesa.kz/a/show/225480956", "Toyota Highlander",
         "2008", "4900000", "300030", "price_anomaly_low", "на ходу", "", ""],
        ["226154999", "https://kolesa.kz/a/show/226154999", "Hyundai Accent",
         "2012", "1500000", "236500", "price_anomaly_low",
         "Был пожар, документы в порядке", "", ""],
    ]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    monkeypatch.setattr(lc, "LABELS_CSV", str(dst))
    monkeypatch.setattr(lc, "LABELS_PREV", str(tmp_path / "prev.csv"))
    monkeypatch.setattr(lc, "_snapshot_done", False)
    return lc, dst


def test_upsert_keeps_one_row_per_ad(tmp_path, monkeypatch):
    """Повторный вердикт ОБНОВЛЯЕТ строку, а не плодит новую. Было наоборот,
    и на одно объявление накопилось четыре строки с разными вердиктами —
    clean.py брал последнюю, то есть считал верно, но журнал стал нечитаемым."""
    import csv
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    n0 = len(list(csv.DictReader(dst.open(encoding="utf-8"))))
    lc.upsert_verdict("111", "legit", "сначала так", {})
    n1 = len(list(csv.DictReader(dst.open(encoding="utf-8"))))
    lc.upsert_verdict("111", "fraud", "передумал", {})
    lc.upsert_verdict("111", "unknown", "не понять", {})
    rows = list(csv.DictReader(dst.open(encoding="utf-8")))
    assert n1 == n0 + 1                     # новое объявление добавилось один раз
    assert len(rows) == n1                  # повторы НЕ добавили строк
    mine = [r for r in rows if r["ad_id"] == "111"]
    assert len(mine) == 1
    assert (mine[0]["verdict"], mine[0]["comment"]) == ("unknown", "не понять")


def test_upsert_updates_existing_queue_row_in_place(tmp_path, monkeypatch):
    """Правится строка, уже стоящая в журнале на своём месте из очереди, —
    порядок файла не съезжает и описательные колонки не теряются."""
    import csv
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    rows = list(csv.DictReader(dst.open(encoding="utf-8")))
    existing = next(r for r in rows if not r["verdict"])
    pos = rows.index(existing)
    lc.upsert_verdict(existing["ad_id"], "legit", "проверено", {})
    after = list(csv.DictReader(dst.open(encoding="utf-8")))
    assert len(after) == len(rows)                    # ни одной новой строки
    assert after[pos]["ad_id"] == existing["ad_id"]   # осталась на месте
    assert after[pos]["verdict"] == "legit"
    assert after[pos]["title"] == existing["title"]   # факты не перезаписаны


def test_upsert_preserves_other_rows_and_backup(tmp_path, monkeypatch):
    """Правка одной строки не должна менять остальные, а прежняя версия
    журнала обязана остаться рядом: это ручной ground truth, его нельзя
    потерять, и в git он не лежит (data/ в .gitignore)."""
    import csv
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    prev = tmp_path / "manual_labels.prev.csv"
    monkeypatch.setattr(lc, "LABELS_PREV", str(prev))
    monkeypatch.setattr(lc, "_snapshot_done", False)
    before_text = dst.read_text(encoding="utf-8")
    before = list(csv.DictReader(dst.open(encoding="utf-8")))
    lc.upsert_verdict("111", "legit", "", {})
    after = list(csv.DictReader(dst.open(encoding="utf-8")))
    for a, b in zip(before, after):                   # все прежние строки целы
        assert a == b
    assert prev.exists() and prev.read_text(encoding="utf-8") == before_text


def test_upsert_writes_ints_without_dot_zero(tmp_path, monkeypatch):
    """pandas round-trip делал из 50 строку "50.0" и ронял вставку в
    INTEGER-колонку — поэтому журнал пишется csv-модулем (правило №4)."""
    import csv
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    lc.upsert_verdict("111", "fraud", "", {
        "year": 1994.0, "price_tenge": 240000.0,
        "mileage_km": float("nan"),
        "seller_comment": 'текст с "кавычками", запятой'})
    row = [r for r in csv.DictReader(dst.open(encoding="utf-8"))
           if r["ad_id"] == "111"][0]
    assert row["year"] == "1994"
    assert row["price_tenge"] == "240000"
    assert row["mileage_km"] == ""            # пропуск, а не "nan"
    assert row["seller_comment"] == 'текст с "кавычками", запятой'


def test_upsert_rejects_bad_verdict(tmp_path, monkeypatch):
    """В журнал попадают только fraud/legit/unknown — иначе clean.py молча
    проигнорирует строку, и разметчик решит, что вердикт учтён."""
    import pytest as _pt
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    before = dst.read_text(encoding="utf-8")
    for bad in ("мошенник", "FRAUD", "", "legit "):
        with _pt.raises(ValueError):
            lc.upsert_verdict("111", bad, "", {})
    assert dst.read_text(encoding="utf-8") == before   # файл вообще не тронут


def test_dedupe_journal_collapses_and_keeps_last_verdict(tmp_path, monkeypatch):
    """Сворачивание накопленных дубликатов: одна строка на объявление,
    побеждает последний НЕПУСТОЙ вердикт (финальный выбор человека), а
    пустая строка-заглушка из очереди его не затирает."""
    import csv
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    monkeypatch.setattr(lc, "LABELS_PREV", str(tmp_path / "prev.csv"))
    monkeypatch.setattr(lc, "_snapshot_done", False)
    header, rows = lc.read_journal()
    base = dict(rows[0])
    aid = base["ad_id"]
    for v, c in [("fraud", "раз"), ("legit", "два"), ("", "")]:
        r = dict(base); r["verdict"] = v; r["comment"] = c
        rows.append(r)
    lc.write_journal(header, rows)
    uniq = len({str(r["ad_id"]) for r in rows})
    before, after = lc.dedupe_journal()
    # Точное число снятых строк не фиксируем: в реальном журнале дубликаты
    # уже могли накопиться. Инвариант — ровно одна строка на объявление.
    assert after == uniq < before
    got = [r for r in csv.DictReader(dst.open(encoding="utf-8"))
           if r["ad_id"] == aid]
    assert len(got) == 1
    assert (got[0]["verdict"], got[0]["comment"]) == ("legit", "два")


def test_serve_only_accepts_shown_ads(tmp_path, monkeypatch):
    """Сервер пишет в журнал только по ad_id из показанных карточек: тело
    запроса не должно решать, что попадёт в ground truth."""
    import json, threading, time, urllib.error, urllib.request
    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    facts = {"111": {"title": "Audi 80", "year": 1994}}
    threading.Thread(target=lc.serve, args=("<p>x</p>", facts, 8798),
                     daemon=True).start()
    time.sleep(0.8)

    def post(payload):
        req = urllib.request.Request(
            "http://127.0.0.1:8798/verdict", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    n = dst.read_text(encoding="utf-8").count("\n")
    assert post({"ad_id": "111", "verdict": "legit", "comment": "ок"}) == 200
    assert post({"ad_id": "999", "verdict": "legit", "comment": ""}) == 400
    assert post({"ad_id": "111", "verdict": "hack", "comment": ""}) == 400
    assert post({}) == 400
    assert dst.read_text(encoding="utf-8").count("\n") == n + 1   # только валидный


def test_file_mode_page_cannot_write_journal():
    """В файловом режиме SERVER=false: страница не должна делать вид, что
    пишет в журнал, если писать физически некуда."""
    from kz.report import label_cards as lc
    import pandas as pd
    rows = pd.DataFrame([{
        "ad_id": "1", "brand": "Audi", "model": "80", "year": 1994,
        "price_tenge": 240000, "photos": [], "status": "active",
        "existing_verdict": None, "suspicion_reasons": "price_anomaly_low",
        "price_z": -4.0,
    }])
    assert "const SERVER = false;" in lc.build(rows, serve_mode=False)
    assert "const SERVER = true;" in lc.build(rows, serve_mode=True)


def test_code_fingerprint_survives_file_moves(tmp_path):
    """Отпечаток обучающего кода не должен зависеть от РАСПОЛОЖЕНИЯ файлов.

    Реальный баг переезда в пакет: fingerprint брался по строкам-путям
    ("train_price_model.py"), и обучение падало с FileNotFoundError, как
    только файлы переехали. Теперь передаётся __file__, а в хэш идёт только
    имя файла — тот же код в другой папке даёт тот же отпечаток."""
    from kz.ml.train_price_model import code_fingerprint
    a = tmp_path / "one" / "mod.py"
    b = tmp_path / "two" / "mod.py"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("x = 1\n", encoding="utf-8")
    assert code_fingerprint(str(a)) == code_fingerprint(str(b))
    b.write_text("x = 2\n", encoding="utf-8")
    assert code_fingerprint(str(a)) != code_fingerprint(str(b))   # код важен


def test_fingerprint_inputs_are_resolvable():
    """Файлы, по которым считается отпечаток, должны реально существовать —
    иначе обучение падает только в момент сохранения артефакта, в самом конце."""
    from pathlib import Path
    from kz.ml import residual_detector, train_price_model
    from kz.transform import data_quality
    for m in (train_price_model, residual_detector, data_quality):
        assert Path(m.__file__).exists(), m.__name__


def test_no_flat_module_imports_left():
    """После переезда в пакет плоских импортов остаться не должно: они
    сработали бы только при запуске из корня старым способом и тихо
    разошлись бы с пакетными."""
    import re
    from pathlib import Path
    flat = {"db", "config", "pacing", "clean", "damage", "enrich", "parser",
            "check_status", "photo_dedup", "backfill_avgprice", "explore",
            "data_quality", "text_features", "train_price_model",
            "residual_detector", "predict_price", "time_to_sell",
            "label_cards", "evaluate_detector", "catch_up", "run_all",
            "pipeline_status", "migrate_to_postgres", "ml_report",
            "ml_dashboard"}
    bad = []
    for p in list(Path("kz").rglob("*.py")) + list(Path("tests").glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"\s*(?:from (\w+) import |import (\w+)(?: as \w+)?$)", line)
            if m and (m.group(1) or m.group(2)) in flat:
                bad.append(f"{p}:{i}: {line.strip()}")
    assert not bad, "плоские импорты:\n" + "\n".join(bad)


def test_dag_commands_use_package_modules():
    """DAG'и запускают код строками shell, и переезд в пакет их не правит
    автоматически. Реальный случай: внутри python -c остался «from db import»,
    и таск падал ModuleNotFoundError уже в контейнере, а не на тестах."""
    import re
    from pathlib import Path
    flat = ("db", "config", "pacing", "clean", "enrich", "parser", "explore",
            "check_status", "photo_dedup", "backfill_avgprice", "label_cards",
            "catch_up", "run_all", "data_quality", "train_price_model")
    bad = []
    for dag in Path("airflow/dags").glob("*.py"):
        text = dag.read_text(encoding="utf-8")
        for mod in flat:
            # плоский импорт внутри строки shell-команды
            if re.search(rf"from {mod} import ", text):
                bad.append(f"{dag.name}: from {mod} import")
            # запуск файла вместо модуля
            if f"python {mod}.py" in text:
                bad.append(f"{dag.name}: python {mod}.py")
    assert not bad, "DAG ссылается на плоские модули:\n" + "\n".join(bad)


def test_learning_curve_subsample_keeps_groups_whole():
    """Подвыборка для кривой обучения берётся ЦЕЛЫМИ группами дублей: иначе
    перезалив одной машины попал бы и в train, и в test, и кривая
    завысила бы качество на малых долях — то есть соврала бы именно там,
    где мы решаем, стоит ли собирать ещё данные."""
    import pandas as pd
    from kz.ml.learning_curve import subsample_by_groups
    df = pd.DataFrame({"x": range(100)})
    groups = pd.Series([f"g{i//4}" for i in range(100)])   # по 4 строки в группе
    part, g = subsample_by_groups(df, groups, 0.5, seed=1)
    # каждая попавшая группа представлена ПОЛНОСТЬЮ
    for name, size in g.value_counts().items():
        assert size == (groups == name).sum(), name
    assert 0 < len(part) < len(df)
    whole, gw = subsample_by_groups(df, groups, 1.0)
    assert len(whole) == len(df)


# ─── Оркестратор: порядок шагов задан зависимостями по артефактам ────────────

def test_ml_chain_order_respects_artifacts():
    """Графики и HTML-отчёт читают СОХРАНЁННЫЕ артефакты, поэтому обучение и
    калибровка пола обязаны идти раньше, а отчёт — после обоих. Переставь
    шаги местами, и цепочка упадёт на FileNotFoundError уже в проде."""
    from kz.ops.run_all import ML_CHAIN
    order = [cmd[-1] for _, cmd in ML_CHAIN]        # имена модулей по порядку
    i = {m: n for n, m in enumerate(order)}
    assert i["kz.ml.train_price_model"] < i["kz.report.ml_dashboard"]
    assert i["kz.ml.train_price_model"] < i["kz.report.ml_report"]
    assert i["kz.ml.residual_detector"] < i["kz.report.ml_report"]


def test_offline_chain_rebuilds_before_reporting():
    """clean пересобирает clean_data (в т.ч. подхватывает новые вердикты),
    очередь строится после него, карточки — последними. Иначе размечать
    пришлось бы по устаревшему списку."""
    from kz.ops.run_all import OFFLINE_CHAIN
    order = [cmd[-1] for _, cmd in OFFLINE_CHAIN]
    assert order == ["kz.transform.clean", "kz.report.explore",
                     "kz.report.label_cards"]


def test_ml_and_offline_chains_never_touch_network():
    """--ml и --fast обязаны быть офлайн: пересчёт после разметки не должен
    тратить суточный лимит запросов к kolesa."""
    from kz.ops.run_all import ML_CHAIN, OFFLINE_CHAIN
    net = {"kz.collect.parser", "kz.collect.check_status", "kz.collect.enrich",
           "kz.collect.photo_dedup", "kz.collect.backfill_avgprice"}
    for _, cmd in ML_CHAIN + OFFLINE_CHAIN:
        assert cmd[-1] not in net, cmd[-1]


def test_ml_flag_implies_offline_rebuild():
    """--ml считает по clean_data, поэтому обязан включать пересборку и не
    обязан ходить в сеть: в коде это выражено как fast = --fast or --ml."""
    from pathlib import Path
    src = Path("kz/ops/run_all.py").read_text(encoding="utf-8")
    assert 'fast  = "--fast" in sys.argv or ml' in src
