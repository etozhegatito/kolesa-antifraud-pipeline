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
    from kz.transform.damage import has_damage
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


def test_enrich_parameters_tolerate_layout_and_label_variants():
    """Пары могут лежать в одном dl, а подписи — содержать NBSP/двоеточие.

    Старый цикл брал первый dt/dd из общего dl и молча терял следующие поля.
    """
    html = """
    <dl class="offer__parameters">
      <dt>Город&nbsp;:</dt><dd>Алматы</dd>
      <dt>Состояние автомобиля:</dt><dd>б/у</dd>
      <dt>VIN-код</dt><dd>JTDBR32E720012345</dd>
    </dl>
    """
    parsed = enrich.parse_ad_page(html)
    assert parsed["page_city"] == "Алматы"
    assert parsed["page_condition"] == "б/у"
    assert parsed["has_vin"] == "Да"
    assert "JTDBR32E720012345" not in str(parsed)


def test_enrich_vin_history_is_positive_only_evidence():
    """Карточка не раскрывает VIN, но явно сообщает о VIN-backed истории.

    Обычная кнопка «Проверить Историю авто» может быть рекламой услуги и не
    считается доказательством. Без явного маркера значение остаётся NULL.
    """
    positive = enrich.parse_ad_page(
        "<section>У этого объявления есть История авто</section>"
    )
    unknown = enrich.parse_ad_page("<a>Проверить Историю авто</a>")
    assert positive["has_vin"] == "Да"
    assert "has_vin" not in unknown


def test_enrich_explicit_missing_vin_is_not_positive():
    html = "<dl><dt>VIN:</dt><dd>не указан</dd></dl>"
    assert enrich.parse_ad_page(html)["has_vin"] == "Нет"


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


# ─── Анализ выживаемости: дата публикации и цензурирование ──────────────────
def test_parse_posted_date():
    from datetime import date
    from kz.ml.survival import parse_posted
    today = date(2026, 8, 10)
    assert parse_posted("18 июля", today) == date(2026, 7, 18)
    assert parse_posted("18 июл.", today) == date(2026, 7, 18)   # сокращение
    assert parse_posted("5 мая", today) == date(2026, 5, 5)
    assert parse_posted("сегодня", today) is None    # относительная — не дата
    assert parse_posted(None, today) is None
    assert parse_posted("99 июля", today) is None    # невалидный день


def test_posted_date_rolls_back_over_new_year():
    """Kolesa пишет дату без года. Наивное «текущий год» ломается на стыке:
    объявление от 28 декабря, разобранное 3 января, получало дату на год
    вперёд. Дальше срок жизни выходил отрицательным, и строка молча
    выпадала из анализа выживаемости — потеря данных без единого warning."""
    from datetime import date
    from kz.ml.survival import parse_posted
    jan = date(2026, 1, 3)
    assert parse_posted("28 декабря", jan) == date(2025, 12, 28)
    assert parse_posted("2 января", jan) == date(2026, 1, 2)     # уже было
    assert parse_posted("3 января", jan) == date(2026, 1, 3)     # сегодня


def _survival_fixture():
    """Три таблицы в том же виде, в каком их отдаёт база."""
    cd = pd.DataFrame({
        "ad_id": ["a", "b", "c", "d"],
        "posted_date": ["1 июля", "1 июля", "1 июля", "1 июля"],
        "status": ["archived", "deleted", "active", "active"],
        "price_tenge": [5e6, 6e6, 7e6, None],
    })
    st = pd.DataFrame({
        "ad_id": ["a", "b", "c", "d"],
        "checked_at": ["2026-07-11", "2026-07-06", None, None],
    })
    sg = pd.DataFrame({
        "ad_id": ["a", "b", "c", "d"],
        "last_seen": ["2026-07-11", "2026-07-06", "2026-07-21", "2026-07-21"],
    })
    return cd, st, sg


def test_censored_ads_are_not_counted_as_sold():
    """Главная ошибка, ради которой и берут анализ выживаемости.

    Объявление, которое ещё висит, — это НЕ «продано сегодня» и не строка
    для выбрасывания. Оно наблюдалось столько-то дней и всё ещё живо. Если
    посчитать его событием, кривая выживания поедет вниз и метод соврёт
    ровно в том, ради чего его брали."""
    from kz.ml.survival import build_lifespans
    d = build_lifespans(*_survival_fixture())

    ev = dict(zip(d.ad_id, d.event, strict=True))
    assert ev["a"] == 1 and ev["b"] == 1      # archived и deleted — события
    assert ev["c"] == 0                       # active — цензурировано, не событие
    assert "d" not in ev                      # без цены в анализ не берём

    days = dict(zip(d.ad_id, d.days, strict=True))
    assert days["a"] == 10                    # 1 → 11 июля, дата проверки
    assert days["c"] == 20                    # 1 → 21 июля, последняя встреча


def test_lifespan_end_comes_from_the_right_column():
    """Для ушедших конец наблюдения — дата проверки, для живых — последняя
    встреча в листинге. Перепутать эти две колонки означает мерить не срок
    жизни объявления, а расписание нашего парсера."""
    from kz.ml.survival import build_lifespans
    cd, st, sg = _survival_fixture()
    # у живого объявления checked_at заполнен И отличается от last_seen
    st.loc[st.ad_id == "c", "checked_at"] = "2026-07-05"
    d = build_lifespans(cd, st, sg)
    assert dict(zip(d.ad_id, d.days, strict=True))["c"] == 20   # взяли last_seen, не checked_at


def test_kaplan_meier_matches_plain_fraction_without_censoring():
    """Проверка на понятном частном случае: когда цензурирования нет,
    Каплан-Мейер обязан совпасть с обычной долей ещё живых. Если бы
    совпадения не было, ошибка сидела бы в самом методе, а не в данных."""
    from kz.ml.survival import kaplan_meier
    d = pd.DataFrame({"days": [2, 4, 6, 8, 10], "event": [1, 1, 1, 1, 1]})
    km = kaplan_meier(d, log=lambda *a, **k: None)
    assert abs(float(km.survival_function_at_times(5).iloc[0]) - 0.6) < 1e-9
    assert abs(float(km.survival_function_at_times(9).iloc[0]) - 0.2) < 1e-9


def test_cox_features_limited_by_event_count():
    """Правило десяти событий на признак. Модель Кокса живёт не на числе
    строк, а на числе СОБЫТИЙ: девятьсот висящих объявлений и восемь
    ушедших дают восемь событий, и четыре признака на них — это переобучение
    с доверительными интервалами шире самого эффекта."""
    import numpy as np
    from kz.ml.survival import MIN_EVENTS_PER_FEATURE, cox_model
    assert MIN_EVENTS_PER_FEATURE >= 10

    rng = np.random.default_rng(0)
    n = 60
    d = pd.DataFrame({
        "days": rng.integers(1, 40, n),
        "event": [1] * 15 + [0] * (n - 15),      # ровно 15 событий → 1 признак
        "price_ratio": rng.normal(1.0, 0.2, n),
        "age": rng.integers(1, 25, n),
        "photos_count": rng.integers(1, 15, n),
        "is_vip": rng.integers(0, 2, n),
    })
    cph = cox_model(d, log=lambda *a, **k: None)
    assert len(cph.params_) == 15 // MIN_EVENTS_PER_FEATURE == 1, list(cph.params_)


# ─── квантильный residual-детектор: конфиг осмыслен, фичи без утечки ────────
def test_residual_detector_config():
    from kz.ml import residual_detector as r
    from kz.ml.train_price_model import FEATURES
    assert 0 < r.ALPHA < 0.5              # нижний квантиль (пол цены)
    assert r.MIN_SUPPORT >= 1 and r.AGE_MAX >= 1
    assert r.FEATURES is FEATURES         # те же фичи модели → та же анти-утечка


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


# ─── catch_up: учёт скользящего бюджета запросов за 24 часа ─────────────────
# Самая дорогая логика в проекте: ошибка здесь стоит бана IP, и один раз уже
# стоила. Тесты подменяют файл бюджета на временный — трогать настоящий
# нельзя, в нём живёт реальный расход за последние 24 часа.

def _budget_file(tmp_path, monkeypatch):
    from kz.ops import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    return catch_up, f


def test_budget_accumulates_within_one_day(tmp_path, monkeypatch):
    """Списания складываются, а не перезаписываются: три порции по двадцать
    запросов — это шестьдесят потраченных, а не двадцать."""
    cu, _ = _budget_file(tmp_path, monkeypatch)
    for _ in range(3):
        used = cu.charge_budget("kolesa", 20)
    assert used["kolesa"] == 60
    assert used["cdn"] == 0
    assert cu.load_budget_used()["kolesa"] == 60


def test_parser_and_catch_up_share_one_daily_budget(tmp_path, monkeypatch):
    """Листинг больше не живёт вне антибан-счётчика.

    Реальный риск: parser тратил 100 запросов, catch_up видел в файле ноль и
    разрешал ещё 200. Сумма приближалась к ~270, на которых IP уже банили.
    """
    import pytest
    from kz.collect import parser
    from kz.ops import catch_up

    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(tmp_path / "budget.json"))
    monkeypatch.setitem(catch_up.DAILY_BUDGET, "kolesa", 2)
    monkeypatch.setattr(parser, "_run_kolesa_requests", 0)

    parser.reserve_kolesa_request()
    parser.reserve_kolesa_request()
    assert catch_up.load_budget_used()["kolesa"] == 2
    with pytest.raises(parser.DailyBudgetExhausted):
        parser.reserve_kolesa_request()


def test_parser_defaults_to_fresh_first_pages(monkeypatch):
    """Дефолт — свежак, а не глубинный backfill.

    Блоки до 100-й страницы дали много повторов, но почти не изменили
    MAPE. Поэтому без явного env-override парсер должен всегда брать
    страницы 1–3 каждого ценового сегмента.
    """
    import importlib
    from kz.collect import parser

    monkeypatch.delenv("KOLESA_MAX_PAGES", raising=False)
    monkeypatch.delenv("KOLESA_START_PAGE", raising=False)
    fresh = importlib.reload(parser)
    assert fresh.START_PAGE == 1
    assert fresh.MAX_PAGES_PER_CATEGORY == 3


def test_parser_fails_fast_when_listing_selectors_drift():
    """Нулевая первая страница — поломка контракта, а не успешный конец."""
    import pytest
    from kz.collect import parser

    changed_html = "<html><body><div class='new-card-class'>машина</div></body></html>"
    with pytest.raises(parser.ListingSchemaError, match="изменила HTML"):
        parser.validate_listing_page(changed_html, [], 1, "almaty_3_7m")
    # Пустая глубокая страница допустима: сегмент действительно мог кончиться.
    assert parser.validate_listing_page(changed_html, [], 20, "almaty_3_7m") == 0


def test_parser_fails_when_raw_cards_stop_parsing():
    """Контейнеры ещё видны, но потеря половины полей тоже означает drift."""
    import pytest
    from kz.collect import parser

    html = "<html><body>" + "".join(
        f"<article class='js__a-card' data-id='{i}'></article>" for i in range(10)
    ) + "</body></html>"
    with pytest.raises(parser.ListingSchemaError, match="разобрано 1/10"):
        parser.validate_listing_page(html, [{"ad_id": "1"}], 2, "segment")


def test_parser_reports_open_freshness_boundary():
    """Unseen на последней разрешённой странице = измеримый недобор свежака."""
    from kz.collect.parser import page_limit_has_unseen

    assert page_limit_has_unseen(3, 9, 23, 1, 3)
    assert not page_limit_has_unseen(3, 0, 23, 1, 3)
    assert not page_limit_has_unseen(2, 9, 23, 1, 3)
    assert not page_limit_has_unseen(30, 9, 23, 26, 30)  # deep backfill ≠ fresh


def test_parser_micro_limit_caps_exactly_ten_cards():
    """Тестовый live-прогон не должен случайно обработать всю страницу."""
    from kz.collect.parser import cap_cards_for_run

    cards = [{"ad_id": str(i)} for i in range(23)]
    selected, stop = cap_cards_for_run(cards, already_processed=0, limit=10)
    assert [row["ad_id"] for row in selected] == [str(i) for i in range(10)]
    assert stop
    assert cap_cards_for_run(cards, 0, 0) == (cards, False)  # обычный режим


def test_parser_does_not_retry_a_corrupt_budget(monkeypatch):
    """Fail-closed budget error не должен превращаться в три псевдосетевых retry."""
    import asyncio
    import pytest
    from kz.collect import parser

    class NeverUsedPage:
        async def goto(self, *_args, **_kwargs):
            raise AssertionError("сеть не должна вызываться")

    def broken_reservation():
        raise parser.request_budget.BudgetStateError("broken budget")

    monkeypatch.setattr(parser, "reserve_kolesa_request", broken_reservation)
    with pytest.raises(parser.request_budget.BudgetStateError):
        asyncio.run(parser.get_html(NeverUsedPage(), "https://example.invalid"))


def test_parser_run_status_marks_unhandled_failure(tmp_path, monkeypatch):
    """После аварии status JSON не должен навсегда остаться `running`."""
    import json
    from kz.collect import parser

    status = tmp_path / "parser_status.json"
    monkeypatch.setattr(parser, "RUN_STATUS_FILE", str(status))
    parser.write_run_status({"schema_version": 1, "status": "running",
                             "segments": {}, "totals": {}})
    parser.mark_unhandled_failure(RuntimeError("boom"))
    saved = json.loads(status.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["message"] == "RuntimeError: boom"
    assert saved["finished_at"]


def test_enrich_done_unions_csv_and_postgres(tmp_path, monkeypatch):
    """Строка только в БД не должна повторно сжигать запрос из-за старого CSV."""
    import csv
    import pandas as pd
    from kz.collect import enrich

    path = tmp_path / "enriched.csv"
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=enrich.FIELDS)
        writer.writeheader()
        writer.writerow({"ad_id": "csv-only"})
    monkeypatch.setattr(enrich, "ENRICHED_CSV", str(path))
    monkeypatch.setattr(enrich, "get_engine", lambda: None)
    monkeypatch.setattr(pd, "read_sql", lambda *_args, **_kwargs:
                        pd.DataFrame({"ad_id": ["db-only"]}))
    assert enrich.load_done() == {"csv-only", "db-only"}


def test_budget_reservation_does_not_overshoot(tmp_path, monkeypatch):
    """Проверка и списание — одна операция, а не два гоняющихся чтения."""
    from kz.ops import catch_up

    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(tmp_path / "budget.json"))
    assert catch_up.reserve_budget("kolesa", 2, 3)["kolesa"] == 2
    assert catch_up.reserve_budget("kolesa", 2, 3) is None
    assert catch_up.load_budget_used()["kolesa"] == 2


def test_chunk_refreshes_budget_after_rolling_window_moves(tmp_path, monkeypatch):
    """В памяти мог остаться полный расход, хотя старые события уже отпали."""
    from kz.ops import catch_up

    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(tmp_path / "budget.json"))
    monkeypatch.setitem(catch_up.DAILY_BUDGET, "kolesa", 2)
    gaps = iter([1, 0])
    monkeypatch.setattr(catch_up, "compute_gaps",
                        lambda: {"backfill": next(gaps)})
    monkeypatch.setattr(catch_up, "run", lambda _script: 0)
    monkeypatch.setattr(catch_up, "count_429", lambda: 0)

    used = {"kolesa": 2, "cdn": 0}  # устаревшая копия в долгом процессе
    result = catch_up.run_one_chunk(
        "backfill", "unused", "backfill", "kolesa", used,
        run_spent={"kolesa": 0, "cdn": 0},
    )
    assert result == "done"
    assert used["kolesa"] == 1
    assert catch_up.load_budget_used()["kolesa"] == 1


def test_budget_is_rolling_and_does_not_reset_at_midnight(tmp_path, monkeypatch):
    """Полночь не дарит вторую квоту: событие живёт ровно 24 часа."""
    import json
    from datetime import datetime, timedelta, timezone
    cu, f = _budget_file(tmp_path, monkeypatch)
    now = datetime(2026, 9, 2, 0, 10, tzinfo=timezone(timedelta(hours=5)))
    monkeypatch.setattr(cu, "_now", lambda: now)
    recent = now - timedelta(minutes=20)       # вчера по календарю, но в окне
    expired = now - timedelta(hours=24)        # ровно граница — уже вне окна
    state = {
        "schema_version": cu.BUDGET_SCHEMA_VERSION,
        "days": {"2026-09-01": {"kolesa": 205, "cdn": 900}},
        "events": [
            {"at": recent.isoformat(), "host": "kolesa", "cost": 200},
            {"at": recent.isoformat(), "host": "cdn", "cost": 900},
            {"at": expired.isoformat(), "host": "kolesa", "cost": 5},
        ],
    }
    f.write_text(json.dumps(state), encoding="utf-8")
    assert cu.load_budget_used() == {"kolesa": 200, "cdn": 900}
    # Через 24 часа после реального запроса квота освобождается, не раньше.
    monkeypatch.setattr(cu, "_now", lambda: recent + timedelta(hours=24))
    assert cu.load_budget_used() == {"kolesa": 0, "cdn": 0}


def test_budget_migrates_yesterdays_legacy_sum_conservatively(tmp_path, monkeypatch):
    """Апдейт старого JSON не должен обнулить потенциально свежий расход."""
    import json
    from datetime import datetime, timedelta, timezone
    cu, f = _budget_file(tmp_path, monkeypatch)
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone(timedelta(hours=5)))
    monkeypatch.setattr(cu, "_now", lambda: now)
    f.write_text(json.dumps({
        "days": {"2026-09-01": {"kolesa": 73, "cdn": 600}}
    }), encoding="utf-8")
    assert cu.load_budget_used() == {"kolesa": 73, "cdn": 600}
    migrated = json.loads(f.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == cu.BUDGET_SCHEMA_VERSION
    assert all(event.get("legacy") for event in migrated["events"])


def test_budget_allows_first_run_but_fails_closed_on_corrupt_file(tmp_path, monkeypatch):
    """Нет файла = первый запуск; битый существующий файл обязан закрыть сеть."""
    import pytest
    cu, f = _budget_file(tmp_path, monkeypatch)
    assert cu.load_budget_used() == {"kolesa": 0, "cdn": 0}      # файла нет
    f.write_text("{не json", encoding="utf-8")
    with pytest.raises(cu.BudgetStateError, match="сеть заблокирована"):
        cu.load_budget_used()


def test_budget_reads_the_old_single_day_format(tmp_path, monkeypatch):
    """Формат до 2026-07-30 хранил одну дату на верхнем уровне. Читать его
    надо, иначе переход на новый формат обнулил бы расход и разрешил
    двойную квоту в день обновления."""
    import json
    from datetime import date
    cu, f = _budget_file(tmp_path, monkeypatch)
    f.write_text(json.dumps({"date": date.today().isoformat(),
                             "kolesa": 150, "cdn": 40}), encoding="utf-8")
    assert cu.load_budget_used() == {"kolesa": 150, "cdn": 40}


def test_budget_history_does_not_grow_forever(tmp_path, monkeypatch):
    """Файл бюджета append-only по смыслу, но не по размеру: старые дни
    подрезаются, иначе он растёт вечно."""
    import json
    from datetime import date, timedelta
    cu, f = _budget_file(tmp_path, monkeypatch)
    days = {(date.today() - timedelta(days=i)).isoformat():
            {"kolesa": i, "cdn": 0} for i in range(1, 40)}
    cu._write_days(days)
    kept = json.loads(f.read_text(encoding="utf-8"))["days"]
    assert len(kept) == cu.BUDGET_KEEP_DAYS


def test_run_cap_is_a_second_defence_even_with_rolling_budget(tmp_path, monkeypatch):
    """Потолок запуска остаётся defence-in-depth поверх rolling-счётчика."""
    cu, _ = _budget_file(tmp_path, monkeypatch)
    full = cu.DAILY_BUDGET["kolesa"]
    fresh_day = {"kolesa": 0, "cdn": 0}          # окно уже освободилось
    spent_this_run = {"kolesa": full, "cdn": 0}  # но прогон уже выбрал квоту
    assert not cu.budget_allows("kolesa", "enrich", 1000, fresh_day,
                                spent_this_run)
    # без учёта запуска старая логика разрешила бы продолжать
    assert cu.budget_allows("kolesa", "enrich", 1000, fresh_day, None)


def test_nearly_finished_job_is_not_starved_at_the_quota_edge(tmp_path, monkeypatch):
    """Стоимость порции = min(размер порции, остаток пробела).

    Иначе джоб, которому осталось три запроса, оценивался бы в полные
    двадцать и не пролезал бы в остаток квоты — вечно откладываясь, хотя
    закрылся бы сразу."""
    cu, _ = _budget_file(tmp_path, monkeypatch)
    near_limit = {"kolesa": cu.DAILY_BUDGET["kolesa"] - 5, "cdn": 0}
    assert cu.budget_allows("kolesa", "enrich", 3, near_limit)      # осталось 3
    assert not cu.budget_allows("kolesa", "enrich", 500, near_limit)


def test_429_detector_ignores_the_number_appearing_as_data():
    """«429» встречается в ad_id, ценах и счётчиках. Считать это
    rate-limit-событием — значит останавливать сбор на ровном месте;
    пропустить настоящее — значит долбить сайт, который просит перестать."""
    from kz.ops.catch_up import is_429_line
    for benign in ["наблюдений: 429", "ad_id=224297431", "цена 4290000",
                   "2026-08-24 12:34:29 INFO готово", "скачано 429 фото"]:
        assert not is_429_line(benign), benign
    for real in ["429: пауза 120с", "HTTP 429, пауза", "429 три подряд — стоп"]:
        assert is_429_line(real), real


def test_next_action_puts_rate_limiting_ahead_of_everything():
    """Порядок проверок — это приоритет анти-бана.

    Новый 429 обязан прерывать цепочку раньше, чем сработает разбор
    остальных исходов: сайт прямым текстом просит остановиться, и продолжать
    «потому что прогресс есть» нельзя."""
    from kz.ops.catch_up import next_action
    assert next_action(100, 0, 0, False) == "done"          # пробел закрыт
    assert next_action(100, 0, 1, True) == "done"           # даже с 429
    assert next_action(100, 50, 1, True) == "rate_limited"  # раньше breaker
    assert next_action(100, 50, 1, False) == "breaker"
    assert next_action(100, 100, 0, False) == "stuck"       # прогресса нет
    assert next_action(100, 101, 0, False) == "stuck"       # пробел вырос
    assert next_action(100, 50, 0, False) == "continue"


def test_risk_zones_are_anchored_to_the_ban_that_actually_happened():
    """Зоны риска — не из статей, а из единственного жёсткого факта: домашний
    IP лёг на ~270 запросах за сутки. Дефолтный бюджет обязан оставаться в
    зоне, на которой банов не наблюдали."""
    from kz.ops.catch_up import DAILY_BUDGET, risk_zone
    assert risk_zone(50)[0] == "спокойно"
    assert risk_zone(200)[0] == "безопасно"
    assert risk_zone(260)[0] == "риск"
    assert risk_zone(400)[0] == "высокий риск"
    assert risk_zone(DAILY_BUDGET["kolesa"])[0] in ("спокойно", "безопасно")


def test_eta_accounts_for_pauses_not_just_requests():
    """«Сколько это займёт» обязано учитывать вежливый ритм. Наивная оценка
    по одному запросу обещала бы минуты вместо часов, и человек запускал бы
    сбор, не понимая, на что подписывается."""
    from kz.core.pacing import mean_pause
    from kz.ops.catch_up import eta_minutes
    naive = 200 * 3.0 / 60
    assert eta_minutes(200) > naive * 2
    assert eta_minutes(200) > 200 * mean_pause(4.0, 8.0) / 60


def test_budget_is_configurable_without_touching_code():
    """Потолок задаётся переменной окружения: понижать его при подозрении на
    блокировку нужно быстро, а не через правку исходника и коммит."""
    import importlib
    import os
    from kz.ops import catch_up
    saved = os.environ.get("KOLESA_BUDGET")
    os.environ["KOLESA_BUDGET"] = "37"
    try:
        assert importlib.reload(catch_up).DAILY_BUDGET["kolesa"] == 37
    finally:
        if saved is None:
            os.environ.pop("KOLESA_BUDGET", None)
        else:
            os.environ["KOLESA_BUDGET"] = saved
        importlib.reload(catch_up)


def test_budget_flag_rejects_nonsense():
    """--budget принимает только положительное целое: пустая или мусорная
    квота молча превратилась бы в «сколько угодно»."""
    from kz.ops.catch_up import parse_budget
    assert parse_budget([]) is None
    assert parse_budget(["--budget", "300"]) == 300
    assert parse_budget(["--budget=300"]) == 300
    for bad in (["--budget", "0"], ["--budget", "-5"], ["--budget", "много"]):
        try:
            parse_budget(bad)
        except SystemExit:
            continue
        raise AssertionError(f"принял мусор: {bad}")


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


# ─── catch_up: rolling-бюджет запросов на хост (анти-бан) ────────────────────
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


def test_catch_up_budget_legacy_recovery_and_corruption(tmp_path, monkeypatch):
    """Древний расход вне окна не мешает; recovery setter работает; мусор стоп."""
    import pytest
    from kz.ops import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    f.write_text('{"date":"2000-01-01","kolesa":399,"cdn":5}', encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 0, "cdn": 0}   # старый день → сброс
    catch_up.save_budget_used({"kolesa": 150, "cdn": 300})
    assert catch_up.load_budget_used() == {"kolesa": 150, "cdn": 300}  # сегодня → как есть
    f.write_text("{ битый json", encoding="utf-8")
    with pytest.raises(catch_up.BudgetStateError):
        catch_up.load_budget_used()


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
            + pd.Timedelta(i, unit="D"), "brand": "B", "model": f"M{i}",
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
def _label_cards_source() -> str:
    """Весь код карточек разметки одной строкой.

    Раньше это был один файл, теперь пакет из пяти. Тесты проверяют СВОЙСТВА
    кода («ни одного запроса к kolesa», «есть горячие клавиши»), а не место,
    где строка лежит, — поэтому читаем каталог целиком и не переписываем
    тесты при каждом переносе функции между модулями."""
    from pathlib import Path
    return "\n".join(f.read_text(encoding="utf-8")
                      for f in sorted(Path("kz/report/label_cards").glob("*.py")))



def test_label_cards_never_requests_kolesa():
    """Карточки — офлайн-инструмент: генератор не делает HTTP-запросов
    вообще (фото подставляются как URL и грузятся браузером с CDN)."""
    src = _label_cards_source()
    for bad in ("requests.get", "requests.head", "urlopen", "httpx"):
        assert bad not in src, bad


def test_label_cards_help_covers_real_flags():
    """У каждого флага, который детектор реально ставит, должна быть
    подсказка «как решать» — иначе разметчик остаётся без критерия."""
    from kz.report import label_cards
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
    # verdict и comment стоят сразу после описательных колонок, а колонки
    # слоя дописаны в конец — так старые журналы читаются без миграции.
    i = header.index("verdict")
    assert header[i + 1] == "comment"
    assert header[i - 1] == "seller_comment"
    # шаблон из JS: id + 7 пустых + verdict,comment + пустые колонки слоя
    line = "123" + "," * 8 + "legit,причина" + "," * (len(header) - 10)
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
    src = _label_cards_source()
    for token in ['class="hero"', 'class="thumb', 'id="box"', "openBox",
                  "setVerdict", "focusCard"]:
        assert token in src, token
    # шаблон должен быть СЫРОЙ строкой, иначе \n в JS сломается
    assert 'TEMPLATE = r"""' in src


def test_catch_up_budget_keeps_calendar_history_only_as_audit(tmp_path, monkeypatch):
    """Rolling-сумма точная, а древняя calendar-запись на неё не влияет."""
    from kz.ops import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    assert catch_up.charge_budget("kolesa", 20) == {"kolesa": 20, "cdn": 0}
    assert catch_up.charge_budget("kolesa", 20) == {"kolesa": 40, "cdn": 0}
    assert catch_up.charge_budget("cdn", 300)["cdn"] == 300
    assert catch_up.load_budget_used() == {"kolesa": 40, "cdn": 300}
    # Древняя audit-запись не влияет на rolling-расход.
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


def test_catch_up_per_run_cap_is_defence_in_depth():
    """Один долгий процесс не получает больше полной rolling-квоты."""
    from kz.ops import catch_up
    B = catch_up.DAILY_BUDGET["kolesa"]
    fresh_day = {"kolesa": 0, "cdn": 0}          # старые события вышли из окна
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


def test_collect_dag_covers_the_collect_chain():
    """Сетевую цепочку сторожили только с одной стороны.

    Для ML и офлайна тест требовал, чтобы каждый шаг run_all был в DAG'е, а
    для сбора такого не было: `photo_fetch` добавили в COLLECT_CHAIN, в DAG
    он попал вручную, и ничто не помешало бы забыть. Airflow-прогон тогда
    молча собирал бы не всё, а расхождение обнаружилось бы по недостающим
    данным через неделю."""
    from pathlib import Path
    from kz.ops.run_all import COLLECT_CHAIN
    src = Path("airflow/dags/kolesa_pipeline_dag.py").read_text(encoding="utf-8")
    for _, cmd in COLLECT_CHAIN:
        mod = cmd[cmd.index("-m") + 1]
        assert mod in src, f"{mod} есть в COLLECT_CHAIN, но нет в сетевом DAG"


def test_offline_dag_dependencies_respect_artifacts():
    """Граф DAG'а обязан уважать зависимости по артефактам: графики читают
    модель цены, отчёт — модель И ценовой пол. Иначе таск упадёт в проде на
    отсутствующем файле, хотя в UI выглядел независимым."""
    from pathlib import Path
    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    assert "clean >> monitor >> train >> dashboard" in src
    assert "train >> residual >> report" in src
    assert "clean >> explore >> cards" in src
    # Мониторинг сравнивает с выборкой РАБОТАЮЩЕЙ модели, значит обязан
    # закончиться ДО того, как train перезапишет её текущими данными.
    assert "monitor >> train" in src
    assert "train >> monitor" not in src

    # Финальный таск логирует состояние и обязан ждать ВСЕ листья графа.
    # Проверяем свойство, а не строку: раньше здесь стоял литерал
    # "monitor, survival] >> state", и он ломался от каждой новой ветки —
    # тест падал на добавлении шага, хотя граф был правильным.
    import re as _re
    tasks = set(_re.findall(r"^\s*(\w+)\s*=\s*job\(", src, _re.M))
    edges = set()
    for line in src.splitlines():
        line = line.split("#")[0]
        if ">>" not in line:
            continue
        parts = [p.strip(" []") for p in line.split(">>")]
        # попарный обход: parts[1:] короче на элемент ПО ОПРЕДЕЛЕНИЮ,
        # поэтому здесь усечение zip — то, что нужно, а не небрежность
        for a, b in zip(parts, parts[1:]):
            for x in (n.strip() for n in a.split(",")):
                for y in (n.strip() for n in b.split(",")):
                    if x in tasks and y in tasks:
                        edges.add((x, y))
    leaves = {t for t in tasks
              if t != "state" and not any(a == t and b != "state" for a, b in edges)}
    waited = {a for a, b in edges if b == "state"}
    assert leaves <= waited, (
        "финальный таск не ждёт ветки: " + ", ".join(sorted(leaves - waited)))


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
    # Патчим ПОДМОДУЛЬ, а не пакет: upsert_verdict читает эти имена из
    # своего модуля, и подмена в __init__ на него бы не повлияла.
    from kz.report.label_cards import journal as lc_journal
    monkeypatch.setattr(lc_journal, "LABELS_CSV", str(dst))
    monkeypatch.setattr(lc_journal, "LABELS_PREV", str(tmp_path / "prev.csv"))
    monkeypatch.setattr(lc_journal, "_snapshot_done", False)
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
    from kz.report.label_cards import journal as lc_journal
    monkeypatch.setattr(lc_journal, "LABELS_PREV", str(prev))
    monkeypatch.setattr(lc_journal, "_snapshot_done", False)
    before_text = dst.read_text(encoding="utf-8")
    before = list(csv.DictReader(dst.open(encoding="utf-8")))
    lc.upsert_verdict("111", "legit", "", {})
    after = list(csv.DictReader(dst.open(encoding="utf-8")))
    # срез, а не усечение zip: upsert дописывает строку, если ad_id новый,
    # поэтому after длиннее — а вот прежние строки обязаны быть нетронуты
    for a, b in zip(before, after[:len(before)], strict=True):
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


def test_legacy_label_cards_serve_delegates_to_unified_web(monkeypatch):
    """Старый флаг не должен поднимать второй сервер с другой очередью.

    Он остаётся совместимым алиасом, но ведёт в то же приложение, где
    /label содержит random control, а /damage — отдельную CV-разметку.
    """
    import sys
    from kz.report.label_cards import __main__ as label_cli
    from kz.web import __main__ as web_cli

    called = []
    monkeypatch.setattr(sys, "argv", ["label_cards", "--serve"])
    monkeypatch.setattr(web_cli, "main", lambda: called.append(True))
    label_cli.main()
    assert called == [True]


def test_unified_verdict_endpoint_only_accepts_shown_ads(monkeypatch):
    """Единая HTTP-точка пишет только ad_id из показанной очереди."""
    import asyncio
    from kz.report import label_cards
    from kz.web import app as web

    class Request:
        def __init__(self, payload):
            self.payload = payload

        async def json(self):
            return self.payload

    saved = []
    monkeypatch.setattr(web, "_cards_html", "<p>готово</p>")
    monkeypatch.setattr(web, "_cards_facts",
                        {"111": {"title": "Audi 80", "year": 1994}})
    monkeypatch.setattr(label_cards, "upsert_verdict",
                        lambda *args: saved.append(args))

    good = asyncio.run(web.save_verdict(Request(
        {"ad_id": "111", "verdict": "legit", "comment": "ок"})))
    bad = asyncio.run(web.save_verdict(Request(
        {"ad_id": "999", "verdict": "legit", "comment": ""})))
    assert good.status_code == 200 and len(saved) == 1
    assert bad.status_code == 400 and len(saved) == 1


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
            "data_quality", "train_price_model",
            "residual_detector", "predict_price", "survival",
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


def test_default_run_uses_budgeted_collect_chain():
    """Без флагов нельзя оставлять скрытый status/enrich вне антибан-бюджета."""
    import inspect
    from kz.ops import run_all

    src = inspect.getsource(run_all.main)
    assert "if collect or (not fast and not light):" in src
    assert "for s in COLLECT_CHAIN" in src
    assert "run_parallel(STEP_ENRICH, STEP_PHOTOS)" not in src


# ─── db_stats: логирование дельт для DAG'ов ──────────────────────────────────

def test_db_stats_tables_cover_pipeline_layers():
    """Отчёт должен покрывать все слои: сырьё, обогащение, clean. Иначе после
    прогона не видно, куда именно ушли новые строки."""
    from kz.ops import db_stats
    names = [t for t, _ in db_stats.TABLES]
    for must in ["raw_ads", "sightings", "photos", "enriched", "clean_data"]:
        assert must in names, must


def test_db_stats_snapshot_roundtrip(tmp_path, monkeypatch):
    """Снимок нужен потому, что каждый таск Airflow — отдельный процесс:
    передать счётчики следующему шагу можно только через файл."""
    from kz.ops import db_stats
    f = tmp_path / "snap.json"
    monkeypatch.setattr(db_stats, "SNAPSHOT_FILE", str(f))
    monkeypatch.setattr(db_stats, "table_counts", lambda: {"raw_ads": 10})
    saved = db_stats.save_snapshot(str(f))
    assert saved == {"raw_ads": 10}
    loaded = db_stats.load_snapshot(str(f))
    assert loaded["counts"] == {"raw_ads": 10}
    assert "taken_at_utc" in loaded
    assert db_stats.load_snapshot(str(tmp_path / "missing.json")) is None


def test_db_stats_delta_formatting():
    """Дельта — это главное, что нужно после прогона: сколько строк пришло."""
    from kz.ops import db_stats
    out = db_stats.format_counts({"raw_ads": 4200, "sightings": 5000},
                                 {"raw_ads": 4000, "sightings": 5000})
    assert "+200" in out              # выросло
    assert "no change" in out         # не изменилось — тоже сигнал
    plain = db_stats.format_counts({"raw_ads": 4200})
    assert "+" not in plain           # без baseline колонки дельты нет


def test_dags_are_in_english():
    """DAG'и читают на собеседовании и в портфолио — держим их на английском.
    Проверяем по кириллице в исходнике."""
    import re
    from pathlib import Path
    for dag in Path("airflow/dags").glob("*.py"):
        cyr = re.findall(r"[а-яА-ЯёЁ]+", dag.read_text(encoding="utf-8"))
        assert not cyr, f"{dag.name}: кириллица в DAG — {cyr[:5]}"


# ─── Веб-интерфейс: логика проверяется без поднятия сервера ──────────────────

def test_web_jsonable_handles_numpy_and_nan():
    """pandas отдаёт int64/float64, а json.dumps их не умеет — ответ падал с
    «Object of type int64 is not JSON serializable». NaN тоже недопустим:
    в JSON его нет, браузер получил бы невалидный ответ."""
    import numpy as np
    from kz.web.service import jsonable
    out = jsonable({"a": np.int64(5), "b": np.float64(1.5), "c": float("nan"),
                    "d": [np.int64(1), np.bool_(True)], "e": "текст", "f": None})
    assert out == {"a": 5, "b": 1.5, "c": None, "d": [1, True], "e": "текст",
                   "f": None}
    assert isinstance(out["a"], int) and not isinstance(out["a"], np.integer)
    import json
    json.dumps(out)          # главное: результат обязан сериализоваться


def test_web_listing_warnings_are_evidence_based():
    """Замечания продавцу опираются на замеры по базе, а не на выдумку.
    Формулировки «смотрят чаще» — про корреляцию, обещаний срока продажи
    здесь быть не должно."""
    from kz.web.service import listing_warnings
    w = listing_warnings({"mileage_km": None, "photos_count": 2},
                         asking_price=1_000_000, fair=5_000_000, text="")
    joined = " ".join(w)
    assert "пробег" in joined.lower()
    assert "фотограф" in joined.lower()
    assert "ниже" in joined.lower()              # дёшево без объяснения
    for promise in ("продаст", "за день", "гарант"):
        assert promise not in joined.lower(), f"обещание срока: {promise}"
    # полное объявление по нормальной цене — без замечаний
    clean = listing_warnings({"mileage_km": 90000, "photos_count": 9},
                             asking_price=5_000_000, fair=5_000_000,
                             text="x" * 120)
    assert clean == []


def test_web_price_position_needs_enough_similar():
    """Позиция среди похожих без достаточной выборки — обман: по трём машинам
    нельзя сказать «дешевле большинства»."""
    from kz.web import service
    assert service.MIN_SIMILAR >= 8
    assert service.price_position({"brand": None}, 1_000_000) is None
    assert service.price_position({"brand": "X", "model": "Y", "age": 5}, None) is None


def test_web_app_routes_exist():
    """Маршруты не должны молча пропасть при рефакторинге."""
    from kz.web.app import app
    paths = {r.path for r in app.routes}
    for p in ("/", "/estimate", "/api/estimate", "/label", "/verdict",
              "/damage", "/damage/label", "/api/health"):
        assert p in paths, p


def test_estimate_page_escapes_values_before_inner_html():
    """Марка/модель и ошибки возвращаются в innerHTML — без escape это XSS."""
    from kz.web.pages import estimate_page

    html = estimate_page()
    assert "function esc(" in html
    for value in ("d.error", "x.value", "s.brand", "s.model"):
        assert f"esc({value})" in html


def test_routed_price_model_uses_specialist_only_below_threshold():
    """Маршрут строится по предикту основной модели, не по неизвестному target."""
    import numpy as np
    import pandas as pd
    from kz.ml.train_price_model import RoutedPriceModel

    class Fake:
        def __init__(self, column):
            self.column = column

        def predict(self, rows):
            return np.log(rows[self.column].to_numpy(dtype=float))

    rows = pd.DataFrame({"base": [4_000_000, 6_000_000],
                         "special": [3_500_000, 1_000_000]})
    routed = np.exp(RoutedPriceModel(Fake("base"), Fake("special")).predict(rows))
    assert np.allclose(routed, [3_500_000, 6_000_000])


def test_routed_model_bootstrap_compares_paired_duplicate_groups():
    """Интервал сравнения маршрута с base должен жить в артефакте, не в заметке."""
    import numpy as np
    import pandas as pd
    from kz.ml.train_price_model import grouped_bootstrap_mape_delta

    df = pd.DataFrame({
        "ad_id": ["a", "b", "c", "d"],
        "price_tenge": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        "text_full": ["", "", "", ""],
    })
    truth = np.log(df.price_tenge.to_numpy())
    result = grouped_bootstrap_mape_delta(
        df, truth, truth + np.log(2), n_boot=200
    )
    assert np.isclose(result["mape_delta_pct_points"], -100.0)
    assert result["bootstrap_95_ci"][1] < 0
    assert result["bootstrap_probability_better"] == 1.0


def test_mape_stability_bootstrap_is_grouped_and_deterministic():
    """Перезалив одной машины нельзя считать независимыми наблюдениями."""
    import numpy as np
    from kz.ml.mape_stability import grouped_bootstrap_mape

    ape = np.array([10.0, 10.0, 50.0, 30.0])
    groups = np.array(["same-car", "same-car", "b", "c"])
    first = grouped_bootstrap_mape(ape, groups, n_boot=200, seed=7)
    second = grouped_bootstrap_mape(ape, groups, n_boot=200, seed=7)
    assert first == second
    assert first["independent_groups"] == 3
    assert first["n"] == 4
    assert np.isclose(first["mape_pct"], 25.0)


def test_mape_stability_separates_age_from_price():
    """Возрастный вывод обязан проверяться внутри цены, а не по корреляции."""
    import pandas as pd
    from kz.ml.mape_stability import build_report

    oof = pd.DataFrame({
        "duplicate_group": ["a", "b", "c", "d"],
        "age": [3, 8, 12, 25],
        "actual_price_tenge": [4e6, 6e6, 4e6, 6e6],
        "absolute_percentage_error_pct": [40.0, 10.0, 30.0, 20.0],
    })
    report, rows = build_report(oof, n_boot=50)
    assert len(report["by_age"]) == 4
    assert set(rows[rows.segment_type == "price"].segment) == {"<5M", "5M+"}
    assert set(rows[rows.segment_type == "age_x_price"].segment) == {
        "0-5 | <5M", "6-10 | 5M+", "11-20 | <5M", "21+ | 5M+",
    }


def test_oof_diagnostics_are_local_minimal_and_atomic(tmp_path, monkeypatch):
    """Диагностика не должна уносить текст/URL и обязана переживать запись."""
    import numpy as np
    import pandas as pd
    from kz.ml import train_price_model as model

    target = tmp_path / "oof.csv"
    monkeypatch.setattr(model, "OOF_DIAGNOSTICS_PATH", target)
    df = pd.DataFrame({
        "ad_id": ["a", "b"],
        "text_full": ["достаточно длинный текст", "другой длинный текст"],
        "brand": ["X", "Y"], "model": ["A", "B"], "year": [2020, 2010],
        "age": [7, 17], "price_tenge": [1_000_000, 2_000_000],
    })
    truth = np.log(df["price_tenge"].to_numpy())
    model.save_oof_diagnostics(df, truth, truth, truth)
    saved = pd.read_csv(target)
    assert len(saved) == 2
    assert "ad_id" not in saved
    assert "text_full" not in saved
    assert "actual_price_tenge" in saved
    assert np.allclose(saved["absolute_percentage_error_pct"], 0)


def test_temporal_metrics_include_routed_vs_base_uncertainty():
    """OOT-выигрыш маршрута тоже нельзя публиковать без интервала разницы."""
    import inspect
    from kz.ml import train_price_model

    src = inspect.getsource(train_price_model.evaluate_temporal)
    assert "grouped_bootstrap_mape_delta" in src
    assert '"routed_vs_base": comparison' in src


def test_web_binds_localhost_only():
    """Приложение локальное: ни аутентификации, ни лимита запросов в нём нет,
    наружу открывать нельзя."""
    from pathlib import Path
    from kz.web.__main__ import HOST
    assert HOST == "127.0.0.1"
    src = Path("kz/web/__main__.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in src


# ─── Скачивание фотографий: сырьё для работы с изображениями ─────────────────

def test_photo_fetch_retries_transient_failures_only():
    """Навсегда пропускать можно только то, чего больше не будет: 200 (уже
    скачано), 404/410 (файла нет). Таймаут и обрыв — временные, и такие
    ссылки обязаны остаться в очереди, иначе одна сетевая икота теряла бы
    фотографию навсегда."""
    from kz.collect.photo_fetch import PERMANENT_STATUSES
    assert PERMANENT_STATUSES == {200, 404, 410}
    for transient in (-1, 500, 502, 503, 429):
        assert transient not in PERMANENT_STATUSES


def test_photo_fetch_skips_unresolvable_hosts():
    """Проверка DNS по одному разу на хост, а не тысяча одинаковых падений.
    Реальный случай 2026-08-09: kolesa вывел из эксплуатации один из двух
    CDN-хостов, и 37% ссылок стали недостижимы."""
    from kz.collect.photo_fetch import live_hosts
    urls = ["https://example.invalid/a.jpg", "https://localhost/b.jpg"]
    alive = live_hosts(urls)
    assert "example.invalid" not in alive       # заведомо нерезолвимый TLD


def test_photo_dedup_skips_unresolvable_hosts():
    """pHash-джоб должен делать тот же DNS-preflight, что и скачивание.

    Реальный сбой 2026-08-30: старый CDN чередовался с живым,
    поэтому consecutive-fail предохранитель не срабатывал, а каждая
    мёртвая ссылка ждала timeout + 30 секунд.
    """
    from kz.collect.photo_dedup import live_hosts
    alive = live_hosts(["https://example.invalid/a.jpg", "https://localhost/b.jpg"])
    assert "example.invalid" not in alive


def test_photo_fetch_path_layout_shards_by_ad_id():
    """Тысячи файлов в одном каталоге тормозят файловую систему, поэтому
    раскладываем по подпапкам."""
    from kz.collect.photo_fetch import local_path
    p = local_path("225678236", 1)
    assert p.parts[-2] == "22"                  # шард по первым символам
    assert p.name == "225678236_1.jpg"


def test_photo_fetch_uses_cdn_budget_not_kolesa():
    """Фото лежат на CDN — это другой хост, и он не должен расходовать квоту,
    которую бережём для основного сайта."""
    from pathlib import Path
    src = Path("kz/collect/photo_fetch.py").read_text(encoding="utf-8")
    assert 'charge_budget("cdn"' in src
    assert 'charge_budget("kolesa"' not in src


# ─── Признаки из фотографий ──────────────────────────────────────────────────

def test_photo_embedding_is_reduced_before_modelling():
    """2048 признаков на ~2700 машин — почти столько же признаков, сколько
    наблюдений, и модель выучила бы шум. Поэтому эмбеддинг обязательно
    сжимается, причём заметно."""
    from kz.ml import photo_features
    assert photo_features.N_COMPONENTS <= 64
    import numpy as np
    emb = np.random.RandomState(0).rand(120, 2048)
    out = photo_features.reduce_embeddings(emb)
    assert out.shape[0] == 120
    assert out.shape[1] <= photo_features.N_COMPONENTS
    assert out.shape[1] < emb.shape[1]


def test_photo_quality_metrics_detect_blur():
    """Резкость через дисперсию перепадов яркости: у размытой картинки она
    заметно ниже. Без этого рекомендация «фото смазано» была бы выдумкой."""
    import tempfile
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageFilter
    from kz.ml.photo_features import quality_metrics

    rng = np.random.RandomState(0)
    sharp = Image.fromarray((rng.rand(256, 256, 3) * 255).astype("uint8"))
    blurred = sharp.filter(ImageFilter.GaussianBlur(6))
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.jpg", Path(d) / "b.jpg"
        sharp.save(a, quality=95); blurred.save(b, quality=95)
        qa, qb = quality_metrics(str(a)), quality_metrics(str(b))
    assert qa["img_sharpness"] > qb["img_sharpness"]
    assert qa["img_pixels"] == 256 * 256


def test_photo_ablation_compares_on_same_rows():
    """Сравнивать наборы признаков можно только на ОДНИХ строках: иначе
    разница отражала бы состав выборки, а не пользу от фотографий."""
    from pathlib import Path
    src = Path("kz/ml/photo_ablation.py").read_text(encoding="utf-8")
    assert 'how="inner"' in src          # только машины с фото
    assert "GroupKFold" in src           # и те же разбиения по группам дублей


# ─── Анализ выживаемости: срок жизни объявления ─────────────────────────────

def test_survival_uses_posted_date_not_first_sighting():
    """Начало отсчёта — дата публикации из карточки, а не первая встреча в
    нашем сборе: иначе срок жизни зависел бы от нашего расписания, а не от
    рынка. Объявление, размещённое до старта проекта, получило бы срок 0."""
    from pathlib import Path
    src = Path("kz/ml/survival.py").read_text(encoding="utf-8")
    assert "parse_posted" in src
    assert 'd["start"]' in src


def test_survival_respects_events_per_feature_rule():
    """Модель Кокса на малом числе событий переобучается. Общепринятое
    правило — не меньше десяти событий на признак, и код обязан его
    соблюдать, а не молча брать все признаки."""
    from kz.ml import survival
    assert survival.MIN_EVENTS_PER_FEATURE >= 10


def test_survival_horizon_is_bounded():
    """Выводы за пределами окна наблюдения — экстраполяция. Горизонт должен
    быть явной константой, а не подразумеваться."""
    from kz.ml import survival
    assert 7 <= survival.HORIZON <= 60


# ─── Мониторинг: дрейф данных относительно обучающей выборки ────────────────

def test_psi_zero_for_identical_distributions():
    """Одинаковые выборки не должны давать сдвига."""
    import numpy as np
    from kz.ml.monitoring import psi
    x = np.random.RandomState(0).normal(size=5000)
    assert abs(psi(x, x.copy())) < 1e-6


def test_psi_grows_with_shift():
    """Чем сильнее сдвинулось распределение, тем больше индекс. Без этого
    свойства метрика бесполезна как сигнал тревоги."""
    import numpy as np
    from kz.ml.monitoring import psi
    rng = np.random.RandomState(0)
    base = rng.normal(size=5000)
    small = psi(base, rng.normal(loc=0.2, size=5000))
    big = psi(base, rng.normal(loc=1.5, size=5000))
    assert 0 <= small < big
    assert big > 0.25          # сильный сдвиг обязан попасть в тревожную зону


def test_psi_survives_empty_bins():
    """Пустая корзина не должна ронять расчёт бесконечным логарифмом —
    реальная ситуация, когда новая выборка не покрывает часть диапазона."""
    import numpy as np
    from kz.ml.monitoring import psi
    base = np.arange(1000, dtype=float)
    shifted = np.arange(500, 1000, dtype=float)      # половина диапазона пуста
    v = psi(base, shifted)
    assert np.isfinite(v) and v > 0


def test_psi_thresholds_are_the_standard_ones():
    """Пороги 0.1 и 0.25 — общепринятые, из банковского скоринга. Менять их
    произвольно значит потерять сопоставимость с чужими отчётами."""
    from kz.ml import monitoring
    assert monitoring.PSI_WATCH == 0.10
    assert monitoring.PSI_ALERT == 0.25
    assert monitoring.level(0.05) == "стабильно"
    assert "заметный" in monitoring.level(0.15)
    assert "СИЛЬНЫЙ" in monitoring.level(0.4)


def test_categorical_psi_detects_new_categories():
    """Появление новых марок — типичный дрейф, и он обязан быть заметен."""
    import pandas as pd
    from kz.ml.monitoring import categorical_psi
    old = pd.Series(["Toyota"] * 80 + ["Lada"] * 20)
    same = categorical_psi(old, old.copy())
    new = categorical_psi(old, pd.Series(["BYD"] * 60 + ["Toyota"] * 40))
    assert abs(same) < 1e-6
    assert new > 0.25


def test_label_cards_show_which_stratum_each_ad_is_from():
    """Разметчик должен понимать, ЧТО он проверяет. У помеченного правилами
    вопрос «флаг верен?», у контрольного — «не пропустили ли обман?». Это
    разные задачи, и без пометки слоя контрольные выглядели бы как ошибочно
    попавшие в очередь."""
    src = _label_cards_source()
    assert "random_control" in src and "rule_positive" in src
    assert "residual_candidate" in src
    # у контрольных подсказка обязана говорить, что legit — ожидаемый ответ
    assert "ожидаемый" in src
    # и объяснять, зачем их вообще размечают
    assert "recall" in src or "полнот" in src


def test_residual_detector_respects_exculpation():
    """Оправдание из clean-слоя действует и в модельном детекторе.

    Реальный случай: Camry 2019 за 5.3 млн с разбитым передом. Правила её
    оправдали по слову «аварийная» в тексте, а квантильный пол об этом не
    знал и всё равно тащил в кандидаты — хотя дёшево она стоит именно
    потому, что разбита. Разнобой между двумя детекторами на одних данных
    хуже, чем отсутствие второго."""
    from pathlib import Path
    src = Path("kz/ml/residual_detector.py").read_text(encoding="utf-8")
    assert "low_price_explained" in src
    assert "~explained" in src


def test_web_labelling_includes_control_group():
    """Веб-интерфейс обязан показывать полную очередь, а не только помеченных
    детектором. Иначе размечается лишь то, что он сам нашёл, и полнота
    (recall) остаётся неизмеримой — а именно ради неё в очередь кладут
    случайные обычные объявления."""
    from pathlib import Path
    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    assert "include_queue=True" in src


def test_full_verdict_queue_is_the_default():
    """Новый caller не должен случайно получить только rule_positive.

    Именно такой default создавал два разных количества карточек между
    старым :8765 и единым /label.
    """
    import inspect
    from kz.report.label_cards.queue import load_rows
    assert inspect.signature(load_rows).parameters["include_queue"].default is True


def test_verdict_page_explains_why_queue_counts_differ():
    """Состав полной очереди и единица разметки должны быть видны в UI."""
    import pandas as pd
    from kz.report.label_cards import build

    base = {"brand": "Audi", "model": "80", "year": 1994,
            "price_tenge": 1_000_000, "photos": [], "status": "active",
            "existing_verdict": None, "suspicion_reasons": "",
            "price_z": 0.0}
    rows = pd.DataFrame([
        {**base, "ad_id": "1", "stratum": "rule_positive"},
        {**base, "ad_id": "2", "stratum": "residual_candidate"},
        {**base, "ad_id": "3", "stratum": "random_control"},
    ])
    page = build(rows, serve_mode=True, journal_total=7)
    for text in ("3 объявлений", "1 пометили правила",
                 "1 добавил residual-детектор", "1 взяты случайно",
                 "Осталось принять окончательное решение по 3"):
        assert text in page


def test_label_cards_can_filter_control_group():
    """Контрольные лежат вперемешку с помеченными, а размечать их надо
    отдельно: только по ним считается полнота. Без фильтра их пришлось бы
    выискивать глазами среди сотни карточек."""
    src = _label_cards_source()
    assert 'data-stratum="{st}"' in src
    assert "only-control" in src
    assert 'not([data-stratum="random_control"])' in src


def test_photo_src_prefers_local_and_drops_dead_hosts():
    """Локальная копия важнее ссылки: она грузится мгновенно и не зависит от
    того, жив ли сервер kolesa. Один из двух хостов раздачи отключён, и для
    таких ссылок надо вернуть None, чтобы карточка честно сказала «фото
    недоступны», а не показывала пустые рамки без объяснения."""
    from kz.report.label_cards import DEAD_HOSTS, photo_src
    dead = f"https://{sorted(DEAD_HOSTS)[0]}/webp/aa/x.jpg"
    live = "https://alaps-photos-kl.kcdn.kz/webp/bb/y.jpg"
    assert photo_src("999999999", 1, dead, False) is None
    assert photo_src("999999999", 1, live, False) == live


def test_photo_route_blocks_directory_traversal():
    """Маршрут отдаёт файлы с диска по пути из URL, поэтому обязан проверять,
    что путь не вылезает за каталог фотографий: иначе через ../ читался бы
    любой файл, включая .env."""
    from pathlib import Path
    sources = {"kz/web/app.py": Path("kz/web/app.py").read_text(encoding="utf-8")}
    for where, src in sources.items():
        assert ".resolve()" in src and "parents" in src, where


def test_basket_hint_matches_the_mode():
    """В серверном режиме вердикты уже в журнале, и советовать копипасту
    значит путать: пользователь решит, что разметка не сохранилась. Текст
    подсказки обязан зависеть от режима."""
    src = _label_cards_source()
    assert "baskethint" in src
    assert "SERVER\n  ?" in src or "SERVER ?" in src


def test_counter_reflects_journal_not_just_draft():
    """Счётчик «размечено» должен показывать итог с журналом. У localStorage
    своя память на каждый адрес, поэтому при открытии на другом порту
    черновик пуст — и счётчик, считавший только его, выглядел так, будто
    разметка пропала."""
    src = _label_cards_source()
    assert "ALREADY" in src
    assert "ALREADY.size" in src


def test_web_coerces_numeric_fields_from_forms():
    """Из HTML-формы всё приходит строками. Сравнение «8» < 5 роняло оценку
    с невнятным «'<' not supported between instances of 'str' and 'int'» —
    пользователь видел ошибку про типы вместо цены."""
    from kz.web.service import listing_warnings
    # строки в числовых полях не должны ронять проверку
    assert listing_warnings({"mileage_km": "95000", "photos_count": "8"},
                            11_000_000, 12_000_000, "x" * 120) == []
    # и мусор тоже: функция публичная, вызвать могут не только из формы
    w = listing_warnings({"mileage_km": "абв", "photos_count": None},
                         11_000_000, 12_000_000, "")
    assert any("пробег" in x.lower() for x in w)


def test_web_converts_every_numeric_feature_not_a_hand_list():
    """Список числовых признаков берётся из модели, а не переписывается в
    обработчике: иначе новый признак однажды приедет строкой, и ошибка
    вылезет у пользователя, а не в тестах."""
    from pathlib import Path
    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    assert "NUM_FEATURES" in src
    assert 'for k in list(NUM_FEATURES)' in src


def test_journal_stores_sampling_stratum():
    """Слой выборки обязан лежать В ЖУРНАЛЕ. Очередь — список работы, она
    пересобирается и выкидывает размеченное; после разметки контрольных
    выяснить, что они были контрольными, стало невозможно, а без этого не
    оценить пропуски. Колонки добавляются к существующему заголовку, чтобы
    старые журналы продолжали читаться."""
    from kz.report.label_cards import BASE_HEADER, STRATUM_COLS, journal_header
    h = journal_header()
    for c in STRATUM_COLS:
        assert c in h, c
    for c in BASE_HEADER:
        assert c in h, c


def test_zero_fraud_is_reported_as_a_bound_not_a_blank():
    """Ноль фрода в случайной выборке — это результат, а не отсутствие
    результата: правило трёх даёт верхнюю границу доли. Без пояснения nan
    читается как «ничего не посчиталось»."""
    from pathlib import Path
    src = Path("kz/report/evaluate_detector.py").read_text(encoding="utf-8")
    assert "control_bound_report" in src
    assert "3 / n_ctrl" in src or "3/n_ctrl" in src


# ─── Контейнер и публичный режим веб-сервиса ────────────────────────────────

def test_missing_db_settings_do_not_break_import():
    """Отсутствие настроек Postgres — законное состояние, а не авария.

    Веб-сервис в облаке работает без базы: модель целиком лежит в артефакте.
    Раньше config.py читал os.environ[...] напрямую и ронял ЛЮБОЙ импорт ещё
    до старта — контейнер умирал, не успев ответить ни на один запрос.
    Проверяем поведение при пустом окружении, а не текст файла: файл можно
    переписать как угодно, лишь бы импорт переживал отсутствие настроек."""
    import importlib
    import os
    import kz.core.config as config

    saved = {k: os.environ.pop(k, None)
             for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")}
    try:
        reloaded = importlib.reload(config)          # не должно бросить
        # На машине разработчика load_dotenv вернёт значения из .env, и
        # проверять будет нечего. В CI и в контейнере .env нет — там условие
        # выполняется и тест работает по-настоящему.
        if reloaded.POSTGRES_USER is None:
            assert reloaded.DATABASE_URL is None
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(config)


def test_engine_without_settings_explains_itself():
    """Без настроек get_engine обязан объяснить, что делать, а не отдать
    create_engine(None) с сообщением про NoneType."""
    import kz.core.db as db
    real = db.DATABASE_URL
    db.get_engine.cache_clear()
    db.DATABASE_URL = None
    try:
        db.get_engine()
    except RuntimeError as e:
        assert "POSTGRES" in str(e)
    else:
        raise AssertionError("get_engine без настроек обязан упасть внятно")
    finally:
        db.DATABASE_URL = real
        db.get_engine.cache_clear()


def test_web_query_survives_dead_database():
    """Похожие машины — украшение, оценка — суть. Падение базы не должно
    превращаться в пятисотку на запрос оценки."""
    import kz.web.service as service
    import kz.core.db as db
    real = db.DATABASE_URL
    db.get_engine.cache_clear()
    db.DATABASE_URL = None
    service._db_warned = False
    try:
        assert service.query("SELECT 1", {}) is None
        assert service.similar_cars(
            {"brand": "X", "model": "Y", "age": 5}).empty
        assert service.price_position(
            {"brand": "X", "model": "Y", "age": 5}, 5_000_000) is None
    finally:
        db.DATABASE_URL = real
        db.get_engine.cache_clear()
        service._db_warned = False


def test_public_demo_closes_the_labelling_journal():
    """Разметка пишет в data/manual_labels.csv — единственный ручной ground
    truth во всём проекте, на котором меряется антифрод. Открыть её анониму
    значит дать кому угодно испортить эталон, и восстановить его будет
    неоткуда."""
    from pathlib import Path
    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    assert "KZ_PUBLIC_DEMO" in src
    # обе точки записи должны быть закрыты, не только страница
    assert src.count("if PUBLIC_DEMO:") >= 2


def test_image_carries_model_but_not_collected_ads():
    """В образ едет производная (веса модели), но не сырьё: объявления
    kolesa.kz не наши, чтобы выкладывать их наружу. По весам дерева
    объявление не восстановить, поэтому модель везём."""
    from pathlib import Path
    docker = Path("Dockerfile").read_text(encoding="utf-8")
    assert "price_model.cbm" in docker
    for forbidden in ("COPY data/raw", "COPY data/clean", "COPY data/ ",
                      "COPY . "):
        assert forbidden not in docker, forbidden
    ignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert "data/*" in ignore and ".env" in ignore


def test_web_image_skips_playwright():
    """Сервису оценки браузер не нужен. Он весит около 300 МБ и качался бы
    при каждой сборке ради кода, который в контейнере не выполняется."""
    from pathlib import Path
    # комментарии отбрасываем: в них playwright как раз и объясняется
    lines = [l.split("#")[0].strip().lower()
             for l in Path("requirements-web.txt").read_text(encoding="utf-8")
                          .splitlines()]
    pkgs = [l for l in lines if l]
    assert not any("playwright" in p for p in pkgs)
    assert any("fastapi" in p for p in pkgs)
    assert any("catboost" in p for p in pkgs)


def test_no_module_advertises_a_command_that_no_longer_works():
    """Подсказки «Запуск: python X.py» пережили переезд в пакет и врали.

    После реорганизации запускать надо `python -m kz.transform.clean`, а
    файлы советовали `python clean.py` — команда падает ModuleNotFoundError.
    Часть таких подсказок ПЕЧАТАЕТСЯ пользователю в момент ошибки: человек
    упирался в отсутствующий артефакт, получал совет и совет тоже не
    работал. Ищем по всему пакету, а не по списку файлов: список устареет
    ровно так же, как устарели подсказки."""
    import re
    from pathlib import Path
    names = {p.name for p in Path("kz").rglob("*.py")
             if "__pycache__" not in str(p)}
    bad = []
    for p in Path("kz").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"python ([a-z_]+)\.py", line):
                if m.group(1) + ".py" in names:
                    bad.append(f"{p}:{i}: {m.group(0)}")
    assert not bad, ("подсказка зовёт файл вместо модуля:\n" + "\n".join(bad))


# ─── Вежливый ритм запросов: инвариант «частота только падает» ──────────────

class _FakeRandom:
    """Подставной генератор: тесты про ритм не должны зависеть от везения.

    Ради этого в pacing.py и заведён параметр rng — он там был с самого
    начала, а тестов, ради которых его добавляли, не было."""

    def __init__(self, randoms=(), uniforms=()):
        self._r, self._u = list(randoms), list(uniforms)
        self.calls = []

    def random(self):
        return self._r.pop(0) if self._r else 0.99

    def uniform(self, a, b):
        self.calls.append((a, b))
        return self._u.pop(0) if self._u else (a + b) / 2


def test_pause_never_shortens_below_the_floor():
    """Главное обещание модуля: это politeness, а не маскировка. Средняя
    пауза обязана быть НЕ МЕНЬШЕ прежней, то есть запросов в час становится
    меньше, а не больше. Если бы «человечность» умела делать паузы короче
    нижней границы, мы бы под видом вежливости увеличили нагрузку на сайт."""
    from kz.core.pacing import human_pause, LONG_TAIL_MULT

    lo, hi = 4.0, 8.0
    # обычный случай: попадает в исходный диапазон
    assert human_pause(lo, hi, _FakeRandom(randoms=[0.99])) == (lo + hi) / 2
    # «отвлёкся»: диапазон ТОЛЬКО вверх, нижняя граница не опускается
    rng = _FakeRandom(randoms=[0.0])
    assert human_pause(lo, hi, rng) >= hi
    assert rng.calls == [(hi, hi * LONG_TAIL_MULT)]


def test_long_break_fires_on_schedule_and_not_at_zero():
    """Перерыв каждые N запросов. Нулевой запрос — не повод для перерыва:
    иначе джоб засыпал бы, ещё ничего не сделав."""
    from kz.core.pacing import long_break, BREAK_RANGE

    rng = _FakeRandom()
    assert long_break(0, every=15, rng=rng) is None
    assert long_break(14, every=15, rng=rng) is None
    got = long_break(15, every=15, rng=rng)
    assert got is not None and BREAK_RANGE[0] <= got <= BREAK_RANGE[1]
    assert long_break(30, every=15, rng=rng) is not None
    assert long_break(7, every=0, rng=rng) is None       # выключенные перерывы


def test_mean_pause_is_strictly_slower_than_flat_uniform():
    """Оценка времени прогона должна учитывать и хвост, и перерывы. Раньше
    пользователю обещали «4-8 секунд на запрос», а реальный ритм был
    разреженнее — и прогноз «это займёт полчаса» врал в полтора раза."""
    from kz.core.pacing import mean_pause

    lo, hi = 4.0, 8.0
    flat = (lo + hi) / 2
    assert mean_pause(lo, hi) > flat
    # реже перерывы → быстрее в среднем, но всё равно не быстрее плоского
    assert flat < mean_pause(lo, hi, every=120) < mean_pause(lo, hi, every=15)


def test_polite_sleep_prefers_the_break_over_the_short_pause():
    """На шаге, где положен перерыв, спим перерыв, а не обычную паузу —
    иначе перерывы просто не наступали бы."""
    import kz.core.pacing as pacing

    slept = []
    real_sleep = pacing.time.sleep
    pacing.time.sleep = slept.append
    try:
        brk = pacing.polite_sleep(15, (4.0, 8.0), rng=_FakeRandom(),
                                  break_every=15)
        assert brk >= pacing.BREAK_RANGE[0] and slept == [brk]
        slept.clear()
        short = pacing.polite_sleep(3, (4.0, 8.0), rng=_FakeRandom(randoms=[0.99]),
                                    break_every=15)
        assert short == 6.0 and slept == [6.0]
    finally:
        pacing.time.sleep = real_sleep


def test_pinned_versions_match_the_python_ci_actually_runs():
    """Точный пин версий привязан к версии Python, и это не формальность.

    Реальный случай: версии сняты с локального Python 3.13, а CI поднимал
    3.11. Часть колёс для 3.11 просто не существует — numpy 2.5 требует
    минимум 3.12 — и установка падала на первом же шаге. Пин без
    зафиксированной версии интерпретатора не воспроизводим.

    Сверяем три места: workflow, настройки линтера и работающий сейчас
    интерпретатор."""
    import re as _re
    import sys
    from pathlib import Path

    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    m = _re.search(r'python-version:\s*"(\d+)\.(\d+)"', ci)
    assert m, "в workflow не найдена версия Python"
    ci_ver = (int(m.group(1)), int(m.group(2)))

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    t = _re.search(r'target-version\s*=\s*"py(\d)(\d+)"', pyproject)
    assert t, "в pyproject не найдена target-version"
    lint_ver = (int(t.group(1)), int(t.group(2)))

    assert ci_ver == lint_ver, (
        f"CI гоняет {ci_ver}, линтер целится в {lint_ver}")
    assert ci_ver == sys.version_info[:2], (
        f"CI гоняет {ci_ver}, а версии в requirements сняты с "
        f"{sys.version_info[:2]} — пины будут неустановимы")


def test_every_import_is_declared_in_some_requirements_file():
    """Пакет, который код импортирует, обязан быть где-то объявлен.

    Реальный случай: survival.py импортирует lifelines, а в requirements.txt
    его не было. Локально всё работало — пакет стоял в venv с прошлых
    экспериментов. В чистом клоне и в CI (`pip install -r requirements.txt`
    и сразу pytest) прогон падал бы на импорте, и виноватым выглядел бы тест,
    а не забытая строка в списке зависимостей.

    Проверяем ВСЕ файлы зависимостей: тяжёлые CV-пакеты живут отдельно
    (см. requirements-photos.txt), и это законно."""
    import ast
    import re as _re
    import sys
    from pathlib import Path

    declared = set()
    for req in Path(".").glob("requirements*.txt"):
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                declared.add(_re.split(r"[><=\[]", line)[0].strip().lower())

    # имя для импорта не всегда совпадает с именем пакета
    alias = {"sklearn": "scikit-learn", "PIL": "pillow", "dotenv": "python-dotenv",
             "psycopg2": "psycopg2-binary", "open_clip": "open_clip_torch",
             "bs4": "beautifulsoup4", "cv2": "opencv-python", "yaml": "pyyaml"}
    stdlib = set(sys.stdlib_module_names)

    missing = {}
    for p in list(Path("kz").rglob("*.py")) + list(Path("tests").glob("*.py")):
        if "__pycache__" in str(p):
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            else:
                continue
            for m in mods:
                if m in stdlib or m in ("kz", "__future__"):
                    continue
                pkg = alias.get(m, m).lower()
                if pkg not in declared:
                    missing.setdefault(pkg, set()).add(str(p))
    assert not missing, "импортируется, но нигде не объявлено:\n" + "\n".join(
        f"  {pkg} ← {', '.join(sorted(files))}" for pkg, files in sorted(missing.items()))


def test_dag_docstring_lists_every_task_it_defines():
    """Схема в докстринге DAG'а — первое, что читает человек, и она обязана
    совпадать с кодом ниже. Задачи добавлялись (fetch_photos, data_drift,
    time_on_market), а нарисованный граф остался прежним: читатель видел
    конвейер, которого уже нет."""
    import ast
    import re as _re
    from pathlib import Path

    bad = []
    for dag in sorted(Path("airflow/dags").glob("*.py")):
        text = dag.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(text)) or ""
        for task_id in _re.findall(r'job\(\s*"([a-z_]+)"', text):
            if task_id not in doc:
                bad.append(f"{dag.name}: задача {task_id} не описана в докстринге")
    assert not bad, "\n".join(bad)


def test_enrichment_queue_puts_fresh_ads_ahead_of_stale_backlog(monkeypatch):
    """Страницы объявлений смертны, и очередь обязана это учитывать.

    Раньше сортировка была только по is_suspicious, а порядок внутри
    остальных определялся тем, как база вернула строки. Новое объявление
    вставало в хвост очереди из тысяч старых — а приток новых при ежедневном
    сборе больше суточного бюджета обогащения, то есть хвост рос быстрее,
    чем разбирался. Шестнадцать объявлений дождались своей очереди мёртвыми:
    страница отдала 404, комментарий продавца и бейдж состояния потеряны.

    Подозрительные всё равно идут первыми — их единицы, стоят они копейки,
    и обогащение снимает с них ложные подозрения."""
    from kz.collect import enrich

    rows = pd.DataFrame([
        {"ad_id": "old_plain", "is_suspicious": 0, "scraped_at": "2026-07-17",
         "price_tenge": 9_000_000},
        {"ad_id": "new_plain", "is_suspicious": 0, "scraped_at": "2026-08-10",
         "price_tenge": 9_000_000},
        {"ad_id": "old_susp", "is_suspicious": 1, "scraped_at": "2026-07-17",
         "price_tenge": 9_000_000},
        {"ad_id": "mid_plain", "is_suspicious": 0, "scraped_at": "2026-08-01",
         "price_tenge": 9_000_000},
    ])
    monkeypatch.setattr(enrich.pd, "read_sql", lambda *a, **k: rows.copy())
    monkeypatch.setattr(enrich, "get_engine", lambda: None)

    got = enrich.pick_targets(set())
    assert got[0] == "old_susp", "подозрительное вперёд, несмотря на возраст"
    assert got[1] == "new_plain", "дальше новейшее — его страница ещё жива"
    assert got.index("mid_plain") < got.index("old_plain")


def test_enrichment_queue_prefers_the_cheap_segment(monkeypatch):
    """Дешёвый сегмент вперёд, но не любой ценой.

    Вся ошибка модели сидит в машинах до 5 млн (29,16% против 15,5%), и
    признаки со страницы дают там −3,4 п.п. Значит запросы тратим туда.

    Две оговорки, обе проверяются здесь. Подозрительные не должны терять
    первое место: их обогащение снимает ложные подозрения и стоит копейки.
    И часть прогона обязана уходить самым свежим ВНЕ зависимости от цены —
    иначе заполненные поля окажутся ровно у дешёвых, и «есть комплектация»
    станет для модели меткой «дешёвая машина». Это та же ловушка, что с
    длиной текста в FINDINGS §17.
    """
    from kz.collect import enrich

    rows = pd.DataFrame(
        [{"ad_id": "susp", "is_suspicious": 1, "scraped_at": "2026-07-01",
          "price_tenge": 20_000_000}]
        + [{"ad_id": f"fresh_rich_{i}", "is_suspicious": 0,
            "scraped_at": "2026-08-20", "price_tenge": 30_000_000}
           for i in range(10)]
        + [{"ad_id": f"stale_cheap_{i}", "is_suspicious": 0,
            "scraped_at": "2026-07-05", "price_tenge": 1_000_000}
           for i in range(30)]
    )
    monkeypatch.setattr(enrich.pd, "read_sql", lambda *a, **k: rows.copy())
    monkeypatch.setattr(enrich, "get_engine", lambda: None)

    got = enrich.pick_targets(set())
    assert got[0] == "susp", "подозрительное по-прежнему первое"

    cheap = sum(a.startswith("stale_cheap") for a in got)
    rich = sum(a.startswith("fresh_rich") for a in got)
    assert cheap > rich, (
        f"дешёвых должно быть больше: дешёвых {cheap}, дорогих {rich}")
    assert rich >= 2, (
        "часть прогона обязана уходить свежим независимо от цены, иначе "
        "пропущенность признаков совпадёт с сегментом и станет утечкой")


def test_drift_check_runs_before_retraining():
    """Дрейф меряется ДО переобучения, иначе он не меряет ничего.

    Вопрос мониторинга — «разъехались ли данные с теми, на которых училась
    РАБОТАЮЩАЯ сейчас модель». После переобучения работающая модель обучена
    ровно на текущих данных: снимок обучающей выборки совпадает с текущей
    выборкой, и PSI выходит нулевым по построению.

    Так и было. Все четыре записи в monitoring_history.csv показывали
    training_rows == current_rows и PSI 0.000, а отчёт бодро сообщал
    «данные стабильны». Ложное спокойствие хуже отсутствия проверки."""
    from kz.ops.run_all import ML_CHAIN
    order = [cmd[-1] for _, cmd in ML_CHAIN]
    assert order.index("kz.ml.monitoring") < order.index("kz.ml.train_price_model")


def test_drift_refuses_to_report_stability_it_did_not_measure():
    """Когда сравнивать не с чем, мониторинг обязан сказать это прямо, а не
    печатать «данные стабильны» и записывать нулевой замер в историю."""
    import ast
    import inspect
    from kz.ml import monitoring

    body = inspect.getsource(monitoring.main)
    assert "if fresh_rows <= 0:" in body
    # ранний выход обязан быть ДО записи истории, иначе нули копятся в CSV
    assert body.index("if fresh_rows <= 0:") < body.index("append_history(")
    # и это именно return, а не просто печать предупреждения
    guard = next(n for n in ast.walk(ast.parse(body.lstrip()))
                 if isinstance(n, ast.If) and "fresh_rows" in ast.dump(n.test))
    assert any(isinstance(x, ast.Return) for x in guard.body)


# ─── Свежесть данных: проект живёт на ноутбуке, который закрывают ────────────

def _fresh(**kw):
    from datetime import date, timedelta
    from kz.core.freshness import Freshness
    today = date.today()
    base = dict(last_collect=today, collect_days=30, span_days=30,
                last_status_check=today, ads_total=1000,
                ads_status_checked=900, model_created=today)
    base.update({k: (today - timedelta(days=v) if k.endswith("_ago") else v)
                 for k, v in kw.items() if not k.endswith("_ago")})
    for k, v in kw.items():
        if k.endswith("_ago"):
            base[k[:-4]] = today - timedelta(days=v)
    return Freshness(**base)


def test_stale_statuses_are_called_out_not_swallowed():
    """Конвейер писался под ежедневный запуск, а живёт на ноутбуке, который
    закрывают на две недели. Статус, проверенный две недели назад, сегодня
    не факт: ушедшее объявление всё ещё числится активным, срок жизни
    завышается, доля ушедших занижается. Отчёт обязан сказать это рядом с
    числом, а не оставить читателя в уверенности."""
    from kz.core.freshness import stale_warnings
    assert stale_warnings(_fresh()) == []                 # всё свежее — молчим
    warned = stale_warnings(_fresh(last_status_check_ago=15))
    assert any("Статусы проверялись 15" in w for w in warned)


def test_default_active_status_is_flagged_as_a_guess():
    """clean.py ставит active всем, у кого статуса нет. Для листинга это
    оправданно, но 78% объявлений не проверялись НИ РАЗУ, и записывать их
    живыми по умолчанию — догадка, а не наблюдение."""
    from kz.core.freshness import stale_warnings
    warned = stale_warnings(_fresh(ads_status_checked=220, ads_total=1000))
    assert any("22%" in w for w in warned)


def test_collection_gaps_are_reported():
    """Провалы в сборе означают, что события внутри них не датируются."""
    from kz.core.freshness import stale_warnings
    assert any("Сбор шёл" in w
               for w in stale_warnings(_fresh(collect_days=8, span_days=39)))


def test_model_freshness_converts_utc_to_almaty_calendar_day():
    """01:15 в Алматы — сегодня, даже если UTC-метка ещё вчерашняя.

    Реальный прогон 1 сентября записал ``2026-08-31T20:15Z`` и сразу после
    обучения статус ошибочно напечатал «1 дн. назад». Метрика времени была
    корректной, календарная дата — нет.
    """
    from datetime import date
    from kz.core.freshness import local_date_from_utc_iso

    assert local_date_from_utc_iso("2026-08-31T20:15:44+00:00") == date(2026, 9, 1)


def test_estimate_form_covers_the_categories_in_the_data():
    """Форма оценки обязана предлагать те значения, что есть в данных.

    Реальный случай: в списке кузовов не было «кроссовера» — второго по
    частоте типа, 1549 объявлений из 6789. Владелец кроссовера выбирал
    «внедорожник», модель получала другой класс машин, и оценка уезжала.
    Молча: ни ошибки, ни предупреждения, просто неверное число.

    Словарь категорий пишется в метаданные при обучении. Без артефакта
    (в CI его нет) проверять нечего, но локально — там, где кузов и
    добавляют — тест сработает."""
    import json

    # Путь берём из кода, а не литералом: во-первых, он уже объявлен там
    # константой и дублировать его незачем; во-вторых, CI отдельным шагом
    # запрещает тестам обращаться к data/ строкой — каталог в .gitignore, в
    # чистом клоне его нет, и такой тест падал бы только в CI.
    from kz.ml.train_price_model import METADATA_PATH

    if not METADATA_PATH.exists():
        return
    vocab = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
        "categorical_vocabulary", {})
    if not vocab:
        return

    from kz.web.pages import estimate_page
    html = estimate_page()
    missing = {f: [v for v in vals if f">{v}<" not in html]
               for f, vals in vocab.items()}
    missing = {f: v for f, v in missing.items() if v}
    assert not missing, f"в форме нет значений из данных: {missing}"


def test_conformal_offset_widens_until_coverage_is_reached():
    """Смысл конформной поправки: не верить номинальному квантилю, а измерить,
    насколько он врёт, и раздвинуть границы ровно на измеренное.

    На проекте разрыв оказался большим: модели, обученные на 10-й и 90-й
    процентили, накрыли лишь 67% машин вместо обещанных 80%."""
    import numpy as np
    from kz.ml.price_interval import conformal_offset, conformity

    # факт ровно посередине интервала шириной 2 → невязки все −1
    y = np.zeros(100)
    lo, hi = np.full(100, -1.0), np.full(100, 1.0)
    assert conformity(y, lo, hi).max() <= 0          # все внутри
    assert conformal_offset(conformity(y, lo, hi), 0.8) < 0   # можно сузить

    # половина наблюдений вылезает за верхнюю границу на 5
    y2 = np.concatenate([np.zeros(50), np.full(50, 6.0)])
    off = conformal_offset(conformity(y2, lo, hi), 0.8)
    assert off > 0, "границы обязаны раздвинуться, когда факты вылезают"


def test_interval_quantile_levels_are_symmetric():
    """Целевое покрытие 80% — это 10-й и 90-й процентили: по десять
    процентов остаётся с каждой стороны, а не двадцать с одной."""
    from kz.ml.price_interval import quantile_levels
    for target, want_lo, want_hi in [(0.80, 0.10, 0.90), (0.50, 0.25, 0.75)]:
        lo, hi = quantile_levels(target)
        assert abs(lo - want_lo) < 1e-9 and abs(hi - want_hi) < 1e-9
        assert abs((hi - lo) - target) < 1e-9      # ширина = обещанное покрытие


def test_tails_are_calibrated_separately():
    """Один максимум на две стороны раздвигает границы одинаково и ничего не
    говорит о том, как промахи распределены по хвостам. У нас они были
    распределены криво, поэтому каждый хвост калибруется своим квантилем."""
    import numpy as np
    from kz.ml.price_interval import tail_offsets

    n = 1000
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, n)
    # верхняя граница занижена на 2 — сверху вылезает много, снизу ничего
    lo = np.full(n, -10.0)
    hi = np.full(n, -2.0)
    d_lo, d_hi = tail_offsets(y, lo, hi, 0.80)
    assert d_hi > 1.0, "верхнюю границу обязаны заметно поднять"
    assert d_lo < d_hi, "нижнюю трогать почти не надо — снизу никто не вылез"


def test_groups_are_keyed_on_prediction_not_on_truth():
    """Группа обязана определяться по ПРЕДСКАЗАННОЙ цене.

    Фактическая — это то, что мы предсказываем: при выдаче прогноза новой
    машине её ещё нет. Калибровка, обусловленная на факте, была бы
    неприменима ровно там, где нужна."""
    import inspect
    import numpy as np
    from kz.ml.price_interval import apply_offsets, group_of

    assert list(group_of(np.array([1e6, 7e6, 15e6, 50e6]))) == [0, 1, 2, 3]

    src = inspect.getsource(apply_offsets)
    assert "lo_log + hi_log" in src, "группа берётся из сырых границ прогноза"
    assert "price_tenge" not in src, "фактическая цена в выдаче недоступна"


def test_group_offsets_fall_back_when_a_group_is_too_small():
    """Своя поправка на полусотне строк — это шум, выданный за настройку.
    Маленькая группа обязана считаться общей поправкой, и это должно быть
    видно в метаданных, а не подразумеваться."""
    import numpy as np
    from kz.ml.price_interval import MIN_GROUP, group_offsets

    n = 400
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.3, n)
    lo, hi = y - 0.5, y + 0.5
    # все прогнозы в одной группе: остальные останутся пустыми
    pred = np.full(n, 1e6)
    off = group_offsets(y, lo, hi, pred)
    assert off["groups"]["<5M"]["source"] == "своя"
    assert off["groups"]["<5M"]["n"] >= MIN_GROUP
    for empty in ("5-10M", "10-20M", "20M+"):
        assert off["groups"][empty]["source"].startswith("общая")
        assert off["groups"][empty]["offsets"] == off["global"]


def test_coverage_is_reported_in_both_cuts():
    """Отчёт обязан показывать оба разреза и объяснять разницу.

    По предсказанной цене хвосты ровные — это доказывает, что калибровка
    работает. По фактической остаётся перекос, и он НЕустраним: группировка
    по факту обусловливает на том, что мы предсказываем, а машины с низкой
    настоящей ценой — по построению те, которые модель переоценила. Долго
    считал это дефектом; принять артефакт измерения за баг и «чинить» его
    было бы хуже, чем оставить как есть."""
    import inspect
    from kz.ml import price_interval
    src = inspect.getsource(price_interval.by_segment)
    assert "по предсказанной цене" in src and "по фактической цене" in src
    assert "неустраним" in src or "устранить его нельзя" in src


def test_coverage_is_reported_with_width():
    """Покрытие без ширины — бессмысленное число: интервал «от нуля до
    бесконечности» даёт 100% попаданий и ноль пользы."""
    import inspect
    from kz.ml import price_interval
    src = inspect.getsource(price_interval.coverage_report)
    assert "median_width_pct" in src and "coverage" in src


def test_service_range_uses_measured_interval_not_a_fixed_corridor():
    """Раньше сервис отдавал один коридор ±12/15% на все машины. Для свежей
    Camry он честный, для тридцатилетней Delica — фикция: там модель не знает
    цену ни с точностью 12%, ни с точностью 40%. Диапазон обязан приходить из
    артефакта с измеренным покрытием, а жёсткие числа остаются только
    резервом на случай отсутствия артефакта."""
    import inspect
    from kz.web import service
    src = inspect.getsource(service.full_estimate)
    assert "price_range(car, fair)" in src
    assert "FALLBACK_LOW" not in src, "резерв не должен быть основным путём"
    assert "conformal" in inspect.getsource(service.price_range)


def test_interval_step_runs_after_training():
    """Интервал калибруется квантильными моделями на том же наборе
    признаков, что и основная модель, и стоит отдельным шагом от ценового
    пола: пол — антифрод, интервал — продукт."""
    from kz.ops.run_all import ML_CHAIN
    order = [cmd[-1] for _, cmd in ML_CHAIN]
    assert "kz.ml.price_interval" in order
    assert order.index("kz.ml.train_price_model") < order.index("kz.ml.price_interval")
    assert order.index("kz.ml.price_interval") < order.index("kz.report.ml_report")


def test_survival_reports_a_bracket_not_a_single_number():
    """Одно число про долю ушедших было бы обманом: по всем объявлениям она
    занижена (непроверенные записаны живыми), по проверенным завышена
    (check_status идёт в первую очередь к пропавшим из листинга). Печатаем
    обе границы и говорим, в какую сторону смещена каждая."""
    import inspect
    from kz.ml import survival
    src = inspect.getsource(survival.verified_bracket)
    assert "занижена" in src and "завышена" in src
    assert "verified_bracket(d)" in inspect.getsource(survival.main)


def test_label_cards_package_keeps_its_public_names():
    """Файл на 1239 строк разнесён по ответственностям, но точка входа
    остаться должна прежней: `from kz.report import label_cards` работает у
    оркестратора, у веб-приложения и в тестах. Переезд не должен требовать
    правок у всех, кто им пользуется."""
    from kz.report import label_cards as lc
    for name in ("build", "load_rows", "upsert_verdict",
                 "journal_facts", "dedupe_journal", "read_journal",
                 "LABELS_CSV", "BASE_HEADER", "STRATUM_COLS", "FLAG_HELP"):
        assert hasattr(lc, name), name


def test_label_cards_modules_stay_in_their_lanes():
    """Смысл разделения — в том, что каждый файл делает одно.

    render не ходит в базу: карточку можно собрать и проверить без Postgres.
    queue не пишет на диск: выборка ничего не меняет. Если эти границы
    размоются, разделение станет косметикой."""
    from pathlib import Path
    render = Path("kz/report/label_cards/render.py").read_text(encoding="utf-8")
    queue = Path("kz/report/label_cards/queue.py").read_text(encoding="utf-8")
    assert "get_engine" not in render, "render не должен ходить в базу"
    assert "read_sql" not in render
    assert "write_text" not in queue and "upsert_verdict" not in queue, \
        "queue только читает"


# ─── Советы по фотографиям ──────────────────────────────────────────────────

def test_photo_advice_says_nothing_about_dents():
    """Ржавчина и грязь распознаются надёжно (AUC 0,935 и 0,948 против
    бейджа «Аварийная»), а повреждения — нет: доверительный интервал
    [0,480; 0,731] накрывает монетку. Советовать по признаку, неотличимому
    от случайности, значит выдавать шум за наблюдение."""
    import inspect
    from kz.ml import photo_advice
    src = inspect.getsource(photo_advice.advise)
    assert "clip_rusty" in src and "clip_dirty" in src
    assert "clip_damaged" not in src, "про повреждения советовать нечем"


def test_photo_advice_thresholds_come_from_the_corpus():
    """«Тёмная фотография» — понятие относительное: абсолютная яркость
    зависит от того, как снимают машины вообще. Порог обязан быть
    процентилем по своим данным, а не числом из головы, иначе совет нельзя
    ни проверить, ни опровергнуть."""
    import numpy as np
    import pandas as pd
    from kz.ml.photo_advice import thresholds

    df = pd.DataFrame({"img_brightness": np.linspace(0, 100, 200),
                       "clip_dirty": np.linspace(-1, 1, 200)})
    cuts = thresholds(df, ["img_brightness", "clip_dirty"], worse_than=0.20)
    # яркость: плохо быть НИЗКО → нижний процентиль
    assert 15 < cuts["img_brightness"] < 25
    # грязь: плохо быть ВЫСОКО → верхний
    assert 0.5 < cuts["clip_dirty"] < 0.7


def test_photo_advice_does_not_promise_more_views():
    """Проверка на просмотрах не удалась: объявления с «плохими» фото
    собирают их БОЛЬШЕ, и объяснение не найдено. Значит совет вправе
    сообщать факт («темнее, чем у 80%») и не вправе обещать следствие
    («переснимите — будут смотреть чаще»). Обещание требует эксперимента,
    а не наблюдения."""
    import inspect
    from kz.ml import photo_advice

    # Запрет касается ТЕКСТА ДЛЯ ЧЕЛОВЕКА. В validate() те же слова законны:
    # там они описывают наблюдение («собирают БОЛЬШЕ просмотров»), а не
    # обещают следствие.
    shown = inspect.getsource(photo_advice.advise)
    for promise in ("чаще смотреть", "больше просмотров", "быстрее продад",
                    "продадите"):
        assert promise not in shown, promise

    # А неудача проверки обязана быть записана, а не забыта
    assert "НЕ УДАЛАСЬ" in inspect.getsource(photo_advice.validate)


def test_photo_redundancy_check_is_out_of_fold():
    """Нельзя обучить логрегрессию на семи бейджах и оценить её там же."""
    import inspect
    from kz.ml import photo_clip

    src = inspect.getsource(photo_clip._oof_logistic_auc)
    assert "cross_val_predict" in src
    assert "StratifiedKFold" in src


def test_photo_stats_count_independent_ads(tmp_path, monkeypatch):
    """Три кадра одной битой машины — одна независимая точка для CV."""
    from kz.report import photo_labels

    monkeypatch.setattr(photo_labels, "LABELS_CSV", str(tmp_path / "labels.csv"))
    photo_labels.write_journal(photo_labels.HEADER, [
        {"ad_id": "a", "position": "1", "label": "damaged"},
        {"ad_id": "a", "position": "2", "label": "damaged"},
        {"ad_id": "b", "position": "1", "label": "wreck"},
        {"ad_id": "c", "position": "1", "label": "intact"},
    ])
    s = photo_labels.stats()
    assert s["damaged"] == 2
    assert s["damaged_ads"] == 1
    assert s["positive_ads"] == 2
    assert s["ads_total"] == 3


def test_photo_damage_metric_aggregates_frames_to_ads():
    """Два кадра одной машины не должны удваивать её вес в финальной AUC."""
    import pandas as pd
    from kz.ml.photo_damage import per_ad

    frames = pd.DataFrame({
        "ad_id": ["a", "a", "b"],
        "target": [0, 1, 0],
        "table": [0.1, 0.2, 0.3],
        "photo": [0.4, 0.8, 0.2],
        "combined": [0.2, 0.7, 0.1],
    })
    ads = per_ad(frames).set_index("ad_id")
    assert len(ads) == 2
    assert ads.loc["a", "target"] == 1
    assert ads.loc["a", "photo"] == 0.8


def test_photo_damage_reports_paired_auc_difference():
    """Продуктовый gate требует интервал разницы, не два отдельных CI."""
    import numpy as np
    from kz.ml.photo_damage import auc_delta_ci

    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = good[::-1]
    delta, lo, hi = auc_delta_ci(y, good, bad, n_boot=200)
    assert delta == 1.0
    assert lo > 0 and hi > 0


def test_photo_damage_groups_same_image_across_different_ads():
    """Новый ad_id не делает переопубликованную фотографию независимой."""
    from kz.ml.photo_damage import groups_from_hashes

    groups = groups_from_hashes(
        ["ad-a", "ad-a", "ad-b", "ad-c"],
        ["hash-1", "hash-2", "hash-1", "hash-3"],
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_photo_damage_reports_pr_auc_with_interval():
    """ROC-AUC скрывает редкий класс; рядом обязана быть PR-AUC."""
    import numpy as np
    from kz.ml.photo_damage import average_precision_ci

    y = np.array([0, 0, 0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    value, lo, hi = average_precision_ci(y, good, n_boot=200)
    assert value == 1.0
    assert 0.9 < lo <= hi <= 1.0


def test_photo_ablation_fits_pca_inside_train_fold():
    """Даже unsupervised PCA не должна видеть test-фотографии до оценки."""
    import inspect
    from kz.ml import photo_ablation

    src = inspect.getsource(photo_ablation.cv_mape_with_embeddings)
    assert "fit_transform(emb[tr])" in src
    assert "transform(emb[te])" in src
    main = inspect.getsource(photo_ablation.main)
    assert "reduce_embeddings" not in main


# ─── Разметка повреждений по фотографиям ────────────────────────────────────

def test_damage_box_is_stored_relative_not_in_pixels():
    """Координаты рамки — доли от размера картинки, а не пиксели.

    Изображение в браузере масштабируется под окно: на ноутбуке одна ширина,
    на внешнем мониторе другая. Абсолютные координаты, снятые на одном
    экране, указывали бы не туда на другом, и обучение получило бы рамки
    мимо повреждений."""
    import inspect
    from kz.report import photo_labels
    src = inspect.getsource(photo_labels._normalise_boxes)
    assert "0 <= x1 < x2 <= 1" in src, "рамка обязана быть в долях 0..1"


def test_new_photo_labels_keep_provenance_and_old_rows(tmp_path, monkeypatch):
    """Добавление split/source не переписывает и не теряет legacy-разметку."""
    from kz.report import photo_labels as pl

    labels = tmp_path / "labels.csv"
    previous = tmp_path / "labels.prev.csv"
    legacy_header = pl.HEADER[:10]
    monkeypatch.setattr(pl, "LABELS_CSV", str(labels))
    monkeypatch.setattr(pl, "LABELS_PREV", str(previous))
    monkeypatch.setattr(pl, "_snapshot_done", False)
    pl.write_journal(legacy_header, [{
        "ad_id": "old", "position": "1", "path": "old.jpg",
        "label": "intact", "labeled_at": "2026-01-01T00:00:00",
    }])

    pl.save_label("new", 2, "new.jpg", "intact",
                  selection_source="random_audit", dataset_split="audit",
                  annotator="sanzhar")
    header, rows = pl.read_journal()
    assert all(c in header for c in pl.HEADER)
    assert [r["ad_id"] for r in rows] == ["old", "new"]
    assert rows[0]["label"] == "intact"
    assert rows[1]["selection_source"] == "random_audit"
    assert rows[1]["dataset_split"] == "audit"
    assert rows[1]["label_version"] == pl.LABEL_VERSION


def test_photo_audit_split_is_stable_and_not_everything():
    """Audit membership воспроизводится из ad_id и не зависит от CSV order."""
    from kz.report.photo_labels import split_for_ad

    once = [split_for_ad(str(i)) for i in range(200)]
    twice = [split_for_ad(str(i)) for i in range(200)]
    assert once == twice
    assert 20 <= once.count("audit") <= 60


def test_photo_labels_export_to_detector_ready_coco(tmp_path):
    """Нормированная рамка переводится в пиксели, intact остаётся негативом."""
    from PIL import Image
    from kz.ml.photo_dataset import build_coco

    damaged = tmp_path / "damaged.jpg"
    intact = tmp_path / "intact.jpg"
    Image.new("RGB", (200, 100), "white").save(damaged)
    Image.new("RGB", (80, 60), "white").save(intact)
    rows = [
        {"ad_id": "a", "position": "1", "path": str(damaged),
         "label": "damaged", "x1": "0.1", "y1": "0.2",
         "x2": "0.6", "y2": "0.7", "dataset_split": "train"},
        {"ad_id": "b", "position": "1", "path": str(intact),
         "label": "intact", "dataset_split": "train"},
        {"ad_id": "c", "position": "1", "path": str(intact),
         "label": "intact", "dataset_split": "audit"},
    ]
    coco = build_coco(rows, "train")
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["bbox"] == [20.0, 20.0, 100.0, 50.0]


def test_multiple_damage_boxes_round_trip_and_export_to_coco(tmp_path, monkeypatch):
    """Один кадр с двумя ударами — одно решение кадра и две COCO-аннотации."""
    from PIL import Image
    from kz.ml.photo_dataset import build_coco
    from kz.report import photo_labels as pl

    labels = tmp_path / "labels.csv"
    image = tmp_path / "two-dents.jpg"
    Image.new("RGB", (200, 100), "white").save(image)
    monkeypatch.setattr(pl, "LABELS_CSV", str(labels))
    monkeypatch.setattr(pl, "LABELS_PREV", str(tmp_path / "labels.prev.csv"))
    monkeypatch.setattr(pl, "_snapshot_done", False)

    pl.save_label("a", 1, str(image), "damaged", boxes=[
        (0.1, 0.2, 0.3, 0.4),
        (0.5, 0.1, 0.9, 0.6),
    ])
    header, rows = pl.read_journal()
    assert "boxes_json" in header
    assert len(pl.boxes_from_row(rows[0])) == 2
    assert len(pl.labelled_frames()[0]["boxes"]) == 2
    assert pl.stats()["damage_boxes"] == 2

    coco = build_coco(rows, "train")
    assert len(coco["images"]) == 1
    assert [a["bbox"] for a in coco["annotations"]] == [
        [20.0, 20.0, 40.0, 20.0],
        [100.0, 10.0, 80.0, 50.0],
    ]


def test_damage_label_rejects_what_would_poison_training(tmp_path, monkeypatch):
    """Валидация на сервере, а не в браузере: страница может прислать что
    угодно, а журнал разметки восстановить пересчётом нельзя."""
    from kz.report import photo_labels as pl
    monkeypatch.setattr(pl, "LABELS_CSV", str(tmp_path / "l.csv"))
    monkeypatch.setattr(pl, "LABELS_PREV", str(tmp_path / "p.csv"))
    monkeypatch.setattr(pl, "_snapshot_done", False)

    bad = [
        (dict(label="damaged", box=None), "метка о повреждении без рамки"),
        (dict(label="damaged", box=(0.9, 0.1, 0.2, 0.5)), "вывернутая рамка"),
        (dict(label="damaged", box=(0.1, 0.1, 1.4, 0.5)), "рамка вне картинки"),
        (dict(label="сломана", box=None), "метка не из словаря"),
    ]
    for kw, why in bad:
        try:
            pl.save_label("1", 1, "p.jpg", **kw)
        except ValueError:
            continue
        raise AssertionError(f"принял: {why}")

    pl.save_label("1", 1, "p.jpg", "damaged", box=(0.2, 0.3, 0.5, 0.6))
    assert pl.stats()["damaged"] == 1


def test_damage_relabel_updates_the_row(tmp_path, monkeypatch):
    """Передумал — строка ОБНОВЛЯЕТСЯ, а не дублируется. То же правило, что
    в журнале вердиктов: один объект — одна строка."""
    from kz.report import photo_labels as pl
    monkeypatch.setattr(pl, "LABELS_CSV", str(tmp_path / "l.csv"))
    monkeypatch.setattr(pl, "LABELS_PREV", str(tmp_path / "p.csv"))
    monkeypatch.setattr(pl, "_snapshot_done", False)

    pl.save_label("1", 1, "p.jpg", "damaged", box=(0.2, 0.2, 0.4, 0.4))
    pl.save_label("1", 1, "p.jpg", "damaged", box=(0.1, 0.1, 0.9, 0.9))
    _, rows = pl.read_journal()
    assert len(rows) == 1
    assert rows[0]["x2"] == "0.9000"


def test_damage_queue_is_stratified_not_random():
    """При доле повреждённых около процента случайная выборка дала бы
    две-три положительные метки на три сотни — учиться было бы не на чем.
    Помеченные объявления идут вперёд, контроль подмешивается."""
    import inspect
    from kz.report import photo_labels
    src = inspect.getsource(photo_labels.queue)
    assert "suspect" in src and "CONTROL_PER_POSITIVE" in src
    # и перемешивание, иначе разметчик первые сто кадров видит только битые
    assert "sample(frac=1.0" in src


def test_damage_routes_are_closed_in_public_mode():
    """Разметка пишет в data/photo_labels.csv — ручной труд, который не
    восстановить. Наружу такое не открывается, как и /label."""
    from pathlib import Path
    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    damage = src[src.index("def damage_page"):src.index("def label_page")]
    assert damage.count("if PUBLIC_DEMO:") >= 2, "закрыты обе точки: показ и запись"


def test_damage_labelling_never_touches_kolesa():
    """Фотографии отдаются с диска. Ручной браузинг по сайту однажды уже
    помог положить IP — разметка обязана быть офлайновой."""
    import ast
    from pathlib import Path

    def code_only(path: str) -> str:
        """Исходник без комментариев и докстрингов.

        Прямой поиск по тексту здесь врёт: в докстринге как раз объясняется,
        что к kolesa мы НЕ ходим, и проверка спотыкалась об это объяснение.
        ast.unparse отбрасывает комментарии сам, докстринги убираем руками."""
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) \
                    and ast.get_docstring(node):
                node.body = node.body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    for f in ("kz/web/damage_page.py", "kz/report/photo_labels.py"):
        src = code_only(f)
        for bad in ("kolesa.kz", "requests.get", "urlopen", "httpx"):
            assert bad not in src, f"{f}: {bad}"


def test_labels_path_can_be_redirected_away_from_the_real_journal():
    """Проверять живой сервер, который пишет в НАСТОЯЩИЙ журнал разметки, —
    прямой путь к её потере. Один раз так и вышло: тестовые записи легли
    рядом с работой пользователя, и уборка унесла обе.

    Переменная окружения уводит журнал в сторону, чтобы ручная проверка
    интерфейса физически не могла коснуться ручного труда."""
    import importlib
    import os
    import subprocess
    import sys
    from kz.report import photo_labels as pl

    saved = os.environ.get("KZ_LABELS_DIR")
    os.environ["KZ_LABELS_DIR"] = "/tmp/kz_scratch"
    try:
        m = importlib.reload(pl)
        assert m.LABELS_CSV.startswith("/tmp/kz_scratch")
        assert m.LABELS_PREV.startswith("/tmp/kz_scratch")
        # Проверяем второй журнал в отдельном процессе, чтобы reload не
        # оставил scratch-константы в уже импортированных queue/render.
        code = ("from kz.report.label_cards.journal import LABELS_CSV, "
                "LABELS_PREV; print(LABELS_CSV); print(LABELS_PREV)")
        paths = subprocess.check_output(
            [sys.executable, "-c", code], env=os.environ, text=True).splitlines()
        assert paths and all(p.startswith("/tmp/kz_scratch") for p in paths)
    finally:
        if saved is None:
            os.environ.pop("KZ_LABELS_DIR", None)
        else:
            os.environ["KZ_LABELS_DIR"] = saved
        importlib.reload(pl)


def test_damage_flow_asks_before_it_records():
    """Рамка НЕ ставит метку сама.

    Первая версия сохраняла «повреждение», как только отпущена мышь. Обвести
    область можно, чтобы разглядеть её поближе или поправить границы, — и
    каждое такое движение становилось записью в журнал, который нельзя
    восстановить пересчётом. Теперь: обвёл → окно с выбором → подтвердил."""
    from pathlib import Path
    src = Path("kz/web/damage_page.py").read_text(encoding="utf-8")
    # после отпускания мыши открывается диалог, а не идёт сохранение
    assert "openAsk('damaged')" in src
    mouseup = src[src.index("addEventListener('mouseup'"):
                  src.index("async function commit")]
    assert "commit(" not in mouseup, "сохранение не должно идти по отпусканию мыши"
    assert "a-save" in src and "a-cancel" in src


def test_damage_relabel_saves_the_visible_frame_not_queue_index():
    """В режиме просмотра старых меток `i` относится к DONE, не к QUEUE.

    Использовать QUEUE[i] означало тихо изменить другой кадр именно в
    операции, которая должна исправлять ручную разметку.
    """
    from pathlib import Path

    src = Path("kz/web/damage_page.py").read_text(encoding="utf-8")
    commit = src[src.index("async function commit"):
                 src.index("document.getElementById('a-save')")]
    assert "const it = view[i]" in commit
    assert "const it = QUEUE[i]" not in commit


def test_damage_ui_collects_multiple_boxes_before_one_commit():
    """Несколько рамок отправляются одним списком, а не перетирают кадр."""
    from pathlib import Path
    from kz.web import damage_page

    src = Path("kz/web/damage_page.py").read_text(encoding="utf-8")
    assert 'id="a-add"' in src
    assert "boxes.push(box.slice())" in src
    assert "label: label, boxes: finalBoxes" in src
    html = damage_page.page([], {}, [])
    assert f"const MAX_BOXES = {damage_page.MAX_BOXES_PER_FRAME};" in html
    assert "__MAX_BOXES__" not in html


def test_damage_ui_uses_exact_english_dataset_labels():
    """Кнопки должны совпадать с CSV-классами и не маскировать их переводом.

    Русское «повреждение кузова» было шире целевого класса и привело к тому,
    что ржавчина и потёртости попадали в damaged. Определение оставляем рядом.
    """
    from kz.web import damage_page

    page = damage_page.page([], {}, [])
    for label in ("Damaged", "Wreck", "Parts", "Intact", "Unclear"):
        assert f">{label}<" in page or f">{label}<kbd>" in page
    assert "Intact = no impact/dent" in page
    assert "Ржавчина, грязь, потёртости тоже" in page
    for old in (">повреждение кузова<", ">серьёзная авария<",
                ">разобрана / снят агрегат<", ">целая<", ">не понять<"):
        assert old not in page


def test_damage_endpoint_ignores_client_supplied_photo_path():
    """Путь берётся с сервера, а сохранённый кадр удаляется из кэша."""
    import importlib
    import inspect

    web = importlib.import_module("kz.web.app")
    src = inspect.getsource(web.damage_label)
    assert 'str(provenance["path"])' in src
    assert 'str(data["path"])' not in src
    assert "_damage_queue = [r for r in _damage_queue" in src


def test_verdict_counter_shows_the_journal_not_just_the_queue():
    """Очередь намеренно состоит из НЕразмеченного, поэтому «размечено среди
    показанных» всегда около нуля. Выходило «20 из 308» при 85 вердиктах в
    журнале, и читалось как «работа пропала»."""
    import inspect
    from kz.report.label_cards import render
    src = inspect.getsource(render.build)
    assert "journal_total" in src
    assert "read_journal" in src, "число берётся из журнала, а не из карточек"


def test_disassembled_car_is_its_own_class_not_damage():
    """Двигатель на брусчатке — не вмятина и не целая машина.

    Реальный кадр из очереди: Hyundai Sonata 2015 за 4,2 млн, комментарий
    «Запчасқа болады», бейдж «Аварийная/Не на ходу», двигатель снят и лежит
    рядом. Свалить это в «повреждение» значит сделать положительный класс
    разнородным: помятое крыло и снятый агрегат выглядят совершенно
    по-разному, и на двух сотнях меток сеть не выучит ни того, ни другого.

    Объединить метки потом можно бесплатно, разделить — невозможно."""
    from kz.report.photo_labels import LABELS
    assert "parts" in LABELS and "damaged" in LABELS
    assert LABELS["parts"] != LABELS["damaged"]


def test_box_is_required_for_damage_and_allowed_everywhere_else(tmp_path,
                                                                monkeypatch):
    """Рамка обязательна для «повреждения» и разрешена при любой метке.

    Обязательна — потому что смысл метки в участке: без рамки в обучение
    попал бы целый кадр, где повреждение тонет в асфальте и небе.

    Разрешена везде — потому что запрет молча съедал ручной труд. Разметчик
    обводил ржавчину, ставил «целая», и координаты отбрасывались; человек
    считал, что отмечает область, а не сохранялось ничего. Терять сделанное
    руками нельзя, даже когда не знаешь, что с ним делать сегодня.
    """
    from kz.report import photo_labels as pl
    monkeypatch.setattr(pl, "LABELS_CSV", str(tmp_path / "l.csv"))
    monkeypatch.setattr(pl, "LABELS_PREV", str(tmp_path / "p.csv"))
    monkeypatch.setattr(pl, "_snapshot_done", False)

    try:
        pl.save_label("1", 1, "p.jpg", "damaged")
    except ValueError:
        pass
    else:
        raise AssertionError("принял «damaged» без рамки")

    for n, label in enumerate(("parts", "intact", "unclear", "wreck"), start=2):
        pl.save_label(str(n), 1, "p.jpg", label, box=(0.1, 0.1, 0.5, 0.5))
    _, rows = pl.read_journal()
    assert all(r["x1"] for r in rows), "рамка должна сохраняться при любой метке"

    # вывернутая или выходящая за кадр рамка не принимается ни при какой метке
    for bad in ((0.5, 0.1, 0.2, 0.5), (-0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 1.5, 0.5)):
        try:
            pl.save_label("9", 1, "p.jpg", "intact", box=bad)
        except ValueError:
            continue
        raise AssertionError(f"принял негодную рамку {bad}")
