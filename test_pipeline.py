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

import parser as listing_parser
import clean
import enrich
import photo_dedup
import evaluate_detector


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
    import damage
    assert clean.DAMAGE_PATTERNS is damage.DAMAGE_PATTERNS
    assert enrich.DAMAGE_PATTERNS is damage.DAMAGE_PATTERNS


# ─── отрицания: «нет гнили» — это НЕ повреждение (реальные кейсы из базы) ────
def test_negated_damage_not_detected():
    """Три реальных объявления, ложно получавших damage_keywords:
    225209601 («На 99% нету никаких гнилей» → 'гнил'),
    225770229 («Вложения не требует» → 'вложения'),
    225838706 («не требует вложений» → 'требует вложений')."""
    from damage import find_damage_keywords
    assert find_damage_keywords(
        "Машина в идеальном состоянии. На 99% нету никаких гнилей.") == []
    assert find_damage_keywords("Вложения не требует. Обмен не интересует!") == []
    assert find_damage_keywords(
        "В хорошем состоянии, не требует вложений, ТО пройдено.") == []


def test_real_damage_still_detected():
    """Позитивный контроль: настоящая убитость не должна потеряться
    из-за окна отрицаний."""
    from damage import find_damage_keywords
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
    from damage import find_damage_keywords, has_damage
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
def test_catch_up_references_real_scripts():
    """Защита от опечатки в имени джоба — оркестратор упал бы в рантайме."""
    import os, catch_up
    scripts = ([s for _, s, _ in catch_up.KOLESA]
               + [s for _, s, _ in catch_up.CDN]
               + [s for _, s in catch_up.OFFLINE])
    for s in scripts:
        assert os.path.exists(s), f"catch_up ссылается на несуществующий {s}"


# ─── catch_up: детект 429 не должен ложно срабатывать на числах ─────────────
def test_catch_up_429_detection_not_fooled_by_numbers():
    """Реальный баг моего же кода: count_429 считал подстроку '429', а она
    есть в ad_id/ценах/таймстемпах («наблюдений: 429») → catch_up ложно
    обрывал бы джобы. Считаем только настоящие rate-limit-строки."""
    import catch_up
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
    from catch_up import next_action
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
    import catch_up, check_status, enrich, backfill_avgprice, photo_dedup
    assert catch_up.CHUNK_MAX["status"]   == check_status.MAX_CHECKS_PER_RUN
    assert catch_up.CHUNK_MAX["enrich"]   == enrich.MAX_PER_RUN
    assert catch_up.CHUNK_MAX["backfill"] == backfill_avgprice.MAX_PER_RUN
    assert catch_up.CHUNK_MAX["photo"]    == photo_dedup.MAX_PER_RUN


def test_catch_up_budget_allows_near_done_at_edge():
    """Оценка стоимости порции = min(MAX_PER_RUN, пробел): полная порция у
    края квоты НЕ влезает, но почти добитый джоб (маленький пробел) — влезает
    в тот же остаток. Иначе near-done джоб голодал бы у границы бюджета."""
    import catch_up
    B = catch_up.DAILY_BUDGET["kolesa"]
    assert catch_up.budget_allows("kolesa", "status", 3809, {"kolesa": 0, "cdn": 0})
    assert not catch_up.budget_allows("kolesa", "enrich", 3000, {"kolesa": B - 50, "cdn": 0})
    assert catch_up.budget_allows("kolesa", "enrich", 30, {"kolesa": B - 50, "cdn": 0})


def test_catch_up_status_thresholds_match_check_status():
    """Пороги staleness/recheck в catch_up.compute_gaps должны совпадать с
    check_status — иначе счётчик пробелов разошёлся бы с реальной выборкой
    джоба (показывал бы «есть что добрать», а джоб ничего бы не брал)."""
    import catch_up, check_status
    assert catch_up.STATUS_STALE_DAYS   == check_status.STALE_DAYS
    assert catch_up.STATUS_RECHECK_DAYS == check_status.RECHECK_DAYS


def test_status_recheck_and_listing_inference():
    """Логика статус-джоба (чистые предикаты, без сети):
      needs_status_check — терминал не трогаем; свежий в листинге не требует
        запроса; недавно проверенный остывает; пропал+давно не проверяли → да.
      infer_active_from_listing — показавшийся в листинге active без запроса;
        уже-active не переписываем; терминал реактивируем ТОЛЬКО если увиден
        ПОСЛЕ пометки терминальным."""
    from check_status import needs_status_check, infer_active_from_listing
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
    import catch_up
    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    f.write_text('{"date":"2000-01-01","kolesa":399,"cdn":5}', encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 0, "cdn": 0}   # старый день → сброс
    catch_up.save_budget_used({"kolesa": 150, "cdn": 300})
    assert catch_up.load_budget_used() == {"kolesa": 150, "cdn": 300}  # сегодня → как есть
    f.write_text("{ битый json", encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 0, "cdn": 0}   # не падаем
