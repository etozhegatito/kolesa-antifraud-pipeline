# -*- coding: utf-8 -*-
"""Implementation for the `tests.test_pipeline` module."""

import pandas as pd

from kz.collect import parser as listing_parser
from kz.transform import clean
from kz.collect import enrich
from kz.collect import photo_dedup
from kz.report import evaluate_detector


def test_mileage_with_probegom():
    r = listing_parser.parse_spec_line(
        "2014 г., Б/у седан, 3 л, бензин, КПП автомат, с пробегом 170 000 км, срочно"
    )
    assert r["mileage_km"] == 170000


def test_mileage_bare_km():
    """Regression coverage for `test_mileage_bare_km`."""
    r = listing_parser.parse_spec_line(
        "1994 г., Б/у седан, 2 л, бензин, КПП механика, 260 000 км, серебристый"
    )
    assert r["mileage_km"] == 260000


def test_vip_line_without_mileage():
    r = listing_parser.parse_spec_line("2026 г., 1.5 л, гибрид, КПП автомат")
    assert r["mileage_km"] is None
    assert r["year"] == 2026
    assert r["transmission"] == "автомат"


def test_gas_petrol_not_mislabeled():
    """Regression coverage for `test_gas_petrol_not_mislabeled`."""
    r = listing_parser.parse_spec_line("2010 г., Б/у седан, 2 л, газ-бензин, КПП автомат")
    assert r["engine_type"] == "газ-бензин"


def test_crossover_body_detected():
    """Regression coverage for `test_crossover_body_detected`."""
    r = listing_parser.parse_spec_line("2020 г., Б/у кроссовер, 2 л, бензин, КПП автомат")
    assert r["body_type"] == "кроссовер"


def test_full_size_url():
    assert listing_parser.to_full_size("https://x.kcdn.kz/webp/aa/bb/13-255x138.jpg").endswith(
        "13-full.jpg"
    )
    assert listing_parser.to_full_size("https://x.kcdn.kz/webp/aa/bb/9-160x120.webp").endswith(
        "9-full.webp"
    )


def test_static_ui_images_not_collected_as_photos():
    """Regression coverage for `test_static_ui_images_not_collected_as_photos`."""
    from bs4 import BeautifulSoup

    html = """<div class="js__a-card" data-id="1">
        <img src="https://m.kolesa.kz/static/mobile/images/app/report/advert/badge.png"/>
        <img src="//kolesa.kz/static/frontend/images/stubs/noPhoto_160x120.svg"/>
        <img src="https://alakt-photos-kl.kcdn.kz/webp/aa/bb/1-255x138.jpg"/>
    </div>"""
    card = BeautifulSoup(html, "html.parser").select_one(".js__a-card")
    urls = listing_parser.extract_photo_urls(card)
    assert urls == ["https://alakt-photos-kl.kcdn.kz/webp/aa/bb/1-full.jpg"]


def test_split_brand_model():
    assert listing_parser.split_brand_model("Kia K7") == ("Kia", "K7")
    assert listing_parser.split_brand_model("Mercedes-Benz GLS 450") == ("Mercedes-Benz", "GLS 450")


def test_normal_page_with_recaptcha_footer_not_blocked():
    """Regression coverage for `test_normal_page_with_recaptcha_footer_not_blocked`."""
    html = '<div class="js__a-card">...</div><p>Защищено reCAPTCHA</p>'
    assert listing_parser.looks_blocked(html) is False


def test_login_page_is_blocked():
    assert listing_parser.looks_blocked("<title>Вход в личный кабинет</title>") is True


def test_seller_comment_unicode_escape():
    html = (
        '{"descriptionText":"\\u041f\\u0440\\u043e\\u0434\\u0430\\u043c '
        '\\u0430\\u0432\\u0442\\u043e<br />\\u0442\\u043e\\u0440\\u0433"}'
    )
    text = enrich.extract_seller_comment(html)
    assert "Продам авто" in text
    assert "<br" not in text


def test_damage_past_tense_running():
    """Regression coverage for `test_damage_past_tense_running`."""
    searchable = "продам пассат был находу двиготель коробка есть"
    hits = [p for p in enrich.DAMAGE_PATTERNS if p in searchable]
    assert "был находу" in hits


def test_damage_patterns_in_sync():
    """Regression coverage for `test_damage_patterns_in_sync`."""
    from kz.transform import damage

    assert clean.DAMAGE_PATTERNS is damage.DAMAGE_PATTERNS
    assert enrich.DAMAGE_PATTERNS is damage.DAMAGE_PATTERNS


def test_negated_damage_not_detected():
    """Regression coverage for `test_negated_damage_not_detected`."""
    from kz.transform.damage import find_damage_keywords

    assert find_damage_keywords("Машина в идеальном состоянии. На 99% нету никаких гнилей.") == []
    assert find_damage_keywords("Вложения не требует. Обмен не интересует!") == []
    assert find_damage_keywords("В хорошем состоянии, не требует вложений, ТО пройдено.") == []


def test_real_damage_still_detected():
    """Regression coverage for `test_real_damage_still_detected`."""
    from kz.transform.damage import find_damage_keywords

    assert "гнил" in find_damage_keywords("кузов гнилой, пороги под замену")
    assert "требует вложений" in find_damage_keywords("машина требует вложений")
    assert "не на ходу" in find_damage_keywords("стоит в гараже, не на ходу")
    assert "после дтп" in find_damage_keywords("продаю после дтп, на запчасти")

    assert "ржавчин" in find_damage_keywords("салон не прокурен, есть ржавчина по аркам")

    kws = find_damage_keywords("Машина без матора, без коробки, остальное на месте")
    assert "без матора" in kws and "без коробки" in kws


def test_price_basis_links_the_listing_price_to_the_nearest_cue():
    """Multiple displayed prices must not be collapsed into one vague keyword hit."""
    from kz.transform.price_basis import classify_price_basis

    text = "Цена: без растаможки — 7 000 000; с растаможкой — 10 900 000; в кредит — 11 400 000."
    assert classify_price_basis(text, "Нет", 7_000_000) == "cash_uncleared"
    assert classify_price_basis(text, "Нет", 10_900_000) == "cash_customs_cleared"
    assert classify_price_basis(text, "Нет", 11_400_000) == "credit_price"


def test_price_basis_uses_contextual_negation_not_any_ne():
    """Unrelated negation and dealer boilerplate must not reject a cash price."""
    from kz.transform.price_basis import classify_price_basis

    assert (
        classify_price_basis("Не требует вложений. Кредит не интересует.", "Да", 7_000_000)
        == "cash_customs_cleared"
    )
    assert classify_price_basis("Автомобиль не растаможен", None, 7_000_000) == "cash_uncleared"
    assert classify_price_basis("Автомобиль не  растоможен", None, 7_000_000) == ("cash_uncleared")
    assert classify_price_basis("Растаможка не оплачена", None, 7_000_000) == ("cash_uncleared")
    assert (
        classify_price_basis("Кредит до 7 лет, первый взнос от 10%.", "Да", 7_000_000)
        == "cash_customs_cleared"
    )


def test_price_basis_recognises_a_real_down_payment_amount():
    """A down payment is rejected only when it matches the saved listing price."""
    from kz.transform.price_basis import classify_price_basis

    text = "Полная цена 8 500 000, первоначальный взнос 1 500 000 тенге."
    assert classify_price_basis(text, "Да", 1_500_000) == "down_payment"
    assert classify_price_basis(text, "Да", 8_500_000) == "cash_customs_cleared"


def test_price_basis_ignores_finance_terms_in_a_later_clause():
    """Dealer finance boilerplate after a cash price must not relabel that price."""
    from kz.transform.price_basis import classify_price_basis

    down_payment_offer = "По супер цене от 7 590 000 т. Первоначальный взнос от 10%."
    credit_offer = "Цена от 9 990 000 тг · в кредит с первоначальным взносом от 474 500 тг."
    assert classify_price_basis(down_payment_offer, "Да", 7_590_000) == ("cash_customs_cleared")
    assert classify_price_basis(credit_offer, "Да", 9_990_000) == ("cash_customs_cleared")


def test_price_basis_marks_conflicting_customs_evidence_ambiguous():
    """A structured customs value must not silently override contradictory prose."""
    from kz.transform.price_basis import classify_price_basis

    text = "Цена с учетом доставки и растаможки."
    assert classify_price_basis(text, "Нет", 12_000_000) == "ambiguous"
    typo = "Расстаможка, НДС оплачен. Остались утильсбор и регистрация."
    assert classify_price_basis(typo, "Нет", 9_900_000) == "ambiguous"


def test_price_training_rejects_only_known_non_comparable_targets():
    """Unknown price bases stay usable while explicit traps leave training."""
    import pandas as pd

    from kz.ml.train_price_model import prepare_training_data

    frame = pd.DataFrame(
        {
            "price_tenge": [7_000_000, 11_400_000, 1_500_000, 9_000_000, 8_000_000],
            "mileage_km": [10_000] * 5,
            "is_suspicious": [0] * 5,
            "price_basis": [
                "cash_uncleared",
                "credit_price",
                "down_payment",
                "ambiguous",
                "cash_customs_cleared",
            ],
        }
    )

    result = prepare_training_data(frame)
    assert result["price_basis"].tolist() == [
        "ambiguous",
        "cash_customs_cleared",
    ]


def test_price_basis_policy_is_shared_by_downstream_consumers():
    """Reports, floor calibration, and examples must use the training cohort."""
    from pathlib import Path

    for path in [
        "kz/ml/predict_price.py",
        "kz/ml/residual_detector.py",
        "kz/report/ml_dashboard.py",
        "kz/report/ml_report.py",
    ]:
        source = Path(path).read_text(encoding="utf-8")
        assert "prepare_training_data" in source, path

    explore = Path("kz/report/explore.py").read_text(encoding="utf-8")
    assert "is_training_eligible" in explore

    service = Path("kz/web/service.py").read_text(encoding="utf-8")
    assert service.count("COALESCE(price_basis, 'ambiguous') NOT IN") == 2


def test_damage_disclosed_rust_and_gearbox():
    """Regression coverage for `test_damage_disclosed_rust_and_gearbox`."""
    from kz.transform.damage import has_damage

    assert has_damage("есть классические рыжики на порогах")
    assert has_damage("не включается 5-я передача")
    assert has_damage("не включается кондиционер")
    assert not has_damage("без рыжиков, кузов идеальный")
    assert not has_damage("нет рыжиков")


def test_robust_z_ignores_single_outlier():
    import pandas as pd
    import numpy as np

    s = pd.Series(np.log([5e6] * 10 + [6e6] * 10 + [200e6]))
    med = s.median()
    mad = (s - med).abs().median()
    z_outlier = 0.6745 * (s.iloc[-1] - med) / mad
    z_normal = 0.6745 * (s.iloc[0] - med) / mad
    assert abs(z_outlier) > 3.5
    assert abs(z_normal) < 3.5


def test_description_after_km():
    r = listing_parser.parse_spec_line(
        "2014 г., Б/у седан, 3 л, бензин, КПП автомат, с пробегом 170 000 км, Срочно нужны деньги"
    )
    assert r["description"] == "Срочно нужны деньги"


def test_description_without_km_after_kpp():
    """Regression coverage for `test_description_without_km_after_kpp`."""
    r = listing_parser.parse_spec_line(
        "2008 г., Б/у седан, 1.6 л, бензин, КПП механика, Авто в хорошем состояний сел поехал"
    )
    assert r["description"] == "Авто в хорошем состояний сел поехал"


def test_description_without_km_and_kpp():
    r = listing_parser.parse_spec_line("1997 г., Б/у минивэн, 2 л, бензин, синий, литые диски")
    assert "синий" in r["description"]


def test_description_empty_when_no_seller_text():
    r = listing_parser.parse_spec_line("2026 г., 1.5 л, гибрид, КПП автомат")
    assert r["description"] == ""


def _dup_df(rows):
    cols = [
        "ad_id",
        "title",
        "year",
        "price_tenge",
        "mileage_km",
        "description",
        "condition",
        "labels",
    ]
    d = pd.DataFrame(rows, columns=cols)
    d["info_flags"] = ""
    return d


def test_repost_confirmed_by_mileage_is_flagged():
    d = _dup_df(
        [
            ("1", "Kia Rio", 2015, 5_000_000, 120_000, "продам", "б/у", None),
            ("2", "Kia Rio", 2015, 5_000_000, 120_000, None, "б/у", None),
        ]
    )
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "possible_repost").all()


def test_repost_unconfirmed_goes_to_info_only():
    """Regression coverage for `test_repost_unconfirmed_goes_to_info_only`."""
    d = _dup_df(
        [
            ("1", "BMW X5", 2016, 17_500_000, 210_000, None, "б/у", None),
            ("2", "BMW X5", 2016, 17_500_000, 241_200, None, "б/у", None),
        ]
    )
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "").all()
    assert (out["info_flags"].str.contains("repost_unconfirmed")).all()


def _dup_df_color(rows):
    cols = [
        "ad_id",
        "title",
        "year",
        "price_tenge",
        "mileage_km",
        "description",
        "condition",
        "labels",
        "color",
    ]
    d = pd.DataFrame(rows, columns=cols)
    d["info_flags"] = ""
    return d


def test_repost_different_base_color_not_confirmed():
    """Regression coverage for `test_repost_different_base_color_not_confirmed`."""
    d = _dup_df_color(
        [
            ("1", "Hyundai Sonata", 2023, 10_500_000, 100_000, None, "б/у", None, "белый"),
            (
                "2",
                "Hyundai Sonata",
                2023,
                10_500_000,
                100_000,
                None,
                "б/у",
                None,
                "черный металлик",
            ),
        ]
    )
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "").all()
    assert out["info_flags"].str.contains("repost_unconfirmed").all()


def test_repost_same_base_color_metallic_still_confirmed():
    """Regression coverage for `test_repost_same_base_color_metallic_still_confirmed`."""
    d = _dup_df_color(
        [
            ("1", "Toyota Camry", 2021, 14_500_000, 110_000, None, "б/у", None, "белый"),
            ("2", "Toyota Camry", 2021, 14_500_000, 110_000, None, "б/у", None, "белый металлик"),
        ]
    )
    out = clean.add_duplicate_flags(d)
    assert (out["dup_reasons"] == "possible_repost").all()


def _cars(rows):
    return pd.DataFrame(rows, columns=["ad_id", "brand", "model", "year", "price_tenge"])


def test_exact_hash_diff_model_is_flagged():
    hashes = pd.DataFrame(
        [
            {"ad_id": "1", "position": 1, "phash": "a" * 16},
            {"ad_id": "2", "position": 1, "phash": "a" * 16},
        ]
    )
    cars = _cars(
        [
            ("1", "Toyota", "Camry", 2015, 5_000_000),
            ("2", "Honda", "Civic", 2015, 5_000_000),
        ]
    )
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert {"1", "2"} == {out.iloc[0]["ad_id_a"], out.iloc[0]["ad_id_b"]}


def test_exact_hash_same_car_not_flagged_dealer_repost():
    """Regression coverage for `test_exact_hash_same_car_not_flagged_dealer_repost`."""
    hashes = pd.DataFrame(
        [
            {"ad_id": "1", "position": 1, "phash": "a" * 16},
            {"ad_id": "2", "position": 1, "phash": "a" * 16},
        ]
    )
    cars = _cars(
        [
            ("1", "Kia", "Rio", 2020, 6_000_000),
            ("2", "Kia", "Rio", 2020, 6_050_000),
        ]
    )
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert out.empty


def test_dealer_press_photo_across_trims_not_flagged():
    """Regression coverage for `test_dealer_press_photo_across_trims_not_flagged`."""
    hashes = pd.DataFrame(
        [
            {"ad_id": "1", "position": 1, "phash": "a" * 16},
            {"ad_id": "2", "position": 1, "phash": "a" * 16},
            {"ad_id": "3", "position": 1, "phash": "a" * 16},
        ]
    )
    cars = pd.DataFrame(
        [
            ("1", "OMODA", "S5 Life", 2025, 7_490_000, "новый", "Новая|Официальный дилер"),
            ("2", "OMODA", "S5 Prestige", 2025, 7_990_000, "новый", "Новая|Официальный дилер"),
            ("3", "Chevrolet", "Nexia", 2015, 3_000_000, "б/у", ""),
        ],
        columns=["ad_id", "brand", "model", "year", "price_tenge", "condition", "labels"],
    )
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    pairs = {frozenset((r.ad_id_a, r.ad_id_b)) for r in out.itertuples()}
    assert frozenset(("1", "2")) not in pairs
    assert frozenset(("1", "3")) in pairs
    assert frozenset(("2", "3")) in pairs


def test_near_hash_not_flagged_studio_lookalike():
    """Regression coverage for `test_near_hash_not_flagged_studio_lookalike`."""
    hashes = pd.DataFrame(
        [
            {"ad_id": "1", "position": 1, "phash": "0000000000000000"},
            {"ad_id": "2", "position": 1, "phash": "0000000000000001"},
        ]
    )
    cars = _cars(
        [
            ("1", "BMW", "X5", 2010, 8_000_000),
            ("2", "BMW", "X5", 2020, 8_000_000),
        ]
    )
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert out.empty


def test_single_photo_no_match_not_flagged():
    """Regression coverage for `test_single_photo_no_match_not_flagged`."""
    hashes = pd.DataFrame([{"ad_id": "1", "position": 1, "phash": "a" * 16}])
    cars = _cars([("1", "Toyota", "Camry", 2015, 5_000_000)])
    out = photo_dedup.find_cross_car_duplicates(hashes, cars)
    assert out.empty
    assert list(out.columns) == [
        "ad_id_a",
        "ad_id_b",
        "hamming_distance",
        "model_key_a",
        "price_a",
        "year_a",
        "model_key_b",
        "price_b",
        "year_b",
    ]


def test_confusion_matrix_counts():
    """Regression coverage for `test_confusion_matrix_counts`."""
    df = pd.DataFrame(
        {
            "ad_id": list("123456"),
            "is_suspicious": [1, 1, 0, 1, 0, 0],
            "verdict": ["fraud", "fraud", "fraud", "legit", "legit", "legit"],
        }
    )
    c = evaluate_detector.confusion(df)
    assert c == {"TP": 2, "FP": 1, "FN": 1, "TN": 2}


def test_weighted_confusion_uses_sampling_probability():
    """Regression coverage for `test_weighted_confusion_uses_sampling_probability`."""
    df = pd.DataFrame(
        {
            "is_suspicious": [1, 1, 0],
            "verdict": ["fraud", "legit", "fraud"],
            "stratum_population": [2, 2, 100],
            "stratum_sample_size": [2, 2, 1],
        }
    )
    c = evaluate_detector.weighted_confusion(df)
    assert c == {"TP": 1.0, "FP": 1.0, "FN": 100.0, "TN": 0.0}
    assert evaluate_detector._prf({"TP": 0, "FP": 2, "FN": 3, "TN": 0}) == (0.0, 0.0, 0.0)


def test_pg_value_coerces_float_strings():
    """Regression coverage for `test_pg_value_coerces_float_strings`."""
    assert listing_parser._pg_value("mileage_km", "50.0") == 50
    assert listing_parser._pg_value("mileage_km", "50") == 50
    assert listing_parser._pg_value("mileage_km", 50.0) == 50
    assert listing_parser._pg_value("mileage_km", "") is None
    assert listing_parser._pg_value("mileage_km", None) is None

    assert listing_parser._pg_value("title", "Toyota Camry") == "Toyota Camry"


def test_extract_avg_price():
    """Regression coverage for `test_extract_avg_price`."""
    assert enrich.extract_avg_price('..."brand":"BYD","avgPrice":23608000},...') == 23608000
    assert enrich.extract_avg_price("нет такого ключа") is None


def test_kolesa_cross_check_downgrades_false_positive():
    """Regression coverage for `test_kolesa_cross_check_downgrades_false_positive`."""
    d = pd.DataFrame(
        {
            "price_tenge": [22_500_000, 5_000_000],
            "kolesa_avg_price": [23_608_000, 20_000_000],
            "stat_reasons": ["price_anomaly_low", "price_anomaly_low"],
            "info_flags": ["", ""],
            "text_full": ["", ""],
            "condition": ["б/у", "б/у"],
            "labels": ["", ""],
        }
    )
    out = clean.exculpate(d)
    assert out.iloc[0]["stat_reasons"] == ""
    assert "kolesa_price_ok" in out.iloc[0]["info_flags"]
    assert out.iloc[1]["stat_reasons"] == "price_anomaly_low"


def test_kolesa_sentinel_and_missing_do_not_exculpate():
    """Regression coverage for `test_kolesa_sentinel_and_missing_do_not_exculpate`."""
    d = pd.DataFrame(
        {
            "price_tenge": [5_000_000, 5_000_000],
            "kolesa_avg_price": [-1, None],
            "stat_reasons": ["price_anomaly_low", "price_anomaly_low"],
            "info_flags": ["", ""],
            "text_full": ["", ""],
            "condition": ["б/у", "б/у"],
            "labels": ["", ""],
        }
    )
    out = clean.exculpate(d)
    assert (out["stat_reasons"] == "price_anomaly_low").all()


def test_extract_status_badge():
    """Regression coverage for `test_extract_status_badge`."""
    html = (
        '<div class="offer__parameters-mortgaged" '
        'data-test="offer-parameters">Аварийная/Не на ходу</div>'
    )
    assert enrich.parse_ad_page(html)["page_status_badge"] == "Аварийная/Не на ходу"

    assert enrich.parse_ad_page("<div>нет бейджа</div>")["page_status_badge"] == "-"


def test_enrich_parameters_tolerate_layout_and_label_variants():
    """Regression coverage for `test_enrich_parameters_tolerate_layout_and_label_variants`."""
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
    """Regression coverage for `test_enrich_vin_history_is_positive_only_evidence`."""
    positive = enrich.parse_ad_page("<section>У этого объявления есть История авто</section>")
    unknown = enrich.parse_ad_page("<a>Проверить Историю авто</a>")
    assert positive["has_vin"] == "Да"
    assert "has_vin" not in unknown


def test_enrich_explicit_missing_vin_is_not_positive():
    html = "<dl><dt>VIN:</dt><dd>не указан</dd></dl>"
    assert enrich.parse_ad_page(html)["has_vin"] == "Нет"


def test_used_zero_mileage_excludes_current_year_new():
    """Regression coverage for `test_used_zero_mileage_excludes_current_year_new`."""
    from kz.transform import clean

    cy = clean.CURRENT_YEAR
    df = pd.DataFrame(
        [
            {"year": cy, "condition": "б/у", "mileage_km": 0, "price_tenge": 7_600_000},
            {"year": cy - 5, "condition": "б/у", "mileage_km": 0, "price_tenge": 3_000_000},
        ]
    )
    df["age"] = cy - df["year"] + 1
    out = clean.apply_hard_rules(df)
    assert "used_but_zero_mileage" not in out.iloc[0]["rule_reasons"]
    assert "used_but_zero_mileage" in out.iloc[1]["rule_reasons"]


def test_young_car_cheap_cleared_when_declared_wreck():
    """Regression coverage for `test_young_car_cheap_cleared_when_declared_wreck`."""
    d = pd.DataFrame(
        {
            "price_tenge": [1_700_000, 3_500_000],
            "rule_reasons": ["young_car_cheap", "young_car_cheap"],
            "stat_reasons": ["", ""],
            "info_flags": ["", ""],
            "text_full": ["", ""],
            "damage_keywords": ["", ""],
            "condition": ["б/у", "б/у"],
            "labels": ["", ""],
            "page_status_badge": ["Аварийная/Не на ходу", None],
        }
    )
    out = clean.exculpate(d)
    assert "young_car_cheap" not in out.iloc[0]["rule_reasons"]
    assert "low_price_explained" in out.iloc[0]["info_flags"]
    assert "young_car_cheap" in out.iloc[1]["rule_reasons"]


def test_catch_up_references_real_modules():
    """Regression coverage for `test_catch_up_references_real_modules`."""
    import importlib.util
    from kz.ops import catch_up

    mods = (
        [s for _, s, _ in catch_up.KOLESA]
        + [s for _, s, _ in catch_up.CDN]
        + [s for _, s in catch_up.OFFLINE]
    )
    for m in mods:
        assert importlib.util.find_spec(m), f"catch_up ссылается на несуществующий {m}"


def test_junk_mileage_placeholder_detection():
    """Regression coverage for `test_junk_mileage_placeholder_detection`."""
    from kz.transform.data_quality import is_junk_mileage

    assert is_junk_mileage(777777)
    assert is_junk_mileage(999999)
    assert is_junk_mileage(888888)
    assert not is_junk_mileage(99999)
    assert not is_junk_mileage(111111)
    assert not is_junk_mileage(150000)
    assert not is_junk_mileage(0)
    assert not is_junk_mileage(None)
    assert not is_junk_mileage(float("nan"))


def test_parse_posted_date():
    from datetime import date
    from kz.ml.survival import parse_posted

    today = date(2026, 8, 10)
    assert parse_posted("18 июля", today) == date(2026, 7, 18)
    assert parse_posted("18 июл.", today) == date(2026, 7, 18)
    assert parse_posted("5 мая", today) == date(2026, 5, 5)
    assert parse_posted("сегодня", today) is None
    assert parse_posted(None, today) is None
    assert parse_posted("99 июля", today) is None


def test_posted_date_rolls_back_over_new_year():
    """Regression coverage for `test_posted_date_rolls_back_over_new_year`."""
    from datetime import date
    from kz.ml.survival import parse_posted

    jan = date(2026, 1, 3)
    assert parse_posted("28 декабря", jan) == date(2025, 12, 28)
    assert parse_posted("2 января", jan) == date(2026, 1, 2)
    assert parse_posted("3 января", jan) == date(2026, 1, 3)


def _survival_fixture():
    """Implement `_survival_fixture`."""
    cd = pd.DataFrame(
        {
            "ad_id": ["a", "b", "c", "d"],
            "posted_date": ["1 июля", "1 июля", "1 июля", "1 июля"],
            "status": ["archived", "deleted", "active", "active"],
            "price_tenge": [5e6, 6e6, 7e6, None],
        }
    )
    st = pd.DataFrame(
        {
            "ad_id": ["a", "b", "c", "d"],
            "checked_at": ["2026-07-11", "2026-07-06", None, None],
        }
    )
    sg = pd.DataFrame(
        {
            "ad_id": ["a", "b", "c", "d"],
            "last_seen": ["2026-07-11", "2026-07-06", "2026-07-21", "2026-07-21"],
        }
    )
    return cd, st, sg


def test_censored_ads_are_not_counted_as_sold():
    """Regression coverage for `test_censored_ads_are_not_counted_as_sold`."""
    from kz.ml.survival import build_lifespans

    d = build_lifespans(*_survival_fixture())

    ev = dict(zip(d.ad_id, d.event, strict=True))
    assert ev["a"] == 1 and ev["b"] == 1
    assert ev["c"] == 0
    assert "d" not in ev

    days = dict(zip(d.ad_id, d.days, strict=True))
    assert days["a"] == 10
    assert days["c"] == 20


def test_lifespan_end_comes_from_the_right_column():
    """Regression coverage for `test_lifespan_end_comes_from_the_right_column`."""
    from kz.ml.survival import build_lifespans

    cd, st, sg = _survival_fixture()

    st.loc[st.ad_id == "c", "checked_at"] = "2026-07-05"
    d = build_lifespans(cd, st, sg)
    assert dict(zip(d.ad_id, d.days, strict=True))["c"] == 20


def test_kaplan_meier_matches_plain_fraction_without_censoring():
    """Regression coverage for `test_kaplan_meier_matches_plain_fraction_without_censoring`."""
    from kz.ml.survival import kaplan_meier

    d = pd.DataFrame({"days": [2, 4, 6, 8, 10], "event": [1, 1, 1, 1, 1]})
    km = kaplan_meier(d, log=lambda *a, **k: None)
    assert abs(float(km.survival_function_at_times(5).iloc[0]) - 0.6) < 1e-9
    assert abs(float(km.survival_function_at_times(9).iloc[0]) - 0.2) < 1e-9


def test_cox_features_limited_by_event_count():
    """Regression coverage for `test_cox_features_limited_by_event_count`."""
    import numpy as np
    from kz.ml.survival import MIN_EVENTS_PER_FEATURE, cox_model

    assert MIN_EVENTS_PER_FEATURE >= 10

    rng = np.random.default_rng(0)
    n = 60
    d = pd.DataFrame(
        {
            "days": rng.integers(1, 40, n),
            "event": [1] * 15 + [0] * (n - 15),
            "price_ratio": rng.normal(1.0, 0.2, n),
            "age": rng.integers(1, 25, n),
            "photos_count": rng.integers(1, 15, n),
            "is_vip": rng.integers(0, 2, n),
        }
    )
    cph = cox_model(d, log=lambda *a, **k: None)
    assert len(cph.params_) == 15 // MIN_EVENTS_PER_FEATURE == 1, list(cph.params_)


def test_residual_detector_config():
    from kz.ml import residual_detector as r
    from kz.ml.train_price_model import FEATURES

    assert 0 < r.ALPHA < 0.5
    assert r.MIN_SUPPORT >= 1 and r.AGE_MAX >= 1
    assert r.FEATURES is FEATURES


def test_price_model_features_no_leakage():
    """Regression coverage for `test_price_model_features_no_leakage`."""
    from kz.ml import train_price_model as m

    banned = {
        "price_tenge",
        "log_price",
        "price_z",
        "kolesa_avg_price",
        "is_suspicious",
        "suspicion_reasons",
        "city",
        "views_count",
    }
    leak = set(m.FEATURES) & banned
    assert not leak, f"утечка цели в фичах модели: {leak}"


def _budget_file(tmp_path, monkeypatch):
    from kz.ops import catch_up

    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    return catch_up, f


def test_budget_accumulates_within_one_day(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_accumulates_within_one_day`."""
    cu, _ = _budget_file(tmp_path, monkeypatch)
    for _ in range(3):
        used = cu.charge_budget("kolesa", 20)
    assert used["kolesa"] == 60
    assert used["cdn"] == 0
    assert cu.load_budget_used()["kolesa"] == 60


def test_parser_and_catch_up_share_one_daily_budget(tmp_path, monkeypatch):
    """Regression coverage for `test_parser_and_catch_up_share_one_daily_budget`."""
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
    """Regression coverage for `test_parser_defaults_to_fresh_first_pages`."""
    import importlib
    from kz.collect import parser

    monkeypatch.delenv("KOLESA_MAX_PAGES", raising=False)
    monkeypatch.delenv("KOLESA_START_PAGE", raising=False)
    fresh = importlib.reload(parser)
    assert fresh.START_PAGE == 1
    assert fresh.MAX_PAGES_PER_CATEGORY == 3


def test_parser_fails_fast_when_listing_selectors_drift():
    """Regression coverage for `test_parser_fails_fast_when_listing_selectors_drift`."""
    import pytest
    from kz.collect import parser

    changed_html = "<html><body><div class='new-card-class'>машина</div></body></html>"
    with pytest.raises(parser.ListingSchemaError, match="changed its HTML"):
        parser.validate_listing_page(changed_html, [], 1, "almaty_3_7m")

    assert parser.validate_listing_page(changed_html, [], 20, "almaty_3_7m") == 0


def test_parser_fails_when_raw_cards_stop_parsing():
    """Regression coverage for `test_parser_fails_when_raw_cards_stop_parsing`."""
    import pytest
    from kz.collect import parser

    html = (
        "<html><body>"
        + "".join(f"<article class='js__a-card' data-id='{i}'></article>" for i in range(10))
        + "</body></html>"
    )
    with pytest.raises(parser.ListingSchemaError, match="parsed 1/10"):
        parser.validate_listing_page(html, [{"ad_id": "1"}], 2, "segment")


def test_parser_reports_open_freshness_boundary():
    """Regression coverage for `test_parser_reports_open_freshness_boundary`."""
    from kz.collect.parser import page_limit_has_unseen

    assert page_limit_has_unseen(3, 9, 23, 1, 3)
    assert not page_limit_has_unseen(3, 0, 23, 1, 3)
    assert not page_limit_has_unseen(2, 9, 23, 1, 3)
    assert not page_limit_has_unseen(30, 9, 23, 26, 30)  # deep backfill ≠ fresh


def test_parser_micro_limit_caps_exactly_ten_cards():
    """Regression coverage for `test_parser_micro_limit_caps_exactly_ten_cards`."""
    from kz.collect.parser import cap_cards_for_run

    cards = [{"ad_id": str(i)} for i in range(23)]
    selected, stop = cap_cards_for_run(cards, already_processed=0, limit=10)
    assert [row["ad_id"] for row in selected] == [str(i) for i in range(10)]
    assert stop
    assert cap_cards_for_run(cards, 0, 0) == (cards, False)


def test_parser_does_not_retry_a_corrupt_budget(monkeypatch):
    """Regression coverage for `test_parser_does_not_retry_a_corrupt_budget`."""
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
    """Regression coverage for `test_parser_run_status_marks_unhandled_failure`."""
    import json
    from kz.collect import parser

    status = tmp_path / "parser_status.json"
    monkeypatch.setattr(parser, "RUN_STATUS_FILE", str(status))
    parser.write_run_status(
        {"schema_version": 1, "status": "running", "segments": {}, "totals": {}}
    )
    parser.mark_unhandled_failure(RuntimeError("boom"))
    saved = json.loads(status.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["message"] == "RuntimeError: boom"
    assert saved["finished_at"]


def test_enrich_done_unions_csv_and_postgres(tmp_path, monkeypatch):
    """Regression coverage for `test_enrich_done_unions_csv_and_postgres`."""
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
    monkeypatch.setattr(
        pd, "read_sql", lambda *_args, **_kwargs: pd.DataFrame({"ad_id": ["db-only"]})
    )
    assert enrich.load_done() == {"csv-only", "db-only"}


def test_budget_reservation_does_not_overshoot(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_reservation_does_not_overshoot`."""
    from kz.ops import catch_up

    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(tmp_path / "budget.json"))
    assert catch_up.reserve_budget("kolesa", 2, 3)["kolesa"] == 2
    assert catch_up.reserve_budget("kolesa", 2, 3) is None
    assert catch_up.load_budget_used()["kolesa"] == 2


def test_chunk_refreshes_budget_after_rolling_window_moves(tmp_path, monkeypatch):
    """Regression coverage for `test_chunk_refreshes_budget_after_rolling_window_moves`."""
    from kz.ops import catch_up

    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(tmp_path / "budget.json"))
    monkeypatch.setitem(catch_up.DAILY_BUDGET, "kolesa", 2)
    gaps = iter([1, 0])
    monkeypatch.setattr(catch_up, "compute_gaps", lambda: {"backfill": next(gaps)})
    monkeypatch.setattr(catch_up, "run", lambda _script: 0)
    monkeypatch.setattr(catch_up, "count_429", lambda: 0)

    used = {"kolesa": 2, "cdn": 0}
    result = catch_up.run_one_chunk(
        "backfill",
        "unused",
        "backfill",
        "kolesa",
        used,
        run_spent={"kolesa": 0, "cdn": 0},
    )
    assert result == "done"
    assert used["kolesa"] == 1
    assert catch_up.load_budget_used()["kolesa"] == 1


def test_budget_is_rolling_and_does_not_reset_at_midnight(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_is_rolling_and_does_not_reset_at_midnight`."""
    import json
    from datetime import datetime, timedelta, timezone

    cu, f = _budget_file(tmp_path, monkeypatch)
    now = datetime(2026, 9, 2, 0, 10, tzinfo=timezone(timedelta(hours=5)))
    monkeypatch.setattr(cu, "_now", lambda: now)
    recent = now - timedelta(minutes=20)
    expired = now - timedelta(hours=24)
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

    monkeypatch.setattr(cu, "_now", lambda: recent + timedelta(hours=24))
    assert cu.load_budget_used() == {"kolesa": 0, "cdn": 0}


def test_budget_migrates_yesterdays_legacy_sum_conservatively(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_migrates_yesterdays_legacy_sum_conservatively`."""
    import json
    from datetime import datetime, timedelta, timezone

    cu, f = _budget_file(tmp_path, monkeypatch)
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone(timedelta(hours=5)))
    monkeypatch.setattr(cu, "_now", lambda: now)
    f.write_text(json.dumps({"days": {"2026-09-01": {"kolesa": 73, "cdn": 600}}}), encoding="utf-8")
    assert cu.load_budget_used() == {"kolesa": 73, "cdn": 600}
    migrated = json.loads(f.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == cu.BUDGET_SCHEMA_VERSION
    assert all(event.get("legacy") for event in migrated["events"])


def test_budget_allows_first_run_but_fails_closed_on_corrupt_file(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_allows_first_run_but_fails_closed_on_corrupt_file`."""
    import pytest

    cu, f = _budget_file(tmp_path, monkeypatch)
    assert cu.load_budget_used() == {"kolesa": 0, "cdn": 0}
    f.write_text("{не json", encoding="utf-8")
    with pytest.raises(cu.BudgetStateError, match="network access is blocked"):
        cu.load_budget_used()


def test_budget_reads_the_old_single_day_format(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_reads_the_old_single_day_format`."""
    import json
    from datetime import date

    cu, f = _budget_file(tmp_path, monkeypatch)
    f.write_text(
        json.dumps({"date": date.today().isoformat(), "kolesa": 150, "cdn": 40}), encoding="utf-8"
    )
    assert cu.load_budget_used() == {"kolesa": 150, "cdn": 40}


def test_budget_history_does_not_grow_forever(tmp_path, monkeypatch):
    """Regression coverage for `test_budget_history_does_not_grow_forever`."""
    import json
    from datetime import date, timedelta

    cu, f = _budget_file(tmp_path, monkeypatch)
    days = {
        (date.today() - timedelta(days=i)).isoformat(): {"kolesa": i, "cdn": 0}
        for i in range(1, 40)
    }
    cu._write_days(days)
    kept = json.loads(f.read_text(encoding="utf-8"))["days"]
    assert len(kept) == cu.BUDGET_KEEP_DAYS


def test_run_cap_is_a_second_defence_even_with_rolling_budget(tmp_path, monkeypatch):
    """Regression coverage for `test_run_cap_is_a_second_defence_even_with_rolling_budget`."""
    cu, _ = _budget_file(tmp_path, monkeypatch)
    full = cu.DAILY_BUDGET["kolesa"]
    fresh_day = {"kolesa": 0, "cdn": 0}
    spent_this_run = {"kolesa": full, "cdn": 0}
    assert not cu.budget_allows("kolesa", "enrich", 1000, fresh_day, spent_this_run)

    assert cu.budget_allows("kolesa", "enrich", 1000, fresh_day, None)


def test_nearly_finished_job_is_not_starved_at_the_quota_edge(tmp_path, monkeypatch):
    """Regression coverage for `test_nearly_finished_job_is_not_starved_at_the_quota_edge`."""
    cu, _ = _budget_file(tmp_path, monkeypatch)
    near_limit = {"kolesa": cu.DAILY_BUDGET["kolesa"] - 5, "cdn": 0}
    assert cu.budget_allows("kolesa", "enrich", 3, near_limit)
    assert not cu.budget_allows("kolesa", "enrich", 500, near_limit)


def test_429_detector_ignores_the_number_appearing_as_data():
    """Regression coverage for `test_429_detector_ignores_the_number_appearing_as_data`."""
    from kz.ops.catch_up import is_429_line

    for benign in [
        "наблюдений: 429",
        "ad_id=224297431",
        "цена 4290000",
        "2026-08-24 12:34:29 INFO готово",
        "скачано 429 фото",
    ]:
        assert not is_429_line(benign), benign
    for real in ["429: пауза 120с", "HTTP 429, пауза", "429 три подряд — стоп"]:
        assert is_429_line(real), real


def test_next_action_puts_rate_limiting_ahead_of_everything():
    """Regression coverage for `test_next_action_puts_rate_limiting_ahead_of_everything`."""
    from kz.ops.catch_up import next_action

    assert next_action(100, 0, 0, False) == "done"
    assert next_action(100, 0, 1, True) == "done"
    assert next_action(100, 50, 1, True) == "rate_limited"
    assert next_action(100, 50, 1, False) == "breaker"
    assert next_action(100, 100, 0, False) == "stuck"
    assert next_action(100, 101, 0, False) == "stuck"
    assert next_action(100, 50, 0, False) == "continue"


def test_risk_zones_are_anchored_to_the_ban_that_actually_happened():
    """Regression coverage for `test_risk_zones_are_anchored_to_the_ban_that_actually_happened`."""
    from kz.ops.catch_up import DAILY_BUDGET, risk_zone

    assert risk_zone(50)[0] == "low"
    assert risk_zone(200)[0] == "normal"
    assert risk_zone(260)[0] == "elevated"
    assert risk_zone(400)[0] == "high"
    assert risk_zone(DAILY_BUDGET["kolesa"])[0] in ("low", "normal")


def test_eta_accounts_for_pauses_not_just_requests():
    """Regression coverage for `test_eta_accounts_for_pauses_not_just_requests`."""
    from kz.core.pacing import mean_pause
    from kz.ops.catch_up import eta_minutes

    naive = 200 * 3.0 / 60
    assert eta_minutes(200) > naive * 2
    assert eta_minutes(200) > 200 * mean_pause(4.0, 8.0) / 60


def test_budget_is_configurable_without_touching_code():
    """Regression coverage for `test_budget_is_configurable_without_touching_code`."""
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
    """Regression coverage for `test_budget_flag_rejects_nonsense`."""
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


def test_catch_up_value_jobs_are_exculpation_fillers():
    """Regression coverage for `test_catch_up_value_jobs_are_exculpation_fillers`."""
    import importlib.util
    from kz.ops import catch_up

    keys = [k for _, _, k in catch_up.VALUE_JOBS]
    assert keys == ["enrich", "backfill"]
    assert all(j in catch_up.KOLESA for j in catch_up.VALUE_JOBS)
    assert "status" not in keys and "photo" not in keys
    for _, mod, _ in catch_up.VALUE_JOBS:
        assert importlib.util.find_spec(mod)

    bkeys = [k for _, _, k in catch_up.BACKFILL_JOBS]
    assert bkeys == ["backfill"]
    assert all(j in catch_up.VALUE_JOBS for j in catch_up.BACKFILL_JOBS)


def test_catch_up_429_detection_not_fooled_by_numbers():
    """Regression coverage for `test_catch_up_429_detection_not_fooled_by_numbers`."""
    from kz.ops import catch_up

    assert catch_up.is_429_line("2026-01-01 12:00:00  INFO  429: пауза 120с")
    assert catch_up.is_429_line("Стоп: 429 подряд — сайт лимитирует")
    assert not catch_up.is_429_line("наблюдений сегодня: 429, всего: 429")
    assert not catch_up.is_429_line("2026-07-18 20:15:23,429  INFO  карточек: 23")
    assert not catch_up.is_429_line("ad_id 224290000 обработан")


def test_catch_up_until_done_next_action():
    """Regression coverage for `test_catch_up_until_done_next_action`."""
    from kz.ops.catch_up import next_action

    assert next_action(500, 380, 0, False) == "continue"

    assert next_action(120, 0, 0, False) == "done"
    assert next_action(120, 0, 0, True) == "done"

    assert next_action(500, 450, 0, True) == "rate_limited"

    assert next_action(500, 480, 1, False) == "breaker"

    assert next_action(30, 30, 0, False) == "stuck"
    assert next_action(30, 31, 0, False) == "stuck"


def test_catch_up_chunk_sizes_match_jobs():
    """Regression coverage for `test_catch_up_chunk_sizes_match_jobs`."""
    from kz.ops import catch_up
    from kz.collect import check_status, enrich, backfill_avgprice, photo_dedup

    assert catch_up.CHUNK_MAX["status"] == check_status.MAX_CHECKS_PER_RUN
    assert catch_up.CHUNK_MAX["enrich"] == enrich.MAX_PER_RUN
    assert catch_up.CHUNK_MAX["backfill"] == backfill_avgprice.MAX_PER_RUN
    assert catch_up.CHUNK_MAX["photo"] == photo_dedup.MAX_PER_RUN


def test_catch_up_budget_allows_near_done_at_edge():
    """Regression coverage for `test_catch_up_budget_allows_near_done_at_edge`."""
    from kz.ops import catch_up

    B = catch_up.DAILY_BUDGET["kolesa"]
    cm = catch_up.CHUNK_MAX["enrich"]
    assert catch_up.budget_allows("kolesa", "status", 10**6, {"kolesa": 0, "cdn": 0})
    edge = B - (cm - 1)
    assert not catch_up.budget_allows("kolesa", "enrich", 10**6, {"kolesa": edge, "cdn": 0})
    assert catch_up.budget_allows("kolesa", "enrich", 1, {"kolesa": edge, "cdn": 0})


def test_catch_up_status_thresholds_match_check_status():
    """Regression coverage for `test_catch_up_status_thresholds_match_check_status`."""
    from kz.ops import catch_up
    from kz.collect import check_status

    assert catch_up.STATUS_STALE_DAYS == check_status.STALE_DAYS
    assert catch_up.STATUS_RECHECK_DAYS == check_status.RECHECK_DAYS


def test_status_recheck_and_listing_inference():
    """Regression coverage for `test_status_recheck_and_listing_inference`."""
    from kz.collect.check_status import needs_status_check, infer_active_from_listing

    # needs_status_check(cur_status, seen_days, checked_days)
    assert not needs_status_check("archived", 30, None)
    assert not needs_status_check("deleted", 30, 30)
    assert not needs_status_check("active", 0, None)
    assert not needs_status_check(None, 5, 1)
    assert needs_status_check("active", 5, 10)
    assert needs_status_check(None, 5, None)
    # infer_active_from_listing(cur_status, seen_days, seen_after_check)
    assert infer_active_from_listing(None, 0, True)
    assert not infer_active_from_listing("active", 0, True)
    assert infer_active_from_listing("archived", 0, True)
    assert not infer_active_from_listing("archived", 0, False)
    assert not infer_active_from_listing(None, 5, True)


def test_catch_up_budget_legacy_recovery_and_corruption(tmp_path, monkeypatch):
    """Regression coverage for `test_catch_up_budget_legacy_recovery_and_corruption`."""
    import pytest
    from kz.ops import catch_up

    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    f.write_text('{"date":"2000-01-01","kolesa":399,"cdn":5}', encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 0, "cdn": 0}
    catch_up.save_budget_used({"kolesa": 150, "cdn": 300})
    assert catch_up.load_budget_used() == {"kolesa": 150, "cdn": 300}
    f.write_text("{ битый json", encoding="utf-8")
    with pytest.raises(catch_up.BudgetStateError):
        catch_up.load_budget_used()


def test_duplicate_groups_keep_repost_in_one_fold():
    """Regression coverage for `test_duplicate_groups_keep_repost_in_one_fold`."""
    import pandas as pd
    from kz.ml.train_price_model import duplicate_groups

    d = pd.DataFrame(
        [
            {
                "ad_id": "1",
                "brand": "Toyota",
                "model": "Camry",
                "year": 2020,
                "mileage_km": 80000,
                "engine_volume": 2.5,
                "body_type": "седан",
                "text_full": "один хозяин, родной окрас, зимняя резина",
                "price_tenge": 10_000_000,
            },
            {
                "ad_id": "2",
                "brand": "Toyota",
                "model": "Camry",
                "year": 2020,
                "mileage_km": 80000,
                "engine_volume": 2.5,
                "body_type": "седан",
                "text_full": "один хозяин, родной окрас, зимняя резина",
                "price_tenge": 9_700_000,
            },
            {
                "ad_id": "3",
                "brand": "Toyota",
                "model": "Camry",
                "year": 2020,
                "mileage_km": 80000,
                "engine_volume": 2.5,
                "body_type": "седан",
                "text_full": "",
                "price_tenge": 10_000_000,
            },
        ]
    )
    g = duplicate_groups(d)
    assert g.iloc[0] == g.iloc[1]
    assert g.iloc[2] != g.iloc[0]


def test_temporal_holdout_is_future_and_removes_group_overlap():
    import pandas as pd
    from kz.ml.train_price_model import duplicate_groups, temporal_holdout

    rows = []
    for i in range(120):
        rows.append(
            {
                "ad_id": str(i),
                "scraped_at": pd.Timestamp("2026-01-01") + pd.Timedelta(i, unit="D"),
                "brand": "B",
                "model": f"M{i}",
                "year": 2020,
                "mileage_km": i + 1000,
                "engine_volume": 2.0,
                "body_type": "седан",
                "text_full": f"уникальное описание машины номер {i}",
            }
        )

    rows[-1].update(
        {
            "brand": rows[0]["brand"],
            "model": rows[0]["model"],
            "mileage_km": rows[0]["mileage_km"],
            "text_full": rows[0]["text_full"],
        }
    )
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
        brand="Toyota",
        model="Camry",
        year=2020,
        mileage_km=0,
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


def test_runtime_python_contract_is_consistent():
    """Regression coverage for `test_runtime_python_contract_is_consistent`."""
    import tomllib
    from pathlib import Path

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.13,<3.14"
    assert "FROM python:3.13-slim" in Path("Dockerfile").read_text(encoding="utf-8")
    assert 'python-version: "3.13"' in Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Python 3.13.x" in Path("docs/SETUP.md").read_text(encoding="utf-8")


def test_web_container_includes_every_routed_model_artifact():
    """A routed image must carry the general model, specialist, and metadata."""
    from pathlib import Path

    docker = Path("Dockerfile").read_text(encoding="utf-8")
    ignore = Path(".dockerignore").read_text(encoding="utf-8")
    ignore_lines = {line.strip() for line in ignore.splitlines()}
    assert "ARG MODEL_DIR=deploy/models" in docker
    assert "data/" in ignore_lines
    assert "deploy/models/" not in ignore_lines
    for name in ("price_model.cbm", "price_cheap_specialist.cbm", "price_model.metadata.json"):
        assert name in docker
        assert (Path("deploy/models") / name).is_file()


def test_ci_smoke_artifact_matches_runtime_schema(tmp_path):
    """Regression coverage for `test_ci_smoke_artifact_matches_runtime_schema`."""
    import json
    from kz.ops.create_smoke_artifact import create
    from kz.ml.train_price_model import FEATURES

    out = tmp_path / "models"
    create(out)
    assert (out / "price_model.cbm").stat().st_size > 0
    assert (out / "price_cheap_specialist.cbm").stat().st_size > 0
    meta = json.loads((out / "price_model.metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert meta["artifact_purpose"] == "ci_smoke_test_only"
    assert meta["target"] == "log(first_seen_listing_price_tenge)"


def test_model_loader_rejects_incompatible_feature_schema(tmp_path, monkeypatch):
    """Regression coverage for `test_model_loader_rejects_incompatible_feature_schema`."""
    import json
    import pytest
    from kz.ml import train_price_model as tm

    model = tmp_path / "model.cbm"
    meta = tmp_path / "model.json"
    model.write_bytes(b"not reached")
    meta.write_text(
        json.dumps(
            {
                "schema_version": tm.ARTIFACT_SCHEMA_VERSION,
                "features": ["wrong_feature"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tm, "MODEL_PATH", model)
    monkeypatch.setattr(tm, "METADATA_PATH", meta)
    with pytest.raises(ValueError, match="feature schema"):
        tm.load_artifact()


def test_labeling_queue_contains_positive_residual_and_control_strata():
    """Regression coverage for `test_labeling_queue_contains_positive_residual_and_control_strata`."""
    import pandas as pd
    from kz.report.explore import select_labeling_rows

    n = 27
    d = pd.DataFrame(
        {
            "ad_id": [str(i) for i in range(n)],
            "is_suspicious": [1] * 3 + [0] * (n - 3),
            "both_detectors_low": [0] * n,
            "price_z": list(range(n)),
            "residual_gap": [0.0] * n,
        }
    )
    residual = pd.Series([False] * 3 + [True] * 4 + [False] * 20)
    q = select_labeling_rows(d, residual, control_n=5)
    counts = q["sampling_stratum"].value_counts().to_dict()
    assert counts == {
        "random_control": 5,
        "residual_candidate": 4,
        "rule_positive": 3,
    }
    assert set(q.loc[q["sampling_stratum"] == "random_control", "stratum_population"]) == {20}


def test_pacing_never_faster_than_base_range():
    """Regression coverage for `test_pacing_never_faster_than_base_range`."""
    import random as _r
    from kz.core import pacing

    lo, hi = 4.0, 8.0
    rng = _r.Random(0)
    pauses = [pacing.human_pause(lo, hi, rng=rng) for _ in range(2000)]
    assert min(pauses) >= lo
    assert max(pauses) <= hi * pacing.LONG_TAIL_MULT

    assert sum(pauses) / len(pauses) > (lo + hi) / 2


def test_pacing_long_break_cadence():
    """Regression coverage for `test_pacing_long_break_cadence`."""
    import random as _r
    from kz.core import pacing

    rng = _r.Random(1)
    hits = [i for i in range(1, 61) if pacing.long_break(i, rng=rng) is not None]
    assert hits == list(range(pacing.BREAK_EVERY, 61, pacing.BREAK_EVERY))
    assert pacing.long_break(0, rng=rng) is None


def test_pacing_mean_pause_accounts_for_breaks():
    """Regression coverage for `test_pacing_mean_pause_accounts_for_breaks`."""
    from kz.core import pacing

    assert pacing.mean_pause(4.0, 8.0) > 6.0


def test_kolesa_jobs_use_shared_pacing():
    """Regression coverage for `test_kolesa_jobs_use_shared_pacing`."""
    from pathlib import Path

    for f in [
        "kz/collect/enrich.py",
        "kz/collect/check_status.py",
        "kz/collect/backfill_avgprice.py",
    ]:
        src = Path(f).read_text(encoding="utf-8")
        assert "pacing.polite_sleep" in src, f
        assert "time.sleep(random.uniform" not in src, f


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
    """Regression coverage for `test_catch_up_risk_zones_match_observed_ban`."""
    from kz.ops import catch_up

    assert catch_up.risk_zone(50)[0] == "low"
    assert catch_up.risk_zone(catch_up.DEFAULT_KOLESA_BUDGET)[0] == "normal"
    assert catch_up.risk_zone(270)[0] == "elevated"
    assert catch_up.risk_zone(500)[0] == "high"

    order = [z[1] for z in catch_up.RISK_ZONES]
    seen = [catch_up.risk_zone(n)[0] for n in (1, 100, 101, 200, 201, 270, 271, 10**6)]
    assert [order.index(s) for s in seen] == sorted(order.index(s) for s in seen)


def test_catch_up_eta_grows_with_volume():
    from kz.ops import catch_up

    assert catch_up.eta_minutes(0) == 0
    assert catch_up.eta_minutes(540) > catch_up.eta_minutes(200) > 0


def _label_cards_source() -> str:
    """Implement `_label_cards_source`."""
    from pathlib import Path

    return "\n".join(
        f.read_text(encoding="utf-8") for f in sorted(Path("kz/report/label_cards").glob("*.py"))
    )


def test_label_cards_never_requests_kolesa():
    """Regression coverage for `test_label_cards_never_requests_kolesa`."""
    src = _label_cards_source()
    for bad in ("requests.get", "requests.head", "urlopen", "httpx"):
        assert bad not in src, bad


def test_label_cards_help_covers_real_flags():
    """Regression coverage for `test_label_cards_help_covers_real_flags`."""
    from kz.report import label_cards
    from pathlib import Path

    src = Path("kz/transform/clean.py").read_text(encoding="utf-8")

    for flag in [
        "price_anomaly_low",
        "young_car_cheap",
        "possible_repost",
        "shared_photo_diff_car",
        "used_but_zero_mileage",
        "cheap_and_urgent",
    ]:
        assert flag in src, f"{flag} исчез из clean.py — обнови FLAG_HELP"
        assert flag in label_cards.FLAG_HELP, f"нет подсказки для {flag}"

    for flag, (what, fr, lg) in label_cards.FLAG_HELP.items():
        assert what and fr and lg, flag
        assert "fraud" in fr and "legit" in lg, flag


def test_label_cards_csv_line_matches_labels_schema():
    """Regression coverage for `test_label_cards_csv_line_matches_labels_schema`."""
    import csv
    from io import StringIO
    from kz.report import label_cards as lc

    header = lc.journal_header()
    assert header[0] == "ad_id"

    i = header.index("verdict")
    assert header[i + 1] == "comment"
    assert header[i - 1] == "seller_comment"

    line = "123" + "," * 8 + "legit,причина" + "," * (len(header) - 10)
    assert len(next(csv.reader(StringIO(line)))) == len(header)


def test_label_cards_money_and_fmt_handle_missing():
    """Regression coverage for `test_label_cards_money_and_fmt_handle_missing`."""
    from kz.report import label_cards

    assert label_cards.money(None) == "—"
    assert label_cards.money(float("nan")) == "—"
    assert label_cards.fmt(None) == "—"
    assert label_cards.fmt(float("nan")) == "—"
    assert label_cards.fmt("") == "—"
    assert label_cards.fmt(2007.0) == "2007"


def test_label_cards_money_reads_naturally():
    """Use millions only from one million; 240,000 should remain readable."""
    from kz.report import label_cards as lc

    assert lc.money(240000) == "240 000 ₸"
    assert lc.money(95000) == "95 000 ₸"
    assert lc.money(1_000_000) == "1M ₸"
    assert lc.money(4_900_000) == "4.9M ₸"
    assert lc.money(12_000_000) == "12M ₸"


def test_label_cards_price_bands_are_monotonic():
    """Price-band language must remain monotonic across ratio boundaries."""
    from kz.report import label_cards as lc

    labels = [lab for _, lab in lc.PRICE_BANDS]
    seen = [lc.price_band(r) for r in (0.2, 0.59, 0.61, 0.84, 0.9, 1.2, 1.5, 9.0)]
    idx = [labels.index(s) for s in seen]
    assert idx == sorted(idx)
    assert lc.price_band(0.59) == "far below the market"
    assert lc.price_band(1.0) == "within the average range"
    # Boundaries belong to the next band rather than falling through.
    assert lc.price_band(0.60) == "noticeably below average"
    assert lc.price_band(1.40).startswith("well above")


def test_label_cards_gallery_and_keyboard_present():
    """Review needs a large photo, thumbnails, lightbox, and shortcuts."""
    src = _label_cards_source()
    for token in ['class="hero"', 'class="thumb', 'id="box"', "openBox", "setVerdict", "focusCard"]:
        assert token in src, token

    assert 'TEMPLATE = r"""' in src


def test_catch_up_budget_keeps_calendar_history_only_as_audit(tmp_path, monkeypatch):
    """Regression coverage for `test_catch_up_budget_keeps_calendar_history_only_as_audit`."""
    from kz.ops import catch_up

    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    assert catch_up.charge_budget("kolesa", 20) == {"kolesa": 20, "cdn": 0}
    assert catch_up.charge_budget("kolesa", 20) == {"kolesa": 40, "cdn": 0}
    assert catch_up.charge_budget("cdn", 300)["cdn"] == 300
    assert catch_up.load_budget_used() == {"kolesa": 40, "cdn": 300}

    import json

    days = json.loads(f.read_text(encoding="utf-8"))["days"]
    days["2000-01-01"] = {"kolesa": 999, "cdn": 999}
    f.write_text(json.dumps({"days": days}), encoding="utf-8")
    assert catch_up.load_budget_used() == {"kolesa": 40, "cdn": 300}


def test_catch_up_budget_file_reads_old_format(tmp_path, monkeypatch):
    """Regression coverage for `test_catch_up_budget_file_reads_old_format`."""
    from kz.ops import catch_up
    from datetime import date

    f = tmp_path / "budget.json"
    monkeypatch.setattr(catch_up, "BUDGET_FILE", str(f))
    f.write_text('{"date":"%s","kolesa":180,"cdn":7}' % date.today().isoformat(), encoding="utf-8")
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
    """Regression coverage for `test_catch_up_per_run_cap_is_defence_in_depth`."""
    from kz.ops import catch_up

    B = catch_up.DAILY_BUDGET["kolesa"]
    fresh_day = {"kolesa": 0, "cdn": 0}
    spent_run = {"kolesa": B, "cdn": 0}
    assert not catch_up.budget_allows("kolesa", "enrich", 10**6, fresh_day, spent_run)

    assert catch_up.budget_allows("kolesa", "enrich", 10**6, fresh_day)
    assert catch_up.budget_allows("kolesa", "enrich", 10**6, fresh_day, {"kolesa": 0, "cdn": 0})


def test_avgprice_sentinel_never_acts_as_price():
    """Regression coverage for `test_avgprice_sentinel_never_acts_as_price`."""
    import numpy as np
    import pandas as pd
    from kz.transform.clean import exculpate

    base = dict(
        stat_reasons="price_anomaly_low",
        rule_reasons="",
        info_flags="",
        suspicion_reasons="",
        is_suspicious=1,
        price_tenge=1_000_000,
        text_full="",
        condition="",
        labels="",
        customs_cleared="Да",
    )
    df = pd.DataFrame(
        [
            {**base, "kolesa_avg_price": -1},
            {**base, "kolesa_avg_price": np.nan},
            {**base, "kolesa_avg_price": 1_100_000},
            {**base, "kolesa_avg_price": 5_000_000},
        ]
    )
    out = exculpate(df.copy())
    assert list(out["stat_reasons"]) == [
        "price_anomaly_low",
        "price_anomaly_low",
        "",
        "price_anomaly_low",
    ]
    assert "kolesa_price_ok" in out.loc[2, "info_flags"]
    assert "kolesa_price_ok" not in out.loc[0, "info_flags"]


def test_badge_sentinel_never_exculpates():
    """Regression coverage for `test_badge_sentinel_never_exculpates`."""
    import numpy as np
    import pandas as pd
    from kz.transform.clean import exculpate

    base = dict(
        stat_reasons="price_anomaly_low",
        rule_reasons="",
        info_flags="",
        suspicion_reasons="",
        is_suspicious=1,
        price_tenge=1_000_000,
        text_full="",
        condition="",
        labels="",
        customs_cleared="Да",
    )
    df = pd.DataFrame(
        [
            {**base, "page_status_badge": "-"},
            {**base, "page_status_badge": np.nan},
            {**base, "page_status_badge": "Аварийная/Не на ходу"},
        ]
    )
    out = exculpate(df.copy())
    assert list(out["stat_reasons"]) == ["price_anomaly_low", "price_anomaly_low", ""]


def test_avgprice_and_badge_stay_out_of_model():
    """Regression coverage for `test_avgprice_and_badge_stay_out_of_model`."""
    from kz.ml.train_price_model import FEATURES

    assert "kolesa_avg_price" not in FEATURES
    assert "page_status_badge" not in FEATURES


def test_label_cards_hint_ignores_sentinel():
    """Regression coverage for `test_label_cards_hint_ignores_sentinel`."""
    from kz.report import label_cards as lc

    assert lc.price_verdict_hint({"kolesa_avg_price": -1, "price_tenge": 1_000_000}) == ""
    assert lc.price_verdict_hint({"kolesa_avg_price": 2_000_000, "price_tenge": 1_000_000}) != ""


def test_network_dag_is_paused_and_single_run():
    """Regression coverage for `test_network_dag_is_paused_and_single_run`."""
    from pathlib import Path

    src = Path("airflow/dags/kolesa_pipeline_dag.py").read_text(encoding="utf-8")
    assert "is_paused_upon_creation=True" in src
    assert "max_active_runs=1" in src
    assert "catchup=False" in src


def test_offline_dag_has_no_network_jobs():
    """Regression coverage for `test_offline_dag_has_no_network_jobs`."""
    from pathlib import Path

    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    for net in [
        "kz.collect.parser",
        "kz.collect.enrich",
        "kz.collect.check_status",
        "kz.collect.photo_dedup",
        "kz.collect.backfill_avgprice",
        "kz.ops.catch_up",
    ]:
        assert net not in src, f"{net} — сетевой, ему не место в офлайн-DAG"
    assert "schedule=None" in src
    assert "is_paused_upon_creation=False" in src


def test_offline_dag_covers_whole_ml_chain():
    """Regression coverage for `test_offline_dag_covers_whole_ml_chain`."""
    from pathlib import Path
    from kz.ops.run_all import ML_CHAIN, OFFLINE_CHAIN

    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    for _, cmd in ML_CHAIN + OFFLINE_CHAIN:
        assert cmd[-1] in src, f"{cmd[-1]} есть в run_all, но нет в офлайн-DAG"


def test_collect_dag_covers_the_collect_chain():
    """Regression coverage for `test_collect_dag_covers_the_collect_chain`."""
    from pathlib import Path
    from kz.ops.run_all import COLLECT_CHAIN

    src = Path("airflow/dags/kolesa_pipeline_dag.py").read_text(encoding="utf-8")
    for _, cmd in COLLECT_CHAIN:
        mod = cmd[cmd.index("-m") + 1]
        assert mod in src, f"{mod} есть в COLLECT_CHAIN, но нет в сетевом DAG"


def test_offline_dag_dependencies_respect_artifacts():
    """Regression coverage for `test_offline_dag_dependencies_respect_artifacts`."""
    from pathlib import Path

    src = Path("airflow/dags/kolesa_offline_dag.py").read_text(encoding="utf-8")
    assert "clean >> monitor >> train >> dashboard" in src
    assert "train >> residual >> report" in src
    assert "clean >> explore >> cards" in src

    assert "monitor >> train" in src
    assert "train >> monitor" not in src

    import re as _re

    tasks = set(_re.findall(r"^\s*(\w+)\s*=\s*job\(", src, _re.M))
    edges = set()
    for line in src.splitlines():
        line = line.split("#")[0]
        if ">>" not in line:
            continue
        parts = [p.strip(" []") for p in line.split(">>")]

        for a, b in zip(parts, parts[1:]):
            for x in (n.strip() for n in a.split(",")):
                for y in (n.strip() for n in b.split(",")):
                    if x in tasks and y in tasks:
                        edges.add((x, y))
    leaves = {
        t for t in tasks if t != "state" and not any(a == t and b != "state" for a, b in edges)
    }
    waited = {a for a, b in edges if b == "state"}
    assert leaves <= waited, "финальный таск не ждёт ветки: " + ", ".join(sorted(leaves - waited))


def test_collect_dag_delegates_budget_to_catch_up():
    """Regression coverage for `test_collect_dag_delegates_budget_to_catch_up`."""
    from pathlib import Path

    src = Path("airflow/dags/kolesa_pipeline_dag.py").read_text(encoding="utf-8")
    assert "kz.ops.catch_up" in src
    for direct in [
        "kz.collect.enrich",
        "kz.collect.check_status",
        "kz.collect.photo_dedup",
        "kz.collect.backfill_avgprice",
    ]:
        assert direct not in src, f"{direct} вызван напрямую — обойдёт суточный лимит catch_up"


def _tmp_journal(tmp_path, monkeypatch):
    """Implement `_tmp_journal`."""
    import csv
    from kz.report import label_cards as lc

    dst = tmp_path / "manual_labels.csv"
    header = [
        "ad_id",
        "url",
        "title",
        "year",
        "price_tenge",
        "mileage_km",
        "suspicion_reasons",
        "seller_comment",
        "verdict",
        "comment",
    ]
    rows = [
        [
            "225936503",
            "https://kolesa.kz/a/show/225936503",
            "Chevrolet Onix",
            "2023",
            "1700000",
            "100000",
            "young_car_cheap",
            "Оникс аварийный",
            "legit",
            "честно битая",
        ],
        [
            "225480956",
            "https://kolesa.kz/a/show/225480956",
            "Toyota Highlander",
            "2008",
            "4900000",
            "300030",
            "price_anomaly_low",
            "на ходу",
            "",
            "",
        ],
        [
            "226154999",
            "https://kolesa.kz/a/show/226154999",
            "Hyundai Accent",
            "2012",
            "1500000",
            "236500",
            "price_anomaly_low",
            "Был пожар, документы в порядке",
            "",
            "",
        ],
    ]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    from kz.report.label_cards import journal as lc_journal

    monkeypatch.setattr(lc_journal, "LABELS_CSV", str(dst))
    monkeypatch.setattr(lc_journal, "LABELS_PREV", str(tmp_path / "prev.csv"))
    monkeypatch.setattr(lc_journal, "_snapshot_done", False)
    return lc, dst


def test_upsert_keeps_one_row_per_ad(tmp_path, monkeypatch):
    """Regression coverage for `test_upsert_keeps_one_row_per_ad`."""
    import csv

    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    n0 = len(list(csv.DictReader(dst.open(encoding="utf-8"))))
    lc.upsert_verdict("111", "legit", "сначала так", {})
    n1 = len(list(csv.DictReader(dst.open(encoding="utf-8"))))
    lc.upsert_verdict("111", "fraud", "передумал", {})
    lc.upsert_verdict("111", "unknown", "не понять", {})
    rows = list(csv.DictReader(dst.open(encoding="utf-8")))
    assert n1 == n0 + 1
    assert len(rows) == n1
    mine = [r for r in rows if r["ad_id"] == "111"]
    assert len(mine) == 1
    assert (mine[0]["verdict"], mine[0]["comment"]) == ("unknown", "не понять")


def test_upsert_updates_existing_queue_row_in_place(tmp_path, monkeypatch):
    """Regression coverage for `test_upsert_updates_existing_queue_row_in_place`."""
    import csv

    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    rows = list(csv.DictReader(dst.open(encoding="utf-8")))
    existing = next(r for r in rows if not r["verdict"])
    pos = rows.index(existing)
    lc.upsert_verdict(existing["ad_id"], "legit", "проверено", {})
    after = list(csv.DictReader(dst.open(encoding="utf-8")))
    assert len(after) == len(rows)
    assert after[pos]["ad_id"] == existing["ad_id"]
    assert after[pos]["verdict"] == "legit"
    assert after[pos]["title"] == existing["title"]


def test_upsert_preserves_other_rows_and_backup(tmp_path, monkeypatch):
    """Regression coverage for `test_upsert_preserves_other_rows_and_backup`."""
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

    for a, b in zip(before, after[: len(before)], strict=True):
        assert all(b[key] == value for key, value in a.items())
        assert b["sampling_stratum"] == ""
        assert b["stratum_population"] == ""
        assert b["stratum_sample_size"] == ""
    assert prev.exists() and prev.read_text(encoding="utf-8") == before_text


def test_upsert_migrates_and_backfills_sampling_metadata(tmp_path, monkeypatch):
    """Legacy rows gain the complete weighting contract without losing facts."""
    import csv

    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    lc.upsert_verdict(
        "225480956",
        "unknown",
        "insufficient evidence",
        {
            "sampling_stratum": "residual_candidate",
            "stratum_population": 174,
            "stratum_sample_size": 174,
        },
    )
    rows = list(csv.DictReader(dst.open(encoding="utf-8")))
    row = next(item for item in rows if item["ad_id"] == "225480956")
    assert row["sampling_stratum"] == "residual_candidate"
    assert row["stratum_population"] == "174"
    assert row["stratum_sample_size"] == "174"
    assert row["title"] == "Toyota Highlander"


def test_journal_facts_carries_complete_sampling_metadata():
    """Browser saves retain every field required for weighted evaluation."""
    import pandas as pd
    from kz.report import label_cards as lc

    rows = pd.DataFrame(
        {
            "ad_id": ["111"],
            "stratum": ["random_control"],
            "stratum_population": [4_745],
            "stratum_sample_size": [50],
        }
    )
    facts = lc.journal_facts(rows)["111"]
    assert facts["sampling_stratum"] == "random_control"
    assert facts["stratum_population"] == 4_745
    assert facts["stratum_sample_size"] == 50


def test_upsert_writes_ints_without_dot_zero(tmp_path, monkeypatch):
    """Regression coverage for `test_upsert_writes_ints_without_dot_zero`."""
    import csv

    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    lc.upsert_verdict(
        "111",
        "fraud",
        "",
        {
            "year": 1994.0,
            "price_tenge": 240000.0,
            "mileage_km": float("nan"),
            "seller_comment": 'текст с "кавычками", запятой',
        },
    )
    row = [r for r in csv.DictReader(dst.open(encoding="utf-8")) if r["ad_id"] == "111"][0]
    assert row["year"] == "1994"
    assert row["price_tenge"] == "240000"
    assert row["mileage_km"] == ""
    assert row["seller_comment"] == 'текст с "кавычками", запятой'


def test_upsert_rejects_bad_verdict(tmp_path, monkeypatch):
    """Regression coverage for `test_upsert_rejects_bad_verdict`."""
    import pytest as _pt

    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    before = dst.read_text(encoding="utf-8")
    for bad in ("мошенник", "FRAUD", "", "legit "):
        with _pt.raises(ValueError):
            lc.upsert_verdict("111", bad, "", {})
    assert dst.read_text(encoding="utf-8") == before


def test_dedupe_journal_collapses_and_keeps_last_verdict(tmp_path, monkeypatch):
    """Regression coverage for `test_dedupe_journal_collapses_and_keeps_last_verdict`."""
    import csv

    lc, dst = _tmp_journal(tmp_path, monkeypatch)
    header, rows = lc.read_journal()
    base = dict(rows[0])
    aid = base["ad_id"]
    for v, c in [("fraud", "раз"), ("legit", "два"), ("", "")]:
        r = dict(base)
        r["verdict"] = v
        r["comment"] = c
        rows.append(r)
    lc.write_journal(header, rows)
    uniq = len({str(r["ad_id"]) for r in rows})
    before, after = lc.dedupe_journal()

    assert after == uniq < before
    got = [r for r in csv.DictReader(dst.open(encoding="utf-8")) if r["ad_id"] == aid]
    assert len(got) == 1
    assert (got[0]["verdict"], got[0]["comment"]) == ("legit", "два")


def test_legacy_label_cards_serve_delegates_to_unified_web(monkeypatch):
    """Regression coverage for `test_legacy_label_cards_serve_delegates_to_unified_web`."""
    import sys
    from kz.report.label_cards import __main__ as label_cli
    from kz.web import __main__ as web_cli

    called = []
    monkeypatch.setattr(sys, "argv", ["label_cards", "--serve"])
    monkeypatch.setattr(web_cli, "main", lambda: called.append(True))
    label_cli.main()
    assert called == [True]


def test_unified_verdict_endpoint_only_accepts_shown_ads(monkeypatch):
    """Regression coverage for `test_unified_verdict_endpoint_only_accepts_shown_ads`."""
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
    monkeypatch.setattr(web, "_cards_facts", {"111": {"title": "Audi 80", "year": 1994}})
    monkeypatch.setattr(label_cards, "upsert_verdict", lambda *args: saved.append(args))

    good = asyncio.run(
        web.save_verdict(Request({"ad_id": "111", "verdict": "legit", "comment": "ок"}))
    )
    bad = asyncio.run(
        web.save_verdict(Request({"ad_id": "999", "verdict": "legit", "comment": ""}))
    )
    assert good.status_code == 200 and len(saved) == 1
    assert bad.status_code == 400 and len(saved) == 1
    assert web._cards_html is None
    assert web._cards_facts == {"111": {"title": "Audi 80", "year": 1994}}


def test_file_mode_page_cannot_write_journal():
    """Regression coverage for `test_file_mode_page_cannot_write_journal`."""
    from kz.report import label_cards as lc
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "ad_id": "1",
                "brand": "Audi",
                "model": "80",
                "year": 1994,
                "price_tenge": 240000,
                "photos": [],
                "status": "active",
                "existing_verdict": None,
                "suspicion_reasons": "price_anomaly_low",
                "price_z": -4.0,
            }
        ]
    )
    assert "const SERVER = false;" in lc.build(rows, serve_mode=False)
    assert "const SERVER = true;" in lc.build(rows, serve_mode=True)


def test_code_fingerprint_survives_file_moves(tmp_path):
    """Regression coverage for `test_code_fingerprint_survives_file_moves`."""
    from kz.ml.train_price_model import code_fingerprint

    a = tmp_path / "one" / "mod.py"
    b = tmp_path / "two" / "mod.py"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("x = 1\n", encoding="utf-8")
    assert code_fingerprint(str(a)) == code_fingerprint(str(b))
    b.write_text("x = 2\n", encoding="utf-8")
    assert code_fingerprint(str(a)) != code_fingerprint(str(b))


def test_fingerprint_inputs_are_resolvable():
    """Regression coverage for `test_fingerprint_inputs_are_resolvable`."""
    from pathlib import Path
    from kz.ml import residual_detector, train_price_model
    from kz.transform import data_quality

    for m in (train_price_model, residual_detector, data_quality):
        assert Path(m.__file__).exists(), m.__name__


def test_no_flat_module_imports_left():
    """Regression coverage for `test_no_flat_module_imports_left`."""
    import re
    from pathlib import Path

    flat = {
        "db",
        "config",
        "pacing",
        "clean",
        "damage",
        "enrich",
        "parser",
        "check_status",
        "photo_dedup",
        "backfill_avgprice",
        "explore",
        "data_quality",
        "train_price_model",
        "residual_detector",
        "predict_price",
        "survival",
        "label_cards",
        "evaluate_detector",
        "catch_up",
        "run_all",
        "pipeline_status",
        "migrate_to_postgres",
        "ml_report",
        "ml_dashboard",
    }
    bad = []
    for p in list(Path("kz").rglob("*.py")) + list(Path("tests").glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"\s*(?:from (\w+) import |import (\w+)(?: as \w+)?$)", line)
            if m and (m.group(1) or m.group(2)) in flat:
                bad.append(f"{p}:{i}: {line.strip()}")
    assert not bad, "плоские импорты:\n" + "\n".join(bad)


def test_dag_commands_use_package_modules():
    """Regression coverage for `test_dag_commands_use_package_modules`."""
    import re
    from pathlib import Path

    flat = (
        "db",
        "config",
        "pacing",
        "clean",
        "enrich",
        "parser",
        "explore",
        "check_status",
        "photo_dedup",
        "backfill_avgprice",
        "label_cards",
        "catch_up",
        "run_all",
        "data_quality",
        "train_price_model",
    )
    bad = []
    for dag in Path("airflow/dags").glob("*.py"):
        text = dag.read_text(encoding="utf-8")
        for mod in flat:
            if re.search(rf"from {mod} import ", text):
                bad.append(f"{dag.name}: from {mod} import")

            if f"python {mod}.py" in text:
                bad.append(f"{dag.name}: python {mod}.py")
    assert not bad, "DAG ссылается на плоские модули:\n" + "\n".join(bad)


def test_learning_curve_subsample_keeps_groups_whole():
    """Regression coverage for `test_learning_curve_subsample_keeps_groups_whole`."""
    import pandas as pd
    from kz.ml.learning_curve import subsample_by_groups

    df = pd.DataFrame({"x": range(100)})
    groups = pd.Series([f"g{i // 4}" for i in range(100)])
    part, g = subsample_by_groups(df, groups, 0.5, seed=1)

    for name, size in g.value_counts().items():
        assert size == (groups == name).sum(), name
    assert 0 < len(part) < len(df)
    whole, gw = subsample_by_groups(df, groups, 1.0)
    assert len(whole) == len(df)


def test_ml_chain_order_respects_artifacts():
    """Regression coverage for `test_ml_chain_order_respects_artifacts`."""
    from kz.ops.run_all import ML_CHAIN

    order = [cmd[-1] for _, cmd in ML_CHAIN]
    i = {m: n for n, m in enumerate(order)}
    assert i["kz.ml.train_price_model"] < i["kz.report.ml_dashboard"]
    assert i["kz.ml.train_price_model"] < i["kz.report.ml_report"]
    assert i["kz.ml.residual_detector"] < i["kz.report.ml_report"]


def test_offline_chain_rebuilds_before_reporting():
    """Regression coverage for `test_offline_chain_rebuilds_before_reporting`."""
    from kz.ops.run_all import OFFLINE_CHAIN

    order = [cmd[-1] for _, cmd in OFFLINE_CHAIN]
    assert order == ["kz.transform.clean", "kz.report.explore", "kz.report.label_cards"]


def test_ml_and_offline_chains_never_touch_network():
    """Regression coverage for `test_ml_and_offline_chains_never_touch_network`."""
    from kz.ops.run_all import ML_CHAIN, OFFLINE_CHAIN

    net = {
        "kz.collect.parser",
        "kz.collect.check_status",
        "kz.collect.enrich",
        "kz.collect.photo_dedup",
        "kz.collect.backfill_avgprice",
    }
    for _, cmd in ML_CHAIN + OFFLINE_CHAIN:
        assert cmd[-1] not in net, cmd[-1]


def test_ml_flag_implies_offline_rebuild():
    """Regression coverage for `test_ml_flag_implies_offline_rebuild`."""
    from pathlib import Path

    src = Path("kz/ops/run_all.py").read_text(encoding="utf-8")
    assert 'fast = "--fast" in sys.argv or ml' in src


def test_default_run_uses_budgeted_collect_chain():
    """Regression coverage for `test_default_run_uses_budgeted_collect_chain`."""
    import inspect
    from kz.ops import run_all

    src = inspect.getsource(run_all.main)
    assert "if collect or (not fast and not light):" in src
    assert "for s in COLLECT_CHAIN" in src
    assert "run_parallel(STEP_ENRICH, STEP_PHOTOS)" not in src


def test_db_stats_tables_cover_pipeline_layers():
    """Regression coverage for `test_db_stats_tables_cover_pipeline_layers`."""
    from kz.ops import db_stats

    names = [t for t, _ in db_stats.TABLES]
    for must in ["raw_ads", "sightings", "photos", "enriched", "clean_data"]:
        assert must in names, must


def test_db_stats_snapshot_roundtrip(tmp_path, monkeypatch):
    """Regression coverage for `test_db_stats_snapshot_roundtrip`."""
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
    """Regression coverage for `test_db_stats_delta_formatting`."""
    from kz.ops import db_stats

    out = db_stats.format_counts(
        {"raw_ads": 4200, "sightings": 5000}, {"raw_ads": 4000, "sightings": 5000}
    )
    assert "+200" in out
    assert "no change" in out
    plain = db_stats.format_counts({"raw_ads": 4200})
    assert "+" not in plain


def test_dags_are_in_english():
    """Regression coverage for `test_dags_are_in_english`."""
    import re
    from pathlib import Path

    for dag in Path("airflow/dags").glob("*.py"):
        cyr = re.findall(r"[а-яА-ЯёЁ]+", dag.read_text(encoding="utf-8"))
        assert not cyr, f"{dag.name}: кириллица в DAG — {cyr[:5]}"


def test_web_jsonable_handles_numpy_and_nan():
    """Regression coverage for `test_web_jsonable_handles_numpy_and_nan`."""
    import numpy as np
    from kz.web.service import jsonable

    out = jsonable(
        {
            "a": np.int64(5),
            "b": np.float64(1.5),
            "c": float("nan"),
            "d": [np.int64(1), np.bool_(True)],
            "e": "текст",
            "f": None,
        }
    )
    assert out == {"a": 5, "b": 1.5, "c": None, "d": [1, True], "e": "текст", "f": None}
    assert isinstance(out["a"], int) and not isinstance(out["a"], np.integer)
    import json

    json.dumps(out)


def test_web_listing_warnings_are_evidence_based():
    """Seller warnings stay actionable without turning correlations into promises."""
    from kz.web.service import listing_warnings

    w = listing_warnings(
        {"mileage_km": None, "photos_count": 2}, asking_price=1_000_000, fair=5_000_000, text=""
    )
    joined = " ".join(w)
    assert "mileage" in joined.lower()
    assert "photos" in joined.lower()
    assert "below" in joined.lower()  # unexplained low price
    for promise in ("will sell", "one day", "guaranteed", "more views", "receive about"):
        assert promise not in joined.lower(), f"unsupported promise: {promise}"
    for unsupported_percentage in ("16%", "36%", "77%"):
        assert unsupported_percentage not in joined
    # A complete listing at a normal price should have no warnings.
    clean = listing_warnings(
        {"mileage_km": 90000, "photos_count": 9},
        asking_price=5_000_000,
        fair=5_000_000,
        text="x" * 120,
    )
    assert clean == []


def test_web_price_position_needs_enough_similar():
    """Regression coverage for `test_web_price_position_needs_enough_similar`."""
    from kz.web import service

    assert service.MIN_SIMILAR >= 8
    assert service.price_position({"brand": None}, 1_000_000) is None
    assert service.price_position({"brand": "X", "model": "Y", "age": 5}, None) is None


def test_web_app_routes_exist():
    """Regression coverage for `test_web_app_routes_exist`."""
    from kz.web.app import app

    paths = {r.path for r in app.routes}
    for p in (
        "/",
        "/estimate",
        "/api/estimate",
        "/label",
        "/verdict",
        "/damage",
        "/damage/label",
        "/price-review",
        "/price-review/label",
        "/api/health",
    ):
        assert p in paths, p


def test_web_health_reports_loaded_model_metadata(monkeypatch):
    """Regression coverage for `test_web_health_reports_loaded_model_metadata`."""
    from kz.web import app as web_app
    from kz.web import service

    meta = {
        "created_at_utc": "2026-09-02T00:00:00+00:00",
        "training_rows": 123,
        "validation": {"grouped_cv": {"model": {"mape_pct": 21.4}}},
    }
    monkeypatch.setattr(service, "get_model", lambda: (object(), meta))
    out = web_app.health()
    assert out["ok"] is True
    assert out["training_rows"] == 123
    assert out["model_mape_pct"] == 21.4


def test_web_estimate_rejects_non_numeric_input_before_model_call(monkeypatch):
    """Invalid numeric input returns 400 before calling the model."""
    import asyncio
    import json
    from kz.web import app as web_app

    class Request:
        async def json(self):
            return {"brand": "Toyota", "model": "Camry", "year": "not a year"}

    monkeypatch.setattr(
        web_app,
        "full_estimate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not be called")),
    )
    response = asyncio.run(web_app.api_estimate(Request()))
    assert response.status_code == 400
    assert "must be numeric" in json.loads(response.body)["error"]


def test_estimate_page_escapes_values_before_inner_html():
    """Regression coverage for `test_estimate_page_escapes_values_before_inner_html`."""
    from kz.web.pages import estimate_page

    html = estimate_page()
    assert "function esc(" in html
    for value in ("d.error", "s.brand", "s.model"):
        assert f"esc({value})" in html
    assert "esc(valueLabel(x.value))" in html


def test_routed_price_model_uses_specialist_only_below_threshold():
    """Regression coverage for `test_routed_price_model_uses_specialist_only_below_threshold`."""
    import numpy as np
    import pandas as pd
    from kz.ml.train_price_model import RoutedPriceModel

    class Fake:
        def __init__(self, column):
            self.column = column

        def predict(self, rows):
            return np.log(rows[self.column].to_numpy(dtype=float))

    rows = pd.DataFrame({"base": [4_000_000, 6_000_000], "special": [3_500_000, 1_000_000]})
    routed = np.exp(RoutedPriceModel(Fake("base"), Fake("special")).predict(rows))
    assert np.allclose(routed, [3_500_000, 6_000_000])


def test_routed_model_bootstrap_compares_paired_duplicate_groups():
    """Regression coverage for `test_routed_model_bootstrap_compares_paired_duplicate_groups`."""
    import numpy as np
    import pandas as pd
    from kz.ml.train_price_model import grouped_bootstrap_mape_delta

    df = pd.DataFrame(
        {
            "ad_id": ["a", "b", "c", "d"],
            "price_tenge": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "text_full": ["", "", "", ""],
        }
    )
    truth = np.log(df.price_tenge.to_numpy())
    result = grouped_bootstrap_mape_delta(df, truth, truth + np.log(2), n_boot=200)
    assert np.isclose(result["mape_delta_pct_points"], -100.0)
    assert result["bootstrap_95_ci"][1] < 0
    assert result["bootstrap_probability_better"] == 1.0


def test_mape_stability_bootstrap_is_grouped_and_deterministic():
    """Regression coverage for `test_mape_stability_bootstrap_is_grouped_and_deterministic`."""
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
    """Regression coverage for `test_mape_stability_separates_age_from_price`."""
    import pandas as pd
    from kz.ml.mape_stability import build_report

    oof = pd.DataFrame(
        {
            "duplicate_group": ["a", "b", "c", "d"],
            "age": [3, 8, 12, 25],
            "actual_price_tenge": [4e6, 6e6, 4e6, 6e6],
            "absolute_percentage_error_pct": [40.0, 10.0, 30.0, 20.0],
        }
    )
    report, rows = build_report(oof, n_boot=50)
    assert len(report["by_age"]) == 4
    assert set(rows[rows.segment_type == "price"].segment) == {"<5M", "5M+"}
    assert set(rows[rows.segment_type == "age_x_price"].segment) == {
        "0-5 | <5M",
        "6-10 | 5M+",
        "11-20 | <5M",
        "21+ | 5M+",
    }


def test_oof_diagnostics_are_local_minimal_and_atomic(tmp_path, monkeypatch):
    """Regression coverage for `test_oof_diagnostics_are_local_minimal_and_atomic`."""
    import numpy as np
    import pandas as pd
    from kz.ml import train_price_model as model

    target = tmp_path / "oof.csv"
    monkeypatch.setattr(model, "OOF_DIAGNOSTICS_PATH", target)
    df = pd.DataFrame(
        {
            "ad_id": ["a", "b"],
            "text_full": ["достаточно длинный текст", "другой длинный текст"],
            "brand": ["X", "Y"],
            "model": ["A", "B"],
            "year": [2020, 2010],
            "age": [7, 17],
            "price_tenge": [1_000_000, 2_000_000],
        }
    )
    truth = np.log(df["price_tenge"].to_numpy())
    model.save_oof_diagnostics(df, truth, truth, truth)
    saved = pd.read_csv(target)
    assert len(saved) == 2
    assert "ad_id" not in saved
    assert "text_full" not in saved
    assert "actual_price_tenge" in saved
    assert np.allclose(saved["absolute_percentage_error_pct"], 0)


def test_temporal_metrics_include_routed_vs_base_uncertainty():
    """Regression coverage for `test_temporal_metrics_include_routed_vs_base_uncertainty`."""
    import inspect
    from kz.ml import train_price_model

    src = inspect.getsource(train_price_model.evaluate_temporal)
    assert "grouped_bootstrap_mape_delta" in src
    assert '"routed_vs_base": comparison' in src


def test_web_binds_localhost_only():
    """Regression coverage for `test_web_binds_localhost_only`."""
    from pathlib import Path
    from kz.web.__main__ import HOST

    assert HOST == "127.0.0.1"
    src = Path("kz/web/__main__.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in src


def test_photo_fetch_retries_transient_failures_only():
    """Regression coverage for `test_photo_fetch_retries_transient_failures_only`."""
    from kz.collect.photo_fetch import PERMANENT_STATUSES

    assert PERMANENT_STATUSES == {200, 404, 410}
    for transient in (-1, 500, 502, 503, 429):
        assert transient not in PERMANENT_STATUSES


def test_photo_fetch_skips_unresolvable_hosts():
    """Regression coverage for `test_photo_fetch_skips_unresolvable_hosts`."""
    from kz.collect.photo_fetch import live_hosts

    urls = ["https://example.invalid/a.jpg", "https://localhost/b.jpg"]
    alive = live_hosts(urls)
    assert "example.invalid" not in alive


def test_photo_dedup_skips_unresolvable_hosts():
    """Regression coverage for `test_photo_dedup_skips_unresolvable_hosts`."""
    from kz.collect.photo_dedup import live_hosts

    alive = live_hosts(["https://example.invalid/a.jpg", "https://localhost/b.jpg"])
    assert "example.invalid" not in alive


def test_photo_fetch_path_layout_shards_by_ad_id():
    """Regression coverage for `test_photo_fetch_path_layout_shards_by_ad_id`."""
    from kz.collect.photo_fetch import local_path

    p = local_path("225678236", 1)
    assert p.parts[-2] == "22"
    assert p.name == "225678236_1.jpg"


def test_photo_fetch_uses_cdn_budget_not_kolesa():
    """Regression coverage for `test_photo_fetch_uses_cdn_budget_not_kolesa`."""
    from pathlib import Path

    src = Path("kz/collect/photo_fetch.py").read_text(encoding="utf-8")
    assert 'charge_budget("cdn"' in src
    assert 'charge_budget("kolesa"' not in src


def test_photo_embedding_is_reduced_before_modelling():
    """Regression coverage for `test_photo_embedding_is_reduced_before_modelling`."""
    from kz.ml import photo_features

    assert photo_features.N_COMPONENTS <= 64
    import numpy as np

    emb = np.random.RandomState(0).rand(120, 2048)
    out = photo_features.reduce_embeddings(emb)
    assert out.shape[0] == 120
    assert out.shape[1] <= photo_features.N_COMPONENTS
    assert out.shape[1] < emb.shape[1]


def test_photo_quality_metrics_detect_blur():
    """Regression coverage for `test_photo_quality_metrics_detect_blur`."""
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
        sharp.save(a, quality=95)
        blurred.save(b, quality=95)
        qa, qb = quality_metrics(str(a)), quality_metrics(str(b))
    assert qa["img_sharpness"] > qb["img_sharpness"]
    assert qa["img_pixels"] == 256 * 256


def test_photo_ablation_compares_on_same_rows():
    """Regression coverage for `test_photo_ablation_compares_on_same_rows`."""
    from pathlib import Path

    src = Path("kz/ml/photo_ablation.py").read_text(encoding="utf-8")
    assert 'how="inner"' in src
    assert "GroupKFold" in src


def test_survival_uses_posted_date_not_first_sighting():
    """Regression coverage for `test_survival_uses_posted_date_not_first_sighting`."""
    from pathlib import Path

    src = Path("kz/ml/survival.py").read_text(encoding="utf-8")
    assert "parse_posted" in src
    assert 'd["start"]' in src


def test_survival_respects_events_per_feature_rule():
    """Regression coverage for `test_survival_respects_events_per_feature_rule`."""
    from kz.ml import survival

    assert survival.MIN_EVENTS_PER_FEATURE >= 10


def test_survival_horizon_is_bounded():
    """Regression coverage for `test_survival_horizon_is_bounded`."""
    from kz.ml import survival

    assert 7 <= survival.HORIZON <= 60


def test_psi_zero_for_identical_distributions():
    """Regression coverage for `test_psi_zero_for_identical_distributions`."""
    import numpy as np
    from kz.ml.monitoring import psi

    x = np.random.RandomState(0).normal(size=5000)
    assert abs(psi(x, x.copy())) < 1e-6


def test_psi_grows_with_shift():
    """Regression coverage for `test_psi_grows_with_shift`."""
    import numpy as np
    from kz.ml.monitoring import psi

    rng = np.random.RandomState(0)
    base = rng.normal(size=5000)
    small = psi(base, rng.normal(loc=0.2, size=5000))
    big = psi(base, rng.normal(loc=1.5, size=5000))
    assert 0 <= small < big
    assert big > 0.25


def test_psi_survives_empty_bins():
    """Regression coverage for `test_psi_survives_empty_bins`."""
    import numpy as np
    from kz.ml.monitoring import psi

    base = np.arange(1000, dtype=float)
    shifted = np.arange(500, 1000, dtype=float)
    v = psi(base, shifted)
    assert np.isfinite(v) and v > 0


def test_psi_thresholds_are_the_standard_ones():
    """Regression coverage for `test_psi_thresholds_are_the_standard_ones`."""
    from kz.ml import monitoring

    assert monitoring.PSI_WATCH == 0.10
    assert monitoring.PSI_ALERT == 0.25
    assert monitoring.level(0.05) == "stable"
    assert "noticeable" in monitoring.level(0.15)
    assert "MAJOR" in monitoring.level(0.4)


def test_categorical_psi_detects_new_categories():
    """Regression coverage for `test_categorical_psi_detects_new_categories`."""
    import pandas as pd
    from kz.ml.monitoring import categorical_psi

    old = pd.Series(["Toyota"] * 80 + ["Lada"] * 20)
    same = categorical_psi(old, old.copy())
    new = categorical_psi(old, pd.Series(["BYD"] * 60 + ["Toyota"] * 40))
    assert abs(same) < 1e-6
    assert new > 0.25


def test_label_cards_show_which_stratum_each_ad_is_from():
    """The interface explains why each sampling stratum is being reviewed."""
    src = _label_cards_source()
    assert "random_control" in src and "rule_positive" in src
    assert "residual_candidate" in src
    # Controls should make it clear that a legit result is normal.
    assert "Most controls should be legit" in src

    assert "recall" in src or "полнот" in src


def test_residual_detector_respects_exculpation():
    """Regression coverage for `test_residual_detector_respects_exculpation`."""
    from pathlib import Path

    src = Path("kz/ml/residual_detector.py").read_text(encoding="utf-8")
    assert "low_price_explained" in src
    assert "~explained" in src


def test_web_labelling_includes_control_group():
    """Regression coverage for `test_web_labelling_includes_control_group`."""
    from pathlib import Path

    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    assert "include_queue=True" in src


def test_full_verdict_queue_is_the_default():
    """Regression coverage for `test_full_verdict_queue_is_the_default`."""
    import inspect
    from kz.report.label_cards.queue import load_rows

    assert inspect.signature(load_rows).parameters["include_queue"].default is True


def test_verdict_queue_keeps_history_reopenable(tmp_path, monkeypatch):
    """Reviewed rows remain reachable after the disposable queue changes."""
    import pandas as pd
    from kz.report.label_cards import queue as label_queue

    queue_csv = tmp_path / "queue.csv"
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame([{"ad_id": "new", "sampling_stratum": "random_control"}]).to_csv(
        queue_csv, index=False
    )
    pd.DataFrame(
        [
            {
                "ad_id": "old",
                "verdict": "fraud",
                "comment": "accidental click",
                "sampling_stratum": "residual_candidate",
            }
        ]
    ).to_csv(labels_csv, index=False)

    clean = pd.DataFrame(
        [
            {"ad_id": "new", "is_suspicious": 0, "price_z": 0.0},
            {"ad_id": "old", "is_suspicious": 0, "price_z": -1.0},
        ]
    )
    enriched = pd.DataFrame(
        columns=["ad_id", "options_text", "page_condition", "has_vin", "fetched_at"]
    )
    photos = pd.DataFrame(columns=["ad_id", "position", "url"])

    def fake_read_sql(query, _engine, **_kwargs):
        if "SELECT * FROM clean_data" in query:
            return clean.copy()
        if "FROM enriched" in query:
            return enriched.copy()
        if "FROM photos" in query:
            return photos.copy()
        raise AssertionError(query)

    monkeypatch.setattr(label_queue, "QUEUE_CSV", str(queue_csv))
    monkeypatch.setattr(label_queue, "LABELS_CSV", str(labels_csv))
    monkeypatch.setattr(label_queue, "get_engine", lambda: object())
    monkeypatch.setattr(label_queue.pd, "read_sql", fake_read_sql)

    rows = label_queue.load_rows()
    assert set(rows["ad_id"]) == {"new", "old"}
    old = rows.set_index("ad_id").loc["old"]
    assert old["existing_verdict"] == "fraud"
    assert old["existing_comment"] == "accidental click"
    assert old["stratum"] == "residual_candidate"


def test_verdict_page_explains_why_queue_counts_differ():
    """Regression coverage for `test_verdict_page_explains_why_queue_counts_differ`."""
    import pandas as pd
    from kz.report.label_cards import build

    base = {
        "brand": "Audi",
        "model": "80",
        "year": 1994,
        "price_tenge": 1_000_000,
        "photos": [],
        "status": "active",
        "existing_verdict": None,
        "suspicion_reasons": "",
        "price_z": 0.0,
    }
    rows = pd.DataFrame(
        [
            {**base, "ad_id": "1", "stratum": "rule_positive"},
            {**base, "ad_id": "2", "stratum": "residual_candidate"},
            {**base, "ad_id": "3", "stratum": "random_control"},
        ]
    )
    page = build(rows, serve_mode=True, journal_total=7)
    for text in (
        "3 listings",
        "1 were flagged by rules",
        "1 came from the residual detector",
        "1 were sampled",
        "3 have not\nbeen reviewed yet",
    ):
        assert text in page


def test_unknown_verdict_is_reviewed_not_unlabelled():
    """Unknown stays revisitable but must not inflate the untouched count."""
    import pandas as pd
    from kz.report.label_cards import build

    rows = pd.DataFrame(
        [
            {
                "ad_id": "1",
                "brand": "Audi",
                "model": "80",
                "year": 1994,
                "price_tenge": 1_000_000,
                "photos": [],
                "status": "active",
                "existing_verdict": "unknown",
                "existing_comment": "insufficient evidence",
                "suspicion_reasons": "",
                "price_z": 0.0,
                "stratum": "residual_candidate",
            }
        ]
    )
    page = build(rows, serve_mode=True, journal_total=1)
    assert "1 are marked unknown pending more evidence" in page
    assert "0 have not\nbeen reviewed yet" in page


def test_label_cards_can_filter_control_group():
    """Regression coverage for `test_label_cards_can_filter_control_group`."""
    src = _label_cards_source()
    assert 'data-stratum="{st}"' in src
    assert "only-control" in src
    assert 'not([data-stratum="random_control"])' in src


def test_label_cards_can_reopen_each_saved_verdict_with_one_click():
    """Fraud, legit, and unknown history have dedicated review tabs."""
    src = _label_cards_source()
    for verdict in ("fraud", "legit", "unknown"):
        assert f'data-verdict-filter="{verdict}"' in src
        assert f'data-verdict="{verdict}"' in src
    assert "document.body.dataset.verdictFilter" in src
    assert "data-existing-verdict" in src


def test_label_cards_keyboard_navigation_respects_active_filters():
    """Hidden cards must never receive a keyboard verdict."""
    src = _label_cards_source()
    assert "function visibleCards()" in src
    assert "function stepVisible(direction)" in src
    assert "stepVisible(1)" in src
    assert "stepVisible(-1)" in src
    assert "getComputedStyle(cards[cur]).display !== 'none'" in src
    assert src.count("focusFirstVisible();") >= 2


def test_photo_src_prefers_local_and_drops_dead_hosts():
    """Regression coverage for `test_photo_src_prefers_local_and_drops_dead_hosts`."""
    from kz.report.label_cards import DEAD_HOSTS, photo_src

    dead = f"https://{sorted(DEAD_HOSTS)[0]}/webp/aa/x.jpg"
    live = "https://alaps-photos-kl.kcdn.kz/webp/bb/y.jpg"
    assert photo_src("999999999", 1, dead, False) is None
    assert photo_src("999999999", 1, live, False) == live


def test_photo_route_blocks_directory_traversal():
    """Regression coverage for `test_photo_route_blocks_directory_traversal`."""
    from pathlib import Path

    sources = {"kz/web/app.py": Path("kz/web/app.py").read_text(encoding="utf-8")}
    for where, src in sources.items():
        assert ".resolve()" in src and "parents" in src, where


def test_basket_hint_matches_the_mode():
    """Regression coverage for `test_basket_hint_matches_the_mode`."""
    src = _label_cards_source()
    assert "baskethint" in src
    assert "SERVER\n  ?" in src or "SERVER ?" in src


def test_counter_reflects_journal_not_just_draft():
    """Regression coverage for `test_counter_reflects_journal_not_just_draft`."""
    src = _label_cards_source()
    assert "ALREADY" in src
    assert "ALREADY.has(card.dataset.id)" in src
    assert "const scope = visibleCards()" in src


def test_web_coerces_numeric_fields_from_forms():
    """Numeric form strings and invalid public inputs are handled safely."""
    from kz.web.service import listing_warnings

    # Numeric strings must not break listing checks.
    assert (
        listing_warnings(
            {"mileage_km": "95000", "photos_count": "8"}, 11_000_000, 12_000_000, "x" * 120
        )
        == []
    )
    # Invalid values are also tolerated because callers may bypass the form.
    w = listing_warnings(
        {"mileage_km": "invalid", "photos_count": None}, 11_000_000, 12_000_000, ""
    )
    assert any("mileage" in x.lower() for x in w)


def test_web_converts_every_numeric_feature_not_a_hand_list():
    """Regression coverage for `test_web_converts_every_numeric_feature_not_a_hand_list`."""
    from pathlib import Path

    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    assert "NUM_FEATURES" in src
    assert "for k in list(NUM_FEATURES)" in src


def test_journal_stores_sampling_stratum():
    """Regression coverage for `test_journal_stores_sampling_stratum`."""
    from kz.report.label_cards import BASE_HEADER, STRATUM_COLS, journal_header

    h = journal_header()
    for c in STRATUM_COLS:
        assert c in h, c
    for c in BASE_HEADER:
        assert c in h, c


def test_zero_fraud_is_reported_as_a_bound_not_a_blank():
    """Regression coverage for `test_zero_fraud_is_reported_as_a_bound_not_a_blank`."""
    from pathlib import Path

    src = Path("kz/report/evaluate_detector.py").read_text(encoding="utf-8")
    assert "control_bound_report" in src
    assert "3 / n_ctrl" in src or "3/n_ctrl" in src


def test_missing_db_settings_do_not_break_import():
    """Regression coverage for `test_missing_db_settings_do_not_break_import`."""
    import importlib
    import os
    import kz.core.config as config

    saved = {
        k: os.environ.pop(k, None) for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    }
    try:
        reloaded = importlib.reload(config)

        if reloaded.POSTGRES_USER is None:
            assert reloaded.DATABASE_URL is None
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(config)


def test_engine_without_settings_explains_itself():
    """Regression coverage for `test_engine_without_settings_explains_itself`."""
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
    """Regression coverage for `test_web_query_survives_dead_database`."""
    import kz.web.service as service
    import kz.core.db as db

    real = db.DATABASE_URL
    db.get_engine.cache_clear()
    db.DATABASE_URL = None
    service._db_warned = False
    try:
        assert service.query("SELECT 1", {}) is None
        assert service.similar_cars({"brand": "X", "model": "Y", "age": 5}).empty
        assert service.price_position({"brand": "X", "model": "Y", "age": 5}, 5_000_000) is None
    finally:
        db.DATABASE_URL = real
        db.get_engine.cache_clear()
        service._db_warned = False


def test_public_demo_closes_the_labelling_journal():
    """Regression coverage for `test_public_demo_closes_the_labelling_journal`."""
    from pathlib import Path

    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    assert "KZ_PUBLIC_DEMO" in src

    assert src.count("if PUBLIC_DEMO:") >= 2


def test_image_carries_model_but_not_collected_ads():
    """The image carries derivative model weights but never source listings."""
    from pathlib import Path

    docker = Path("Dockerfile").read_text(encoding="utf-8")
    assert "price_model.cbm" in docker
    for forbidden in ("COPY data/raw", "COPY data/clean", "COPY data/ ", "COPY . "):
        assert forbidden not in docker, forbidden
    ignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert "data/" in ignore and ".env" in ignore


def test_web_image_skips_playwright():
    """Regression coverage for `test_web_image_skips_playwright`."""
    from pathlib import Path

    lines = [
        l.split("#")[0].strip().lower()
        for l in Path("requirements-web.txt").read_text(encoding="utf-8").splitlines()
    ]
    pkgs = [l for l in lines if l]
    assert not any("playwright" in p for p in pkgs)
    assert any("fastapi" in p for p in pkgs)
    assert any("catboost" in p for p in pkgs)


def test_no_module_advertises_a_command_that_no_longer_works():
    """Regression coverage for `test_no_module_advertises_a_command_that_no_longer_works`."""
    import re
    from pathlib import Path

    names = {p.name for p in Path("kz").rglob("*.py") if "__pycache__" not in str(p)}
    bad = []
    for p in Path("kz").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"python ([a-z_]+)\.py", line):
                if m.group(1) + ".py" in names:
                    bad.append(f"{p}:{i}: {m.group(0)}")
    assert not bad, "подсказка зовёт файл вместо модуля:\n" + "\n".join(bad)


class _FakeRandom:
    """Implementation of `_FakeRandom`."""

    def __init__(self, randoms=(), uniforms=()):
        self._r, self._u = list(randoms), list(uniforms)
        self.calls = []

    def random(self):
        return self._r.pop(0) if self._r else 0.99

    def uniform(self, a, b):
        self.calls.append((a, b))
        return self._u.pop(0) if self._u else (a + b) / 2


def test_pause_never_shortens_below_the_floor():
    """Regression coverage for `test_pause_never_shortens_below_the_floor`."""
    from kz.core.pacing import human_pause, LONG_TAIL_MULT

    lo, hi = 4.0, 8.0

    assert human_pause(lo, hi, _FakeRandom(randoms=[0.99])) == (lo + hi) / 2

    rng = _FakeRandom(randoms=[0.0])
    assert human_pause(lo, hi, rng) >= hi
    assert rng.calls == [(hi, hi * LONG_TAIL_MULT)]


def test_long_break_fires_on_schedule_and_not_at_zero():
    """Regression coverage for `test_long_break_fires_on_schedule_and_not_at_zero`."""
    from kz.core.pacing import long_break, BREAK_RANGE

    rng = _FakeRandom()
    assert long_break(0, every=15, rng=rng) is None
    assert long_break(14, every=15, rng=rng) is None
    got = long_break(15, every=15, rng=rng)
    assert got is not None and BREAK_RANGE[0] <= got <= BREAK_RANGE[1]
    assert long_break(30, every=15, rng=rng) is not None
    assert long_break(7, every=0, rng=rng) is None


def test_mean_pause_is_strictly_slower_than_flat_uniform():
    """Regression coverage for `test_mean_pause_is_strictly_slower_than_flat_uniform`."""
    from kz.core.pacing import mean_pause

    lo, hi = 4.0, 8.0
    flat = (lo + hi) / 2
    assert mean_pause(lo, hi) > flat

    assert flat < mean_pause(lo, hi, every=120) < mean_pause(lo, hi, every=15)


def test_polite_sleep_prefers_the_break_over_the_short_pause():
    """Regression coverage for `test_polite_sleep_prefers_the_break_over_the_short_pause`."""
    import kz.core.pacing as pacing

    slept = []
    real_sleep = pacing.time.sleep
    pacing.time.sleep = slept.append
    try:
        brk = pacing.polite_sleep(15, (4.0, 8.0), rng=_FakeRandom(), break_every=15)
        assert brk >= pacing.BREAK_RANGE[0] and slept == [brk]
        slept.clear()
        short = pacing.polite_sleep(3, (4.0, 8.0), rng=_FakeRandom(randoms=[0.99]), break_every=15)
        assert short == 6.0 and slept == [6.0]
    finally:
        pacing.time.sleep = real_sleep


def test_pinned_versions_match_the_python_ci_actually_runs():
    """Regression coverage for `test_pinned_versions_match_the_python_ci_actually_runs`."""
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

    assert ci_ver == lint_ver, f"CI гоняет {ci_ver}, линтер целится в {lint_ver}"
    assert ci_ver == sys.version_info[:2], (
        f"CI гоняет {ci_ver}, а версии в requirements сняты с "
        f"{sys.version_info[:2]} — пины будут неустановимы"
    )


def test_every_import_is_declared_in_some_requirements_file():
    """Regression coverage for `test_every_import_is_declared_in_some_requirements_file`."""
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

    alias = {
        "sklearn": "scikit-learn",
        "PIL": "pillow",
        "dotenv": "python-dotenv",
        "psycopg2": "psycopg2-binary",
        "open_clip": "open_clip_torch",
        "bs4": "beautifulsoup4",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
    }
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
        f"  {pkg} ← {', '.join(sorted(files))}" for pkg, files in sorted(missing.items())
    )


def test_dag_docstring_lists_every_task_it_defines():
    """Regression coverage for `test_dag_docstring_lists_every_task_it_defines`."""
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
    """Regression coverage for `test_enrichment_queue_puts_fresh_ads_ahead_of_stale_backlog`."""
    from kz.collect import enrich

    rows = pd.DataFrame(
        [
            {
                "ad_id": "old_plain",
                "is_suspicious": 0,
                "scraped_at": "2026-07-17",
                "price_tenge": 9_000_000,
            },
            {
                "ad_id": "new_plain",
                "is_suspicious": 0,
                "scraped_at": "2026-08-10",
                "price_tenge": 9_000_000,
            },
            {
                "ad_id": "old_susp",
                "is_suspicious": 1,
                "scraped_at": "2026-07-17",
                "price_tenge": 9_000_000,
            },
            {
                "ad_id": "mid_plain",
                "is_suspicious": 0,
                "scraped_at": "2026-08-01",
                "price_tenge": 9_000_000,
            },
        ]
    )
    monkeypatch.setattr(enrich.pd, "read_sql", lambda *a, **k: rows.copy())
    monkeypatch.setattr(enrich, "get_engine", lambda: None)

    got = enrich.pick_targets(set())
    assert got[0] == "old_susp", "подозрительное вперёд, несмотря на возраст"
    assert got[1] == "new_plain", "дальше новейшее — его страница ещё жива"
    assert got.index("mid_plain") < got.index("old_plain")


def test_enrichment_queue_prefers_the_cheap_segment(monkeypatch):
    """Regression coverage for `test_enrichment_queue_prefers_the_cheap_segment`."""
    from kz.collect import enrich

    rows = pd.DataFrame(
        [
            {
                "ad_id": "susp",
                "is_suspicious": 1,
                "scraped_at": "2026-07-01",
                "price_tenge": 20_000_000,
            }
        ]
        + [
            {
                "ad_id": f"fresh_rich_{i}",
                "is_suspicious": 0,
                "scraped_at": "2026-08-20",
                "price_tenge": 30_000_000,
            }
            for i in range(10)
        ]
        + [
            {
                "ad_id": f"stale_cheap_{i}",
                "is_suspicious": 0,
                "scraped_at": "2026-07-05",
                "price_tenge": 1_000_000,
            }
            for i in range(30)
        ]
    )
    monkeypatch.setattr(enrich.pd, "read_sql", lambda *a, **k: rows.copy())
    monkeypatch.setattr(enrich, "get_engine", lambda: None)

    got = enrich.pick_targets(set())
    assert got[0] == "susp", "подозрительное по-прежнему первое"

    cheap = sum(a.startswith("stale_cheap") for a in got)
    rich = sum(a.startswith("fresh_rich") for a in got)
    assert cheap > rich, f"дешёвых должно быть больше: дешёвых {cheap}, дорогих {rich}"
    assert rich >= 2, (
        "часть прогона обязана уходить свежим независимо от цены, иначе "
        "пропущенность признаков совпадёт с сегментом и станет утечкой"
    )


def test_drift_check_runs_before_retraining():
    """Regression coverage for `test_drift_check_runs_before_retraining`."""
    from kz.ops.run_all import ML_CHAIN

    order = [cmd[-1] for _, cmd in ML_CHAIN]
    assert order.index("kz.ml.monitoring") < order.index("kz.ml.train_price_model")


def test_drift_refuses_to_report_stability_it_did_not_measure():
    """Regression coverage for `test_drift_refuses_to_report_stability_it_did_not_measure`."""
    import ast
    import inspect
    from kz.ml import monitoring

    body = inspect.getsource(monitoring.main)
    assert "if fresh_rows <= 0:" in body

    assert body.index("if fresh_rows <= 0:") < body.index("append_history(")

    guard = next(
        n
        for n in ast.walk(ast.parse(body.lstrip()))
        if isinstance(n, ast.If) and "fresh_rows" in ast.dump(n.test)
    )
    assert any(isinstance(x, ast.Return) for x in guard.body)


def _fresh(**kw):
    from datetime import date, timedelta
    from kz.core.freshness import Freshness

    today = date.today()
    base = dict(
        last_collect=today,
        collect_days=30,
        span_days=30,
        last_status_check=today,
        ads_total=1000,
        ads_status_checked=900,
        model_created=today,
    )
    base.update(
        {
            k: (today - timedelta(days=v) if k.endswith("_ago") else v)
            for k, v in kw.items()
            if not k.endswith("_ago")
        }
    )
    for k, v in kw.items():
        if k.endswith("_ago"):
            base[k[:-4]] = today - timedelta(days=v)
    return Freshness(**base)


def test_stale_statuses_are_called_out_not_swallowed():
    """Regression coverage for `test_stale_statuses_are_called_out_not_swallowed`."""
    from kz.core.freshness import stale_warnings

    assert stale_warnings(_fresh()) == []
    warned = stale_warnings(_fresh(last_status_check_ago=15))
    assert any("Statuses were last checked 15" in w for w in warned)


def test_default_active_status_is_flagged_as_a_guess():
    """Regression coverage for `test_default_active_status_is_flagged_as_a_guess`."""
    from kz.core.freshness import stale_warnings

    warned = stale_warnings(_fresh(ads_status_checked=220, ads_total=1000))
    assert any("22%" in w for w in warned)


def test_collection_gaps_are_reported():
    """Regression coverage for `test_collection_gaps_are_reported`."""
    from kz.core.freshness import stale_warnings

    assert any("Collection ran" in w for w in stale_warnings(_fresh(collect_days=8, span_days=39)))


def test_model_freshness_converts_utc_to_almaty_calendar_day():
    """Regression coverage for `test_model_freshness_converts_utc_to_almaty_calendar_day`."""
    from datetime import date
    from kz.core.freshness import local_date_from_utc_iso

    assert local_date_from_utc_iso("2026-08-31T20:15:44+00:00") == date(2026, 9, 1)


def test_estimate_form_covers_the_categories_in_the_data():
    """Regression coverage for `test_estimate_form_covers_the_categories_in_the_data`."""
    import json

    from kz.ml.train_price_model import METADATA_PATH

    if not METADATA_PATH.exists():
        return
    vocab = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("categorical_vocabulary", {})
    if not vocab:
        return

    from kz.web.pages import estimate_page

    html = estimate_page()
    missing = {f: [v for v in vals if f'value="{v}"' not in html] for f, vals in vocab.items()}
    missing = {f: v for f, v in missing.items() if v}
    assert not missing, f"form is missing trained category values: {missing}"


def test_conformal_offset_widens_until_coverage_is_reached():
    """Regression coverage for `test_conformal_offset_widens_until_coverage_is_reached`."""
    import numpy as np
    from kz.ml.price_interval import conformal_offset, conformity

    y = np.zeros(100)
    lo, hi = np.full(100, -1.0), np.full(100, 1.0)
    assert conformity(y, lo, hi).max() <= 0
    assert conformal_offset(conformity(y, lo, hi), 0.8) < 0

    y2 = np.concatenate([np.zeros(50), np.full(50, 6.0)])
    off = conformal_offset(conformity(y2, lo, hi), 0.8)
    assert off > 0, "границы обязаны раздвинуться, когда факты вылезают"


def test_interval_quantile_levels_are_symmetric():
    """Regression coverage for `test_interval_quantile_levels_are_symmetric`."""
    from kz.ml.price_interval import quantile_levels

    for target, want_lo, want_hi in [(0.80, 0.10, 0.90), (0.50, 0.25, 0.75)]:
        lo, hi = quantile_levels(target)
        assert abs(lo - want_lo) < 1e-9 and abs(hi - want_hi) < 1e-9
        assert abs((hi - lo) - target) < 1e-9


def test_tails_are_calibrated_separately():
    """Regression coverage for `test_tails_are_calibrated_separately`."""
    import numpy as np
    from kz.ml.price_interval import tail_offsets

    n = 1000
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, n)

    lo = np.full(n, -10.0)
    hi = np.full(n, -2.0)
    d_lo, d_hi = tail_offsets(y, lo, hi, 0.80)
    assert d_hi > 1.0, "верхнюю границу обязаны заметно поднять"
    assert d_lo < d_hi, "нижнюю трогать почти не надо — снизу никто не вылез"


def test_groups_are_keyed_on_prediction_not_on_truth():
    """Regression coverage for `test_groups_are_keyed_on_prediction_not_on_truth`."""
    import inspect
    import numpy as np
    from kz.ml.price_interval import apply_offsets, group_of

    assert list(group_of(np.array([1e6, 7e6, 15e6, 50e6]))) == [0, 1, 2, 3]

    src = inspect.getsource(apply_offsets)
    assert "lo_log + hi_log" in src, "группа берётся из сырых границ прогноза"
    assert "price_tenge" not in src, "фактическая цена в выдаче недоступна"


def test_group_offsets_fall_back_when_a_group_is_too_small():
    """Regression coverage for `test_group_offsets_fall_back_when_a_group_is_too_small`."""
    import numpy as np
    from kz.ml.price_interval import MIN_GROUP, group_offsets

    n = 400
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.3, n)
    lo, hi = y - 0.5, y + 0.5

    pred = np.full(n, 1e6)
    off = group_offsets(y, lo, hi, pred)
    assert off["groups"]["<5M"]["source"] == "group-specific"
    assert off["groups"]["<5M"]["n"] >= MIN_GROUP
    for empty in ("5-10M", "10-20M", "20M+"):
        assert off["groups"][empty]["source"].startswith("global")
        assert off["groups"][empty]["offsets"] == off["global"]


def test_coverage_is_reported_in_both_cuts():
    """Regression coverage for `test_coverage_is_reported_in_both_cuts`."""
    import inspect
    from kz.ml import price_interval

    src = inspect.getsource(price_interval.by_segment)
    assert "by predicted price" in src and "by actual price" in src
    assert "unavoidable" in src


def test_coverage_is_reported_with_width():
    """Regression coverage for `test_coverage_is_reported_with_width`."""
    import inspect
    from kz.ml import price_interval

    src = inspect.getsource(price_interval.coverage_report)
    assert "median_width_pct" in src and "coverage" in src


def test_service_range_uses_measured_interval_not_a_fixed_corridor():
    """Regression coverage for `test_service_range_uses_measured_interval_not_a_fixed_corridor`."""
    import inspect
    from kz.web import service

    src = inspect.getsource(service.full_estimate)
    assert "price_range(car, fair)" in src
    assert "FALLBACK_LOW" not in src, "резерв не должен быть основным путём"
    assert "conformal" in inspect.getsource(service.price_range)


def test_interval_step_runs_after_training():
    """Regression coverage for `test_interval_step_runs_after_training`."""
    from kz.ops.run_all import ML_CHAIN

    order = [cmd[-1] for _, cmd in ML_CHAIN]
    assert "kz.ml.price_interval" in order
    assert order.index("kz.ml.train_price_model") < order.index("kz.ml.price_interval")
    assert order.index("kz.ml.price_interval") < order.index("kz.report.ml_report")


def test_survival_reports_a_bracket_not_a_single_number():
    """Regression coverage for `test_survival_reports_a_bracket_not_a_single_number`."""
    import inspect
    from kz.ml import survival

    src = inspect.getsource(survival.verified_bracket)
    assert "first bound is low" in src and "second is high" in src
    assert "verified_bracket(d)" in inspect.getsource(survival.main)


def test_label_cards_package_keeps_its_public_names():
    """Regression coverage for `test_label_cards_package_keeps_its_public_names`."""
    from kz.report import label_cards as lc

    for name in (
        "build",
        "load_rows",
        "upsert_verdict",
        "journal_facts",
        "dedupe_journal",
        "read_journal",
        "LABELS_CSV",
        "BASE_HEADER",
        "STRATUM_COLS",
        "FLAG_HELP",
    ):
        assert hasattr(lc, name), name


def test_label_cards_modules_stay_in_their_lanes():
    """Regression coverage for `test_label_cards_modules_stay_in_their_lanes`."""
    from pathlib import Path

    render = Path("kz/report/label_cards/render.py").read_text(encoding="utf-8")
    queue = Path("kz/report/label_cards/queue.py").read_text(encoding="utf-8")
    assert "get_engine" not in render, "render не должен ходить в базу"
    assert "read_sql" not in render
    assert "write_text" not in queue and "upsert_verdict" not in queue, "queue только читает"


def test_photo_advice_says_nothing_about_dents():
    """Regression coverage for `test_photo_advice_says_nothing_about_dents`."""
    import inspect
    from kz.ml import photo_advice

    src = inspect.getsource(photo_advice.advise)
    assert "clip_rusty" in src and "clip_dirty" in src
    assert "clip_damaged" not in src, "про повреждения советовать нечем"


def test_photo_advice_thresholds_come_from_the_corpus():
    """Regression coverage for `test_photo_advice_thresholds_come_from_the_corpus`."""
    import numpy as np
    import pandas as pd
    from kz.ml.photo_advice import thresholds

    df = pd.DataFrame(
        {"img_brightness": np.linspace(0, 100, 200), "clip_dirty": np.linspace(-1, 1, 200)}
    )
    cuts = thresholds(df, ["img_brightness", "clip_dirty"], worse_than=0.20)

    assert 15 < cuts["img_brightness"] < 25

    assert 0.5 < cuts["clip_dirty"] < 0.7


def test_photo_advice_does_not_promise_more_views():
    """Regression coverage for `test_photo_advice_does_not_promise_more_views`."""
    import inspect
    from kz.ml import photo_advice

    shown = inspect.getsource(photo_advice.advise)
    for promise in ("чаще смотреть", "больше просмотров", "быстрее продад", "продадите"):
        assert promise not in shown, promise

    assert "FAILED" in inspect.getsource(photo_advice.validate)


def test_photo_redundancy_check_is_out_of_fold():
    """Regression coverage for `test_photo_redundancy_check_is_out_of_fold`."""
    import inspect
    from kz.ml import photo_clip

    src = inspect.getsource(photo_clip._oof_logistic_auc)
    assert "cross_val_predict" in src
    assert "StratifiedKFold" in src


def test_photo_stats_count_independent_ads(tmp_path, monkeypatch):
    """Regression coverage for `test_photo_stats_count_independent_ads`."""
    from kz.report import photo_labels

    monkeypatch.setattr(photo_labels, "LABELS_CSV", str(tmp_path / "labels.csv"))
    photo_labels.write_journal(
        photo_labels.HEADER,
        [
            {"ad_id": "a", "position": "1", "label": "damaged"},
            {"ad_id": "a", "position": "2", "label": "damaged"},
            {"ad_id": "b", "position": "1", "label": "wreck"},
            {"ad_id": "c", "position": "1", "label": "intact"},
        ],
    )
    s = photo_labels.stats()
    assert s["damaged"] == 2
    assert s["damaged_ads"] == 1
    assert s["positive_ads"] == 2
    assert s["ads_total"] == 3


def test_photo_damage_metric_aggregates_frames_to_ads():
    """Regression coverage for `test_photo_damage_metric_aggregates_frames_to_ads`."""
    import pandas as pd
    from kz.ml.photo_damage import per_ad

    frames = pd.DataFrame(
        {
            "ad_id": ["a", "a", "b"],
            "target": [0, 1, 0],
            "table": [0.1, 0.2, 0.3],
            "photo": [0.4, 0.8, 0.2],
            "combined": [0.2, 0.7, 0.1],
        }
    )
    ads = per_ad(frames).set_index("ad_id")
    assert len(ads) == 2
    assert ads.loc["a", "target"] == 1
    assert ads.loc["a", "photo"] == 0.8


def test_photo_damage_reports_paired_auc_difference():
    """Regression coverage for `test_photo_damage_reports_paired_auc_difference`."""
    import numpy as np
    from kz.ml.photo_damage import auc_delta_ci

    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = good[::-1]
    delta, lo, hi = auc_delta_ci(y, good, bad, n_boot=200)
    assert delta == 1.0
    assert lo > 0 and hi > 0


def test_photo_damage_groups_same_image_across_different_ads():
    """Regression coverage for `test_photo_damage_groups_same_image_across_different_ads`."""
    from kz.ml.photo_damage import groups_from_hashes

    groups = groups_from_hashes(
        ["ad-a", "ad-a", "ad-b", "ad-c"],
        ["hash-1", "hash-2", "hash-1", "hash-3"],
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_photo_damage_reports_pr_auc_with_interval():
    """Regression coverage for `test_photo_damage_reports_pr_auc_with_interval`."""
    import numpy as np
    from kz.ml.photo_damage import average_precision_ci

    y = np.array([0, 0, 0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    value, lo, hi = average_precision_ci(y, good, n_boot=200)
    assert value == 1.0
    assert 0.9 < lo <= hi <= 1.0


def test_photo_ablation_fits_pca_inside_train_fold():
    """Regression coverage for `test_photo_ablation_fits_pca_inside_train_fold`."""
    import inspect
    from kz.ml import photo_ablation

    src = inspect.getsource(photo_ablation.cv_mape_with_embeddings)
    assert "fit_transform(emb[tr])" in src
    assert "transform(emb[te])" in src
    main = inspect.getsource(photo_ablation.main)
    assert "reduce_embeddings" not in main


def test_damage_box_is_stored_relative_not_in_pixels():
    """Regression coverage for `test_damage_box_is_stored_relative_not_in_pixels`."""
    import inspect
    from kz.report import photo_labels

    src = inspect.getsource(photo_labels._normalise_boxes)
    assert "0 <= x1 < x2 <= 1" in src, "рамка обязана быть в долях 0..1"


def test_new_photo_labels_keep_provenance_and_old_rows(tmp_path, monkeypatch):
    """Regression coverage for `test_new_photo_labels_keep_provenance_and_old_rows`."""
    from kz.report import photo_labels as pl

    labels = tmp_path / "labels.csv"
    previous = tmp_path / "labels.prev.csv"
    legacy_header = pl.HEADER[:10]
    monkeypatch.setattr(pl, "LABELS_CSV", str(labels))
    monkeypatch.setattr(pl, "LABELS_PREV", str(previous))
    monkeypatch.setattr(pl, "_snapshot_done", False)
    pl.write_journal(
        legacy_header,
        [
            {
                "ad_id": "old",
                "position": "1",
                "path": "old.jpg",
                "label": "intact",
                "labeled_at": "2026-01-01T00:00:00",
            }
        ],
    )

    pl.save_label(
        "new",
        2,
        "new.jpg",
        "intact",
        selection_source="random_audit",
        dataset_split="audit",
        annotator="sanzhar",
    )
    header, rows = pl.read_journal()
    assert all(c in header for c in pl.HEADER)
    assert [r["ad_id"] for r in rows] == ["old", "new"]
    assert rows[0]["label"] == "intact"
    assert rows[1]["selection_source"] == "random_audit"
    assert rows[1]["dataset_split"] == "audit"
    assert rows[1]["label_version"] == pl.LABEL_VERSION


def test_photo_audit_split_is_stable_and_not_everything():
    """Regression coverage for `test_photo_audit_split_is_stable_and_not_everything`."""
    from kz.report.photo_labels import split_for_ad

    once = [split_for_ad(str(i)) for i in range(200)]
    twice = [split_for_ad(str(i)) for i in range(200)]
    assert once == twice
    assert 20 <= once.count("audit") <= 60


def test_photo_labels_export_to_detector_ready_coco(tmp_path):
    """Regression coverage for `test_photo_labels_export_to_detector_ready_coco`."""
    from PIL import Image
    from kz.ml.photo_dataset import build_coco

    damaged = tmp_path / "damaged.jpg"
    intact = tmp_path / "intact.jpg"
    Image.new("RGB", (200, 100), "white").save(damaged)
    Image.new("RGB", (80, 60), "white").save(intact)
    rows = [
        {
            "ad_id": "a",
            "position": "1",
            "path": str(damaged),
            "label": "damaged",
            "x1": "0.1",
            "y1": "0.2",
            "x2": "0.6",
            "y2": "0.7",
            "dataset_split": "train",
        },
        {
            "ad_id": "b",
            "position": "1",
            "path": str(intact),
            "label": "intact",
            "dataset_split": "train",
        },
        {
            "ad_id": "c",
            "position": "1",
            "path": str(intact),
            "label": "intact",
            "dataset_split": "audit",
        },
    ]
    coco = build_coco(rows, "train")
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["bbox"] == [20.0, 20.0, 100.0, 50.0]


def test_multiple_damage_boxes_round_trip_and_export_to_coco(tmp_path, monkeypatch):
    """Regression coverage for `test_multiple_damage_boxes_round_trip_and_export_to_coco`."""
    from PIL import Image
    from kz.ml.photo_dataset import build_coco
    from kz.report import photo_labels as pl

    labels = tmp_path / "labels.csv"
    image = tmp_path / "two-dents.jpg"
    Image.new("RGB", (200, 100), "white").save(image)
    monkeypatch.setattr(pl, "LABELS_CSV", str(labels))
    monkeypatch.setattr(pl, "LABELS_PREV", str(tmp_path / "labels.prev.csv"))
    monkeypatch.setattr(pl, "_snapshot_done", False)

    pl.save_label(
        "a",
        1,
        str(image),
        "damaged",
        boxes=[
            (0.1, 0.2, 0.3, 0.4),
            (0.5, 0.1, 0.9, 0.6),
        ],
    )
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
    """Regression coverage for `test_damage_label_rejects_what_would_poison_training`."""
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
    """Regression coverage for `test_damage_relabel_updates_the_row`."""
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
    """Regression coverage for `test_damage_queue_is_stratified_not_random`."""
    import inspect
    from kz.report import photo_labels

    src = inspect.getsource(photo_labels.queue)
    assert "suspect" in src and "CONTROL_PER_POSITIVE" in src

    assert "sample(frac=1.0" in src


def test_damage_routes_are_closed_in_public_mode():
    """Regression coverage for `test_damage_routes_are_closed_in_public_mode`."""
    from pathlib import Path

    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    damage = src[src.index("def damage_page") : src.index("def label_page")]
    assert damage.count("if PUBLIC_DEMO:") >= 2, "закрыты обе точки: показ и запись"


def test_price_review_journal_is_validated_atomic_and_recoverable(tmp_path, monkeypatch):
    """A correction updates one row and preserves a recovery snapshot."""
    from kz.report import price_review as review

    journal = tmp_path / "price_review.csv"
    previous = tmp_path / "price_review.prev.csv"
    monkeypatch.setattr(review, "LABELS_CSV", str(journal))
    monkeypatch.setattr(review, "LABELS_PREV", str(previous))
    monkeypatch.setattr(review, "_snapshot_done", False)

    review.save_review(
        "123",
        "cosmetic",
        "comparable_cash",
        "photos",
        selection_source="random_cheap_control",
        dataset_split="discovery",
    )
    assert not previous.exists()  # no old journal existed before the first write

    monkeypatch.setattr(review, "_snapshot_done", False)
    review.save_review(
        "123",
        "repair_needed",
        "comparable_cash",
        "both",
        comment="dent is disclosed and visible",
        selection_source="random_cheap_control",
        dataset_split="discovery",
    )
    _, rows = review.read_journal()
    assert len(rows) == 1
    assert rows[0]["vehicle_state"] == "repair_needed"
    assert previous.exists()

    try:
        review.save_review(
            "124",
            "looks_bad",
            "comparable_cash",
            "photos",
            selection_source="x",
            dataset_split="discovery",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a vehicle state outside the fixed label dictionary")


def test_price_review_routes_are_closed_in_public_mode():
    """The public estimator must never expose or mutate diagnosis ground truth."""
    from pathlib import Path

    src = Path("kz/web/app.py").read_text(encoding="utf-8")
    review = src[src.index("def price_review_page") : src.index("def label_page")]
    assert review.count("if PUBLIC_DEMO:") >= 2


def test_price_review_pilot_is_fixed_and_contains_controls():
    """The pilot mixes high-error old cars, random controls, and a blind audit."""
    import pandas as pd
    from kz.report import price_review as review

    rows = []
    for i in range(120):
        rows.append(
            {
                "ad_id": str(i),
                "dataset_split": "audit" if i < 25 else "discovery",
                "age": 25 if i % 2 else 15,
                "absolute_percentage_error_pct": float(i),
            }
        )
    selected = review.select_pilot(pd.DataFrame(rows))
    assert len(selected) == review.PILOT_SIZE
    counts = selected["selection_source"].value_counts().to_dict()
    assert counts["random_local_audit"] == review.AUDIT_SIZE
    assert counts["old_high_oof_error"] == review.HIGH_ERROR_SIZE
    assert counts["random_cheap_control"] == review.RANDOM_CONTROL_SIZE
    assert (selected.loc[selected.selection_source == "old_high_oof_error", "age"] >= 21).all()
    assert selected.ad_id.tolist() == review.select_pilot(pd.DataFrame(rows)).ad_id.tolist()


def test_price_review_oof_alignment_fails_closed(monkeypatch):
    """A stale OOF file must never be attached to the wrong listing."""
    import pandas as pd
    from kz.report import price_review as review

    training = pd.DataFrame(
        {
            "ad_id": ["1", "2"],
            "brand": ["A", "B"],
            "model": ["X", "Y"],
            "year": [2000, 2001],
            "age": [27, 26],
            "price_tenge": [1_000_000, 2_000_000],
            "description": ["one meaningful description", "another meaningful description"],
        }
    )
    groups = review.duplicate_groups(training).astype(str).tolist()
    oof = pd.DataFrame(
        {
            "duplicate_group": groups[::-1],
            "age": [27, 26],
            "actual_price_tenge": [1_000_000, 2_000_000],
            "routed_pred_tenge": [1_100_000, 2_100_000],
            "base_pred_tenge": [1_100_000, 2_100_000],
            "baseline_pred_tenge": [1_100_000, 2_100_000],
            "absolute_percentage_error_pct": [10.0, 5.0],
        }
    )
    try:
        review.align_oof(training, oof)
    except ValueError as exc:
        assert "stale or reordered" in str(exc)
    else:
        raise AssertionError("attached reordered OOF diagnostics to listings")


def test_price_review_page_combines_gallery_reason_labels_and_precise_cv_link():
    """One listing-level screen links to exact frame-level CV annotation."""
    from kz.web.price_review_page import page

    row = {
        "ad_id": "123",
        "brand": "Toyota",
        "model": "Camry",
        "price_tenge": 4_000_000,
        "year": 1999,
        "age": 28,
        "mileage_km": 200_000,
        "engine_volume": 2.2,
        "transmission": "автомат",
        "body_type": "седан",
        "photos_count": 5,
        "price_basis": "ambiguous",
        "description": "summary",
        "seller_comment": "comment",
        "photos": [
            {"position": 1, "path": "data/photos/12/123_1.jpg", "src": "/photos/12/123_1.jpg"}
        ],
    }
    html = page([row], [])
    assert "Below-5M condition review" in html
    assert "vehicle_state" in html and "price_validity" in html and "evidence_source" in html
    assert "Precisely label this frame for CV" in html
    assert "/damage?ad_id=" in html
    assert "predictions and errors are hidden" in html.lower()


def test_damage_labelling_never_touches_kolesa():
    """Regression coverage for `test_damage_labelling_never_touches_kolesa`."""
    import ast
    from pathlib import Path

    def code_only(path: str) -> str:
        """Implement `code_only`."""
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(
                node
            ):
                node.body = node.body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    for f in ("kz/web/damage_page.py", "kz/report/photo_labels.py"):
        src = code_only(f)
        for bad in ("kolesa.kz", "requests.get", "urlopen", "httpx"):
            assert bad not in src, f"{f}: {bad}"


def test_labels_path_can_be_redirected_away_from_the_real_journal():
    """Regression coverage for `test_labels_path_can_be_redirected_away_from_the_real_journal`."""
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

        code = (
            "from kz.report.label_cards.journal import LABELS_CSV, "
            "LABELS_PREV; print(LABELS_CSV); print(LABELS_PREV)"
        )
        paths = subprocess.check_output(
            [sys.executable, "-c", code], env=os.environ, text=True
        ).splitlines()
        assert paths and all(p.startswith("/tmp/kz_scratch") for p in paths)
    finally:
        if saved is None:
            os.environ.pop("KZ_LABELS_DIR", None)
        else:
            os.environ["KZ_LABELS_DIR"] = saved
        importlib.reload(pl)


def test_damage_flow_asks_before_it_records():
    """Regression coverage for `test_damage_flow_asks_before_it_records`."""
    from pathlib import Path

    src = Path("kz/web/damage_page.py").read_text(encoding="utf-8")

    assert "openAsk('damaged')" in src
    mouseup = src[src.index("addEventListener('mouseup'") : src.index("async function commit")]
    assert "commit(" not in mouseup, "сохранение не должно идти по отпусканию мыши"
    assert "a-save" in src and "a-cancel" in src


def test_damage_relabel_saves_the_visible_frame_not_queue_index():
    """Regression coverage for `test_damage_relabel_saves_the_visible_frame_not_queue_index`."""
    from pathlib import Path

    src = Path("kz/web/damage_page.py").read_text(encoding="utf-8")
    commit = src[
        src.index("async function commit") : src.index("document.getElementById('a-save')")
    ]
    assert "const it = view[i]" in commit
    assert "const it = QUEUE[i]" not in commit


def test_damage_ui_collects_multiple_boxes_before_one_commit():
    """Regression coverage for `test_damage_ui_collects_multiple_boxes_before_one_commit`."""
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
    """Regression coverage for `test_damage_ui_uses_exact_english_dataset_labels`."""
    from kz.web import damage_page

    page = damage_page.page([], {}, [])
    for label in ("Damaged", "Wreck", "Parts", "Intact", "Unclear"):
        assert f">{label}<" in page or f">{label}<kbd>" in page
    assert "Intact = no impact/dent" in page
    assert "Rust, dirt, and scuffs still belong here" in page
    for old in (
        ">повреждение кузова<",
        ">серьёзная авария<",
        ">разобрана / снят агрегат<",
        ">целая<",
        ">не понять<",
    ):
        assert old not in page
    assert "poor appearance is not the same as impact damage" in page
    assert "Needs review" in page


def test_legacy_damaged_labels_are_quarantined_until_review(tmp_path, monkeypatch):
    """Regression coverage for `test_legacy_damaged_labels_are_quarantined_until_review`."""
    import csv
    from PIL import Image
    from kz.report import photo_labels as pl
    from kz.ml.photo_dataset import build_coco

    journal = tmp_path / "photo_labels.csv"
    backup = tmp_path / "photo_labels.before_review.csv"
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), "white").save(image)
    with journal.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ad_id",
                "position",
                "path",
                "label",
                "x1",
                "y1",
                "x2",
                "y2",
                "comment",
                "labeled_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ad_id": "1",
                "position": "1",
                "path": str(image),
                "label": "damaged",
                "x1": "0.1",
                "y1": "0.1",
                "x2": "0.5",
                "y2": "0.5",
                "comment": "ржавчина",
                "labeled_at": "2026-08-29T00:00:00",
            }
        )
    monkeypatch.setattr(pl, "LABELS_CSV", str(journal))
    monkeypatch.setattr(pl, "LABELS_PREV", str(tmp_path / "prev.csv"))
    monkeypatch.setattr(pl, "LABELS_REVIEW_BACKUP", str(backup))
    monkeypatch.setattr(pl, "_snapshot_done", False)

    assert pl.mark_legacy_damaged_for_review() == 1
    assert backup.exists()
    _, rows = pl.read_journal()
    assert rows[0]["label"] == "damaged"
    assert rows[0]["review_status"] == pl.NEEDS_REVIEW
    assert build_coco(rows, "train")["images"] == []

    pl.save_label("1", 1, str(image), "intact", comment="только ржавчина")
    _, reviewed = pl.read_journal()
    assert len(reviewed) == 1
    assert reviewed[0]["review_status"] == pl.REVIEWED
    assert reviewed[0]["label"] == "intact"


def test_damage_endpoint_ignores_client_supplied_photo_path():
    """Regression coverage for `test_damage_endpoint_ignores_client_supplied_photo_path`."""
    import importlib
    import inspect

    web = importlib.import_module("kz.web.app")
    src = inspect.getsource(web.damage_label)
    assert 'str(provenance["path"])' in src
    assert 'str(data["path"])' not in src
    assert "_damage_queue = [r for r in _damage_queue" in src


def test_verdict_counter_shows_the_journal_not_just_the_queue():
    """Regression coverage for `test_verdict_counter_shows_the_journal_not_just_the_queue`."""
    import inspect
    from kz.report.label_cards import render

    src = inspect.getsource(render.build)
    assert "journal_total" in src
    assert "read_journal" in src, "число берётся из журнала, а не из карточек"


def test_disassembled_car_is_its_own_class_not_damage():
    """Regression coverage for `test_disassembled_car_is_its_own_class_not_damage`."""
    from kz.report.photo_labels import LABELS

    assert "parts" in LABELS and "damaged" in LABELS
    assert LABELS["parts"] != LABELS["damaged"]


def test_box_is_required_for_damage_and_allowed_everywhere_else(tmp_path, monkeypatch):
    """Regression coverage for `test_box_is_required_for_damage_and_allowed_everywhere_else`."""
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

    for bad in ((0.5, 0.1, 0.2, 0.5), (-0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 1.5, 0.5)):
        try:
            pl.save_label("9", 1, "p.jpg", "intact", box=bad)
        except ValueError:
            continue
        raise AssertionError(f"принял негодную рамку {bad}")
