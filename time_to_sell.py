# -*- coding: utf-8 -*-
"""
time_to_sell.py — ПРОТОТИП уровня 2: сколько объявление «живёт» до продажи и
как это зависит от цены (roadmap п.2 — оптимальная цена под срок продажи).

Как считаем срок: `дней_на_рынке ≈ дата_архива − posted_date`.
  • posted_date — когда объявление опубликовано (из карточки листинга);
  • дата_архива — когда check_status нашёл его archived (≈ продано/снято).
Для оценки «цена vs скорость»: берём отклонение цены от модельной «справедливой»
(residual): гипотеза — дешевле справедливой → продаётся быстрее.

⚠️ ЧЕСТНЫЕ ОГРАНИЧЕНИЯ (пока данных мало):
  • истории ~неделя, всё запощено 14-20 июля → сроки короткие;
  • archived ≠ обязательно «продано» (может истекло/снято);
  • дата_архива = когда МЫ проверили, а не точный момент архивации (завышает срок);
  • posted_date может быть датой «поднятия» VIP (занижает срок).
Это КАРКАС: выводы слабые сейчас, но усилятся с накоплением истории.

Запуск: python time_to_sell.py   (офлайн)
"""

import re
from datetime import date

import numpy as np
import pandas as pd
from data_quality import scrub_junk_mileage
from db import get_engine
from train_price_model import FEATURES, load, load_artifact

YEAR = date.today().year
_MON = {"янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "июн": 6,
        "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12}


def parse_posted(s):
    """'18 июля' / '18 июл.' → date(2026, 7, 18). Иначе None."""
    m = re.match(r"\s*(\d{1,2})\s+([а-яё]+)", str(s).lower())
    if not m:
        return None
    day, mon = int(m.group(1)), _MON.get(m.group(2)[:3])
    if not mon:
        return None
    try:
        return date(YEAR, mon, day)
    except ValueError:
        return None


def fair_price_model():
    """Сохранённая production-модель и данные для расчёта residual."""
    df = load()
    df = df[df["price_tenge"] > 0].copy()
    df["log_price"] = np.log(df["price_tenge"])
    df, _ = scrub_junk_mileage(df)
    m, _ = load_artifact()
    return m, df


def main():
    model, df = fair_price_model()
    eng = get_engine()

    posted = pd.read_sql("SELECT ad_id, posted_date FROM raw_ads", eng, dtype={"ad_id": str})
    posted["posted"] = posted["posted_date"].map(parse_posted)
    st = pd.read_sql("SELECT ad_id, status, checked_at FROM ad_status WHERE status='archived'",
                     eng, dtype={"ad_id": str})
    st["archived_at"] = pd.to_datetime(st["checked_at"]).dt.date

    m = st.merge(posted[["ad_id", "posted"]], on="ad_id").dropna(subset=["posted"])
    m["days"] = m.apply(lambda r: (r["archived_at"] - r["posted"]).days, axis=1)
    m = m[m["days"] >= 0]

    # residual = цена vs модельная справедливая (лог): <0 = дешевле справедливой
    feats = df.set_index("ad_id")
    m = m.merge(df[["ad_id"]], on="ad_id")
    X = feats.loc[m["ad_id"], FEATURES]
    m["resid"] = np.log(pd.to_numeric(feats.loc[m["ad_id"], "price_tenge"]).values) \
        - model.predict(X)

    print(f"Архивных с датируемым сроком: {len(m)}")
    print(f"\nДней на рынке до архива (распределение):")
    print(m["days"].describe().round(1).to_string())

    below = m[m["resid"] < 0]["days"]        # дешевле справедливой
    above = m[m["resid"] >= 0]["days"]       # дороже/по справедливой
    print(f"\nГипотеза «дешевле → быстрее»:")
    print(f"  дешевле справедливой (n={len(below)}): медиана {below.median():.0f} дн")
    print(f"  дороже/ровно      (n={len(above)}): медиана {above.median():.0f} дн")
    if len(m) > 5:
        print(f"  корреляция days↔residual: {m['days'].corr(m['resid']):+.2f} "
              f"(ждём >0: дороже → дольше)")
    print("\n⚠️ n мал и сроки короткие — это КАРКАС, не вывод. Усилится с историей.")


if __name__ == "__main__":
    main()
