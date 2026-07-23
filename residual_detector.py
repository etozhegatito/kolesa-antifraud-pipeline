# -*- coding: utf-8 -*-
"""
residual_detector.py — модельный антифрод: «дёшево ДЛЯ ЭТОЙ машины» (roadmap п.3).

Обучаем CatBoost предсказывать НИЖНИЙ КВАНТИЛЬ цены (alpha) по признакам —
«справедливый пол». Цена ниже пола = подозрительно дёшево.

Почему квантиль, а не точечный прогноз: он CONFIDENCE-AWARE. У сегмента с
широким разбросом (редкие/старые машины) пол уходит низко → туда трудно
«провалиться» → меньше ложных флагов. У плотного сегмента (популярные модели)
пол близко к медиане → реально дешёвая проваливается → ловится. Это и чинит
«грязный top-10» сырого residual, где редкие машины всплывали из-за
неуверенности модели, а не из-за приманки.

Учим на чистых (is_suspicious=0), как и модель цены; фичи те же (без утечки).
Предсказываем для ВСЕХ, флагим actual < пол, ранжируем по глубине провала.

Запуск: python residual_detector.py   (офлайн, только Postgres)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from data_quality import scrub_junk_mileage
from train_price_model import CAT_FEATURES, FEATURES, load

ALPHA = 0.10       # нижний квантиль = «пол» справедливой цены (10-й перцентиль)
MIN_SUPPORT = 8    # минимум похожих (brand+model) в обучении, чтобы ДОВЕРЯТЬ полу
                   # (иначе редкая машина проваливается из-за незнания, а не приманки)
AGE_MAX = 10       # residual осмыслен только на молодых: у старья (80-90-е) цену
                   # решает состояние (данных о нём нет) → разброс огромен → шум


def main():
    df = load()
    df = df[df["price_tenge"].notna() & (df["price_tenge"] > 0)].copy()
    df["log_price"] = np.log(df["price_tenge"])
    df, _ = scrub_junk_mileage(df)                      # та же чистка, что у модели

    clean = df[df["is_suspicious"] == 0]                # учим только на чистых
    Xtr, Xte, ytr, yte = train_test_split(
        clean[FEATURES], clean["log_price"], test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=8,
        loss_function=f"Quantile:alpha={ALPHA}",        # ← ключ: квантильная потеря
        random_seed=42, verbose=False)
    model.fit(Pool(Xtr, ytr, cat_features=CAT_FEATURES))

    # Калибровка: у честного квантиля α доля реальных цен НИЖЕ пола ≈ α.
    frac_below = float((yte < model.predict(Xte)).mean())
    print(f"Калибровка (holdout): доля ниже пола = {frac_below:.2f} "
          f"(ожидаем ≈ α = {ALPHA})")

    # Пол для ВСЕХ строк; кто ниже — и насколько (в логе = «во сколько раз»).
    df["floor_log"] = model.predict(df[FEATURES])
    df["below_floor"] = df["log_price"] < df["floor_log"]
    df["gap"] = df["floor_log"] - df["log_price"]       # >0 = ниже пола

    # min-support: сколько похожих (brand+model) было в ОБУЧЕНИИ. Полу доверяем
    # только при достаточной опоре — иначе редкая машина проваливается из-за
    # незнания модели, а не из-за приманки (кейс DongFeng Nano, ретро).
    support = clean.groupby(["brand", "model"]).size().rename("support").reset_index()
    df = df.merge(support, on=["brand", "model"], how="left")
    df["support"] = df["support"].fillna(0).astype(int)
    df["flag"] = (df["below_floor"] & (df["support"] >= MIN_SUPPORT)
                  & (df["age"] <= AGE_MAX))

    n_below = int(df["below_floor"].sum())
    n_flag = int(df["flag"].sum())
    print(f"\nНиже пола: {n_below} ({n_below/len(df):.0%}); из них с опорой "
          f"≥{MIN_SUPPORT}: {n_flag} — это и есть кандидаты (остальное — редкие,"
          f" полу не доверяем)")

    # Согласие с правиловым детектором (независимая проверка).
    rb = df["is_suspicious"] == 1
    agree = int((df["flag"] & rb).sum())
    print(f"Из правиловых подозрительных ({int(rb.sum())}) попали в кандидаты: {agree} "
          f"({agree/max(int(rb.sum()),1):.0%})")

    # Топ кандидатов = самый глубокий провал под пол (сильнейший сигнал).
    print("\nТоп-12 кандидатов (глубже всего под полом, с достаточной опорой):")
    top = df[df["flag"]].nlargest(12, "gap").copy()
    top["факт_М"] = (top["price_tenge"] / 1e6).round(1)
    top["пол_М"] = (np.exp(top["floor_log"]) / 1e6).round(1)
    print(top[["ad_id", "brand", "model", "year", "факт_М", "пол_М",
               "gap", "is_suspicious"]].to_string(index=False))


if __name__ == "__main__":
    main()
