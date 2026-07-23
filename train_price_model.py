# -*- coding: utf-8 -*-
"""
train_price_model.py — baseline модель оценки цены (ЯДРО продукта).

Идея: регрессор предсказывает log(цена) по признакам машины.
  • Почему log: цены лог-нормальны (медиана 7.8М, среднее 13.5М — правый
    хвост). В лог-пространстве ошибка симметрична и «в процентах».
  • Обучаем ТОЛЬКО на чистых (is_suspicious=0): детектор для того и чистит
    данные — приманки/дилерский маркетинг отравили бы модель.
  • БЕЗ утечки цели: выкинуты kolesa_avg_price (чужая оценка цены, правило
    №6), price_z (посчитан ИЗ цены), city (везде Алматы), views_count
    (копится пост-фактум, не свойство машины).

Бонус — residual-детектор: предсказываем цену для ВСЕХ (вкл. подозрительных);
большой отрицательный остаток (реальная цена << предсказанной) = кандидат в
приманку. Это модельный антифрод-сигнал ПОВЕРХ правил (roadmap п.3).

Запуск: python train_price_model.py   (офлайн, только Postgres)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_quality import scrub_junk_mileage, iforest_anomaly
from db import get_engine

# Числовые (CatBoost ест NaN нативно — пробег с 33% пропусков ок)
NUM_FEATURES = ["age", "mileage_km", "engine_volume", "photos_count",
                "is_mileage_missing", "is_vip", "has_monthly_price"]
# Категориальные (CatBoost сам кодирует — model с 815 значениями не проблема)
CAT_FEATURES = ["brand", "model", "engine_type", "transmission",
                "body_type", "condition"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def load() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM clean_data", get_engine())
    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["price_tenge"] = pd.to_numeric(df["price_tenge"], errors="coerce")
    # у категориальных CatBoost требует строки без NaN
    for c in CAT_FEATURES:
        df[c] = df[c].astype("string").fillna("NA").astype(str)
    return df


def main():
    df = load()
    df = df[df["price_tenge"].notna() & (df["price_tenge"] > 0)].copy()
    df["log_price"] = np.log(df["price_tenge"])

    # data-quality: занулить плейсхолдер-пробеги (777777 и т.п.) ДО обучения —
    # иначе модель учит мусор как реальные 777k км.
    df, n_junk = scrub_junk_mileage(df)
    if n_junk:
        print(f"Data-quality: занулено {n_junk} плейсхолдер-пробегов (репдигит >300k)")
    # iForest — только репорт для ревью (не удаляем: ловит и редкое-честное)
    dq = iforest_anomaly(df)
    print(f"Data-quality: iForest пометил {int(dq.sum())} строк для ревью "
          f"(глобальные выбросы — старьё/редкость/мусор; НЕ авто-удаляем, НЕ фрод)")

    clean = df[df["is_suspicious"] == 0]            # обучаем только на чистых
    X, y = clean[FEATURES], clean["log_price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8,
                              loss_function="RMSE", random_seed=42, verbose=False)
    model.fit(Pool(Xtr, ytr, cat_features=CAT_FEATURES),
              eval_set=Pool(Xte, yte, cat_features=CAT_FEATURES))

    # ── метрики: R² в логе; MAE/MAPE обратно в тенге ────────────────────────
    pred_log = model.predict(Xte)
    r2 = r2_score(yte, pred_log)
    pred_price, true_price = np.exp(pred_log), np.exp(yte)
    mae = mean_absolute_error(true_price, pred_price)
    mape = (np.abs(pred_price - true_price) / true_price).mean() * 100
    print(f"\nHoldout ({len(Xte)} машин): R²(log)={r2:.3f}  "
          f"MAE={mae/1e6:.2f}М ₸  MAPE={mape:.1f}%")

    print("\nВажность признаков:")
    for f, v in sorted(zip(FEATURES, model.get_feature_importance()),
                       key=lambda x: -x[1]):
        print(f"  {f:<18} {v:5.1f}")

    # ── residual-детектор на ВСЕХ (вкл. подозрительных) ─────────────────────
    df["pred_log"] = model.predict(df[FEATURES])
    df["resid"] = df["log_price"] - df["pred_log"]   # <0 = дешевле предсказанного
    ok = df[df.is_suspicious == 0]["resid"]
    susp = df[df.is_suspicious == 1]["resid"]
    print(f"\nСредний остаток: чистые {ok.mean():+.2f}, "
          f"подозрительные (правила) {susp.mean():+.2f}")
    print("→ если подозрительные заметно НИЖЕ — модель и правила согласны")
    print("\nТоп-10 «дешевле модели» (residual-кандидаты, вкл. неразмеченные):")
    top = df.nsmallest(10, "resid").copy()
    top["факт_М"] = (top["price_tenge"] / 1e6).round(1)
    top["предикт_М"] = (np.exp(top["pred_log"]) / 1e6).round(1)
    print(top[["ad_id", "brand", "model", "year", "факт_М", "предикт_М",
               "resid", "is_suspicious"]].to_string(index=False))


if __name__ == "__main__":
    main()
