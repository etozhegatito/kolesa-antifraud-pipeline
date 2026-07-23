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

Метрика — честный 5-fold CV (mean ± std), а не один случайный сплит.
Модельный антифрод (residual «дёшево для этой машины») вынесен в
residual_detector.py (квантильный пол + min-support + ограничение возраста).

Запуск: python train_price_model.py   (офлайн, только Postgres)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from data_quality import scrub_junk_mileage, iforest_anomaly
from db import get_engine

# Числовые (CatBoost ест NaN нативно — пробег с 33% пропусков ок)
NUM_FEATURES = ["age", "mileage_km", "engine_volume", "photos_count",
                "is_mileage_missing", "is_vip", "has_monthly_price"]
# Категориальные (CatBoost сам кодирует — model с 815 значениями не проблема)
CAT_FEATURES = ["brand", "model", "engine_type", "transmission",
                "body_type", "condition"]
# ИЗМЕРЕНО 2026-07-23: структурные фичи обогащения (customs_cleared/steering/
# drive) и текстовые (text_features.py) обе дали flat — MAPE ~24% без сдвига,
# важность <3%. Причины: покрытие 28% (72% NaN), малая вариация (справа 53,
# «нерастаможен» 16), частичная избыточность с brand/model. Вывод: baseline
# (R²≈0.91) близок к ПОТОЛКУ табличных фич — следующий прирост от БОЛЬШЕГО
# объёма (backfill) и НОВЫХ модальностей (фото/CV, полный текст), не от новых
# табличных признаков. Вернуть customs/drive стоит после роста покрытия.
FEATURES = NUM_FEATURES + CAT_FEATURES
# Текстовые фичи (text_features.py) ПРОБОВАЛИ и измерили: на текущих данных
# (медиана 51 символ, лишь 27% полных комментов) дали flat — R² тот же 0.914,
# MAPE 24.0→24.5, суммарная важность ~3%. Причина: сигнал избыточен с brand/
# engine_volume/body_type. Вернём, когда backfill наберёт полные комменты
# (там уже TF-IDF/эмбеддинги). Модуль готов и покрыт тестом — не выкидываем.


def load() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM clean_data", get_engine())
    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["price_tenge"] = pd.to_numeric(df["price_tenge"], errors="coerce")
    # у категориальных CatBoost требует строки без NaN
    for c in CAT_FEATURES:
        df[c] = df[c].astype("string").fillna("NA").astype(str)
    return df


def cross_validate(X, y, n_splits=5):
    """k-fold CV: честный разброс метрики вместо ОДНОГО случайного сплита
    (один сплит мог повезти/не повезти). Возвращает массив (n_splits, 3):
    по фолдам [R²(log), MAE(₸), MAPE(%)]."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = []
    for tr, te in kf.split(X):
        m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8,
                              loss_function="RMSE", random_seed=42, verbose=False)
        m.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        pred = m.predict(X.iloc[te])
        tp, pp = np.exp(y.iloc[te]), np.exp(pred)      # обратно в тенге
        out.append([r2_score(y.iloc[te], pred),
                    mean_absolute_error(tp, pp),
                    float((np.abs(pp - tp) / tp).mean() * 100)])
    return np.array(out)


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

    # Честная метрика — 5-fold CV (не один случайный сплит): mean ± std.
    cv = cross_validate(X, y)
    r2, mae, mape = cv.mean(axis=0)
    r2s, maes, mapes = cv.std(axis=0)
    print(f"\n5-fold CV ({len(X)} машин):  R²(log)={r2:.3f}±{r2s:.3f}   "
          f"MAE={mae/1e6:.2f}±{maes/1e6:.2f}М ₸   MAPE={mape:.1f}±{mapes:.1f}%")

    # Финальная модель на ВСЕХ чистых — для важности признаков.
    model = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8,
                              loss_function="RMSE", random_seed=42, verbose=False)
    model.fit(Pool(X, y, cat_features=CAT_FEATURES))
    print("\nВажность признаков:")
    for f, v in sorted(zip(FEATURES, model.get_feature_importance()),
                       key=lambda x: -x[1]):
        print(f"  {f:<18} {v:5.1f}")
    print("\n(residual-антифрод — в residual_detector.py: квантиль + min-support + age)")


if __name__ == "__main__":
    main()
