# -*- coding: utf-8 -*-
"""
predict_price.py — «дай характеристики машины → получи справедливую цену».

Показывает модель в деле:
  (a) estimate(...) — оценка для ЛЮБОЙ машины по её признакам (что знаешь —
      укажи, остальное модель додумает: CatBoost терпит пропуски);
  (b) на 8 машинах, которых модель НЕ видела при обучении: предсказание vs
      реальная цена + ошибка в % — наглядно, ОТКУДА берётся средняя ~24% (MAPE).

Запуск: python predict_price.py   (офлайн, только Postgres)
"""

from datetime import date

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from data_quality import scrub_junk_mileage
from train_price_model import CAT_FEATURES, FEATURES, NUM_FEATURES, load

CY = date.today().year


def estimate(mdl, **car) -> float:
    """Оценка цены (₸) по признакам машины. Пример:
       estimate(m, brand='Toyota', model='Camry', year=2019, engine_volume=2.5).
       (первый аргумент — mdl, а не model, чтобы не спутать с фичей model=марка)"""
    row = {f: np.nan for f in NUM_FEATURES}
    row.update({f: "NA" for f in CAT_FEATURES})
    if "year" in car:
        row["age"] = CY - car.pop("year") + 1          # модель знает возраст, не год
    row["is_mileage_missing"] = 0 if car.get("mileage_km") else 1
    for k, v in car.items():
        if k in FEATURES:
            row[k] = v
    X = pd.DataFrame([row])[FEATURES]
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return float(np.exp(mdl.predict(X)[0]))            # exp: обратно из log в тенге


def main():
    df = load()
    df = df[df["price_tenge"] > 0].copy()
    df["log_price"] = np.log(df["price_tenge"])
    df, _ = scrub_junk_mileage(df)
    clean = df[df["is_suspicious"] == 0]

    Xtr, Xte, ytr, yte = train_test_split(
        clean[FEATURES], clean["log_price"], test_size=0.2, random_state=42)
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8,
                          loss_function="RMSE", random_seed=42, verbose=False)
    m.fit(Pool(Xtr, ytr, cat_features=CAT_FEATURES))

    # (a) оценка «своей» машины
    p = estimate(m, brand="Toyota", model="Camry", year=2019, engine_volume=2.5,
                 mileage_km=90000, engine_type="бензин", transmission="автомат",
                 body_type="седан", condition="б/у")
    print(f"(a) Toyota Camry 2019, 2.5 бензин, 90 000 км → оценка ≈ {p/1e6:.1f}М ₸")

    # (b) откуда 24%: pred vs факт на НЕВИДЕННЫХ моделью машинах
    te = clean.loc[Xte.index].copy()
    te["pred"] = np.exp(m.predict(Xte))
    te["err"] = (np.abs(te["pred"] - te["price_tenge"]) / te["price_tenge"] * 100)
    s = te.sample(8, random_state=1).copy()
    s["факт_М"] = (s["price_tenge"] / 1e6).round(1)
    s["пред_М"] = (s["pred"] / 1e6).round(1)
    s["ошибка_%"] = s["err"].round(0)
    print("\n(b) 8 машин, которых модель НЕ видела при обучении:")
    print(s[["brand", "model", "year", "факт_М", "пред_М", "ошибка_%"]].to_string(index=False))
    print(f"\nСредняя ошибка по ВСЕМ {len(te)} невиденным = {te['err'].mean():.0f}% "
          f"= MAPE. Одни угаданы точно, другие мимо — среднее ≈ 24%.")


if __name__ == "__main__":
    main()
