# -*- coding: utf-8 -*-
"""Inference сохранённого артефакта модели справедливой цены.

Этот файл модель НЕ переобучает. Поэтому один и тот же артефакт используется
во всех вызовах, а его метрики, commit и fingerprint данных можно проверить в
data/models/price_model.metadata.json.
"""

from datetime import date

import numpy as np
import pandas as pd

from train_price_model import (
    CAT_FEATURES,
    FEATURES,
    NUM_FEATURES,
    coerce_features,
    load,
    load_artifact,
)

CY = date.today().year


def make_row(**car) -> pd.DataFrame:
    """Преобразует публичные поля машины в точную train-схему признаков."""
    car = car.copy()
    row = {f: np.nan for f in NUM_FEATURES}
    row.update({f: "NA" for f in CAT_FEATURES})
    if "year" in car:
        row["age"] = CY - car.pop("year") + 1
    mileage = car.get("mileage_km")
    row["is_mileage_missing"] = int(
        mileage is None or (isinstance(mileage, float) and np.isnan(mileage))
    )
    for k, v in car.items():
        if k in FEATURES:
            row[k] = v
    return coerce_features(pd.DataFrame([row]))[FEATURES]


def estimate(mdl, **car) -> float:
    """Оценка цены (₸) по признакам машины. Пример:
       estimate(m, brand='Toyota', model='Camry', year=2019, engine_volume=2.5).
       (первый аргумент — mdl, а не model, чтобы не спутать с фичей model=марка)"""
    return float(np.exp(mdl.predict(make_row(**car))[0]))


def main():
    m, metadata = load_artifact()
    temporal = metadata["validation"].get("temporal_holdout")
    print(f"Артефакт: {metadata['training_rows']} машин, "
          f"создан {metadata['created_at_utc']}")
    if temporal:
        print(f"Честный out-of-time MAPE: {temporal['model']['mape_pct']:.1f}% "
              f"(test={temporal['test_rows']})")

    # Случайная строка — только демонстрация inference, НЕ оценка качества:
    # финальный артефакт обучен на всех чистых данных.
    df = load()
    clean = df[(df["price_tenge"] > 0) & (df["is_suspicious"] == 0)]
    car = clean.sample(1).iloc[0]
    p = estimate(m, brand=car["brand"], model=car["model"], age=int(car["age"]),
                 engine_volume=car["engine_volume"], mileage_km=car["mileage_km"],
                 engine_type=car["engine_type"], transmission=car["transmission"],
                 body_type=car["body_type"], condition=car["condition"])
    print(f"(a) {car['brand']} {car['model']} {int(car['year'])} — оценка модели "
          f"≈ {p/1e6:.1f}М ₸  (в объявлении: {car['price_tenge']/1e6:.1f}М)")
    print("Это иллюстрация. Качество берётся из сохранённой out-of-time "
          "валидации, а не из этой обучающей строки.")


if __name__ == "__main__":
    main()
