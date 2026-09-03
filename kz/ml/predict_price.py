# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.predict_price` module."""

from datetime import date

import numpy as np
import pandas as pd

from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    NUM_FEATURES,
    coerce_features,
    load,
    load_artifact,
)

CY = date.today().year


def make_row(**car) -> pd.DataFrame:
    """Implement `make_row`."""
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
    """Implement `estimate`."""
    return float(np.exp(mdl.predict(make_row(**car))[0]))


def main():
    m, metadata = load_artifact()
    temporal = metadata["validation"].get("temporal_holdout")
    print(f"Artifact: {metadata['training_rows']} vehicles, created {metadata['created_at_utc']}")
    if temporal:
        print(
            f"Held-out out-of-time MAPE: {temporal['model']['mape_pct']:.1f}% "
            f"(test={temporal['test_rows']})"
        )

    df = load()
    clean = df[(df["price_tenge"] > 0) & (df["is_suspicious"] == 0)]
    car = clean.sample(1).iloc[0]
    p = estimate(
        m,
        brand=car["brand"],
        model=car["model"],
        age=int(car["age"]),
        engine_volume=car["engine_volume"],
        mileage_km=car["mileage_km"],
        engine_type=car["engine_type"],
        transmission=car["transmission"],
        body_type=car["body_type"],
        condition=car["condition"],
    )
    print(
        f"(a) {car['brand']} {car['model']} {int(car['year'])} — model estimate "
        f"≈ {p / 1e6:.1f}M ₸  (listed: {car['price_tenge'] / 1e6:.1f}M)"
    )
    print(
        "This is an illustration. Reported quality comes from saved out-of-time "
        "validation, not from this training row."
    )


if __name__ == "__main__":
    main()
