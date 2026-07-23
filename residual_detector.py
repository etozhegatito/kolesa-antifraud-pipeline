# -*- coding: utf-8 -*-
"""Калиброванный детектор «цена ниже справедливого пола».

CatBoost обучается на нижний квантиль, но номинальный alpha не гарантирует,
что на новых данных ровно alpha наблюдений окажутся ниже пола. Поэтому:

  1. Для чистых строк считаются out-of-fold предсказания без leakage дублей.
  2. По OOF-остаткам вычисляется поправка к полу.
  3. Финальная квантильная модель и поправка сохраняются как артефакт.

Это калибрует порог, но НЕ доказывает fraud: кандидат становится детекцией
только после ручной разметки precision/recall.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from data_quality import scrub_junk_mileage
from train_price_model import (
    CAT_FEATURES,
    FEATURES,
    coerce_features,
    code_fingerprint,
    duplicate_groups,
    load,
    new_model,
)

ALPHA = 0.10
MIN_SUPPORT = 8
AGE_MAX = 10
N_SPLITS = 5

FLOOR_MODEL_PATH = Path("data/models/price_floor.cbm")
FLOOR_METADATA_PATH = Path("data/models/price_floor.metadata.json")
FLOOR_SCHEMA_VERSION = 1


def calibration_offset(y_log, raw_floor_log, alpha: float = ALPHA) -> float:
    """Сдвиг пола: доля y ниже (raw_floor + offset) становится ≈ alpha."""
    residual = np.asarray(y_log, dtype=float) - np.asarray(raw_floor_log, dtype=float)
    try:
        return float(np.quantile(residual, alpha, method="lower"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(residual, alpha, interpolation="lower"))


def oof_quantile_floor(clean: pd.DataFrame) -> np.ndarray:
    """Out-of-fold сырой пол: строка не участвует в модели, которая её оценивает."""
    groups = duplicate_groups(clean)
    n = min(N_SPLITS, groups.nunique())
    if n < 2:
        raise ValueError("Недостаточно независимых групп для residual CV")
    oof = np.full(len(clean), np.nan)
    X, y = clean[FEATURES], clean["log_price"]
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        model = new_model(loss_function=f"Quantile:alpha={ALPHA}")
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        oof[te] = model.predict(X.iloc[te])
    return oof


def fit_calibrated_floor(clean: pd.DataFrame):
    """Финальная модель + OOF-поправка + диагностические OOF-предикты."""
    oof_raw = oof_quantile_floor(clean)
    offset = calibration_offset(clean["log_price"], oof_raw)
    model = new_model(loss_function=f"Quantile:alpha={ALPHA}")
    model.fit(Pool(clean[FEATURES], clean["log_price"], cat_features=CAT_FEATURES))
    return model, offset, oof_raw + offset


def save_floor_artifact(model: CatBoostRegressor, metadata: dict) -> None:
    FLOOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="price_floor.", suffix=".cbm", dir=FLOOR_MODEL_PATH.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        model.save_model(str(tmp))
        os.replace(tmp, FLOOR_MODEL_PATH)
    finally:
        tmp.unlink(missing_ok=True)
    tmp_meta = FLOOR_METADATA_PATH.with_suffix(".json.tmp")
    tmp_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_meta, FLOOR_METADATA_PATH)


def load_floor_artifact():
    if not FLOOR_MODEL_PATH.exists() or not FLOOR_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Нет артефакта ценового пола. Сначала: python residual_detector.py"
        )
    metadata = json.loads(FLOOR_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != FLOOR_SCHEMA_VERSION:
        raise ValueError("Несовместимая версия артефакта ценового пола")
    if metadata.get("features") != FEATURES:
        raise ValueError("Схема признаков ценового пола не совпадает с кодом")
    model = CatBoostRegressor()
    model.load_model(str(FLOOR_MODEL_PATH))
    return model, metadata


def score_floor(model, metadata: dict, X: pd.DataFrame) -> np.ndarray:
    prepared = coerce_features(X)
    return model.predict(prepared[FEATURES]) + float(metadata["calibration_offset_log"])


def main():
    df = load()
    df = df[df["price_tenge"].notna() & (df["price_tenge"] > 0)].copy()
    df, _ = scrub_junk_mileage(df)
    df["log_price"] = np.log(df["price_tenge"])
    clean = df[df["is_suspicious"] == 0].copy().reset_index()

    model, offset, oof_floor = fit_calibrated_floor(clean)
    frac_below = float((clean["log_price"].to_numpy() < oof_floor).mean())
    print(f"OOF-калибровка: доля ниже пола={frac_below:.3f}, цель alpha={ALPHA:.3f}")
    print(f"Поправка к log-полу={offset:+.4f} "
          f"(множитель цены ×{np.exp(offset):.3f})")

    # Для чистых строк — строго OOF-пол. Для уже правилово подозрительных,
    # которых не было в train, — предикт финальной модели.
    df["floor_log"] = model.predict(df[FEATURES]) + offset
    df.loc[clean["index"], "floor_log"] = oof_floor
    df["below_floor"] = df["log_price"] < df["floor_log"]
    df["gap"] = df["floor_log"] - df["log_price"]

    support = clean.groupby(["brand", "model"]).size().rename("support").reset_index()
    df = df.merge(support, on=["brand", "model"], how="left")
    df["support"] = df["support"].fillna(0).astype(int)
    df["flag"] = (
        df["below_floor"]
        & (df["support"] >= MIN_SUPPORT)
        & (df["age"] <= AGE_MAX)
    )

    metadata = {
        "schema_version": FLOOR_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_sha256": code_fingerprint(
            "residual_detector.py", "train_price_model.py", "data_quality.py"
        ),
        "features": FEATURES,
        "alpha": ALPHA,
        "calibration": "grouped_out_of_fold",
        "calibration_rows": int(len(clean)),
        "calibration_offset_log": offset,
        "oof_fraction_below": frac_below,
        "min_support": MIN_SUPPORT,
        "age_max": AGE_MAX,
    }
    save_floor_artifact(model, metadata)

    n_below, n_flag = int(df["below_floor"].sum()), int(df["flag"].sum())
    print(f"\nНиже калиброванного пола: {n_below}/{len(df)}; "
          f"после support/age gates: {n_flag}")
    rb = df["is_suspicious"] == 1
    agree = int((df["flag"] & rb).sum())
    print(f"Согласие с правиловым детектором: {agree}/{int(rb.sum())}")

    top = df[df["flag"]].nlargest(12, "gap").copy()
    top["факт_М"] = (top["price_tenge"] / 1e6).round(1)
    top["пол_М"] = (np.exp(top["floor_log"]) / 1e6).round(1)
    print("\nТоп-12 кандидатов (это очередь на разметку, не доказанный fraud):")
    print(top[[
        "ad_id", "brand", "model", "year", "факт_М", "пол_М",
        "gap", "is_suspicious",
    ]].to_string(index=False))
    print(f"\nАртефакт пола → {FLOOR_MODEL_PATH}")


if __name__ == "__main__":
    main()
