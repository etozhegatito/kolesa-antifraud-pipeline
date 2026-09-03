# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.residual_detector` module."""

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

from kz.ml import train_price_model as _tpm
from kz.transform import data_quality
from kz.transform.data_quality import scrub_junk_mileage
from kz.ml.train_price_model import (
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
    """Implement `calibration_offset`."""
    residual = np.asarray(y_log, dtype=float) - np.asarray(raw_floor_log, dtype=float)
    try:
        return float(np.quantile(residual, alpha, method="lower"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(residual, alpha, interpolation="lower"))


def oof_quantile_floor(clean: pd.DataFrame) -> np.ndarray:
    """Implement `oof_quantile_floor`."""
    groups = duplicate_groups(clean)
    n = min(N_SPLITS, groups.nunique())
    if n < 2:
        raise ValueError("Not enough independent groups for residual CV")
    oof = np.full(len(clean), np.nan)
    X, y = clean[FEATURES], clean["log_price"]
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        model = new_model(loss_function=f"Quantile:alpha={ALPHA}")
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        oof[te] = model.predict(X.iloc[te])
    return oof


def fit_calibrated_floor(clean: pd.DataFrame):
    """Implement `fit_calibrated_floor`."""
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
            "Price-floor artifact is missing. Run: python -m kz.ml.residual_detector"
        )
    metadata = json.loads(FLOOR_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != FLOOR_SCHEMA_VERSION:
        raise ValueError("Incompatible price-floor artifact version")
    if metadata.get("features") != FEATURES:
        raise ValueError("Price-floor feature schema does not match the code")
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
    print(f"OOF calibration: fraction below floor={frac_below:.3f}, target alpha={ALPHA:.3f}")
    print(f"Log-floor correction={offset:+.4f} (price multiplier ×{np.exp(offset):.3f})")

    df["floor_log"] = model.predict(df[FEATURES]) + offset
    df.loc[clean["index"], "floor_log"] = oof_floor
    df["below_floor"] = df["log_price"] < df["floor_log"]
    df["gap"] = df["floor_log"] - df["log_price"]

    support = clean.groupby(["brand", "model"]).size().rename("support").reset_index()
    df = df.merge(support, on=["brand", "model"], how="left")
    df["support"] = df["support"].fillna(0).astype(int)

    #

    #

    explained = (
        df.get("info_flags", pd.Series("", index=df.index))
        .fillna("")
        .str.contains("low_price_explained")
    )
    df["flag"] = (
        df["below_floor"] & (df["support"] >= MIN_SUPPORT) & (df["age"] <= AGE_MAX) & ~explained
    )

    metadata = {
        "schema_version": FLOOR_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_sha256": code_fingerprint(__file__, _tpm.__file__, data_quality.__file__),
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
    print(f"\nBelow calibrated floor: {n_below}/{len(df)}; after support/age gates: {n_flag}")
    rb = df["is_suspicious"] == 1
    agree = int((df["flag"] & rb).sum())
    print(f"Agreement with rule-based detector: {agree}/{int(rb.sum())}")

    top = df[df["flag"]].nlargest(12, "gap").copy()
    top["actual_M"] = (top["price_tenge"] / 1e6).round(1)
    top["floor_M"] = (np.exp(top["floor_log"]) / 1e6).round(1)
    print("\nTop 12 candidates (a review queue, not confirmed fraud):")
    print(
        top[
            [
                "ad_id",
                "brand",
                "model",
                "year",
                "actual_M",
                "floor_M",
                "gap",
                "is_suspicious",
            ]
        ].to_string(index=False)
    )
    print(f"\nPrice-floor artifact → {FLOOR_MODEL_PATH}")


if __name__ == "__main__":
    main()
