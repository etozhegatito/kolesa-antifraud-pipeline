# -*- coding: utf-8 -*-
"""Implementation for the `kz.transform.data_quality` module."""

import numpy as np
import pandas as pd


DQ_FEATURES = ["age", "mileage_km", "engine_volume", "photos_count"]


def is_junk_mileage(m) -> bool:
    """Implement `is_junk_mileage`."""
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return False
    try:
        s = str(int(m))
    except (ValueError, TypeError):
        return False
    return len(s) >= 5 and len(set(s)) == 1 and int(m) > 300_000


def scrub_junk_mileage(df: pd.DataFrame, col: str = "mileage_km"):
    """Implement `scrub_junk_mileage`."""
    df = df.copy()
    junk = df[col].map(is_junk_mileage)
    df.loc[junk, col] = np.nan
    return df, int(junk.sum())


def iforest_anomaly(df: pd.DataFrame, features=None, contamination: float = 0.02) -> pd.Series:
    """Implement `iforest_anomaly`."""
    from sklearn.ensemble import IsolationForest

    feats = features or DQ_FEATURES
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    iso = IsolationForest(contamination=contamination, random_state=42)
    return pd.Series(iso.fit_predict(X) == -1, index=df.index)
