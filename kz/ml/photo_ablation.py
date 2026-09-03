# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.photo_ablation` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from kz.ml import photo_features
from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    duplicate_groups,
    load,
    new_model,
    prepare_training_data,
    regression_metrics,
)

N_SPLITS = 5


def cv_mape(df: pd.DataFrame, features: list[str], cats: list[str]) -> dict:
    """Implement `cv_mape`."""
    groups = duplicate_groups(df)
    n = min(N_SPLITS, groups.nunique())
    X, y = df[features], df["log_price"]
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        m = new_model()
        m.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=cats))
        oof[te] = m.predict(X.iloc[te])
    r = regression_metrics(y, oof)
    ape = np.abs(np.exp(oof) - np.exp(y)) / np.exp(y)
    return {"mape": r["mape_pct"], "median": float(np.median(ape) * 100), "r2": r["r2_log"]}


def cv_mape_with_embeddings(
    df: pd.DataFrame, emb: np.ndarray, base_features: list[str], cats: list[str]
) -> dict:
    """Implement `cv_mape_with_embeddings`."""
    groups = duplicate_groups(df)
    n_splits = min(N_SPLITS, groups.nunique())
    y = df["log_price"].to_numpy()
    oof = np.full(len(df), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(df, y, groups):
        n_components = min(photo_features.N_COMPONENTS, len(tr), emb.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        train_pc = pca.fit_transform(emb[tr])
        test_pc = pca.transform(emb[te])
        ecols = [f"img_pc{i}" for i in range(n_components)]

        train_x = df.iloc[tr][base_features].reset_index(drop=True).copy()
        test_x = df.iloc[te][base_features].reset_index(drop=True).copy()
        train_x[ecols] = train_pc
        test_x[ecols] = test_pc

        model = new_model()
        model.fit(Pool(train_x, y[tr], cat_features=cats))
        oof[te] = model.predict(test_x)

    r = regression_metrics(y, oof)
    ape = np.abs(np.exp(oof) - np.exp(y)) / np.exp(y)
    return {"mape": r["mape_pct"], "median": float(np.median(ape) * 100), "r2": r["r2_log"]}


def main():
    df = prepare_training_data(load())
    quality, emb = photo_features.load()

    df = df.merge(quality, on="ad_id", how="inner")
    order = {a: i for i, a in enumerate(quality["ad_id"])}
    emb = emb[[order[a] for a in df["ad_id"]]]
    df = df.reset_index(drop=True)
    print(f"Vehicles with a photo and price: {len(df)}\n")
    if len(df) < 300:
        print(
            "WARNING: the sample is too small; any difference will be lost in noise. Download more photos:"
        )
        print("  python -m kz.collect.photo_fetch --limit 1200")

    qcols = [c for c in quality.columns if c != "ad_id"]
    n_components = min(photo_features.N_COMPONENTS, len(df), emb.shape[1])
    print(f"The embedding is reduced to {n_components} components INSIDE each training fold\n")

    runs = [
        ("tabular features only", FEATURES, CAT_FEATURES),
        ("+ image quality", FEATURES + qcols, CAT_FEATURES),
        ("+ image embedding", None, CAT_FEATURES),
    ]
    base = None
    for label, feats, cats in runs:
        if feats is None:
            r = cv_mape_with_embeddings(df, emb, FEATURES + qcols, cats)
        else:
            r = cv_mape(df, feats, cats)
        delta = "" if base is None else f"   {r['mape'] - base:+.2f} points"
        base = r["mape"] if base is None else base
        print(
            f"  {label:28} MAPE={r['mape']:5.2f}%  median={r['median']:5.2f}%"
            f"  R²={r['r2']:.3f}{delta}"
        )

    print(
        "\nA difference below 0.5 points is noise: fold-to-fold variation is about one point.\n"
        "Only a paired comparison on identical splits makes smaller differences\n"
        "detectable at all."
    )


if __name__ == "__main__":
    main()
