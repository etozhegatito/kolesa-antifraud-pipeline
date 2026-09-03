# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.learning_curve` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    duplicate_groups,
    load,
    new_model,
    prepare_training_data,
    regression_metrics,
)

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
N_SPLITS = 5
OUT_PNG = Path("data/eda/learning_curve.png")
RANDOM_SEED = 42


def subsample_by_groups(df: pd.DataFrame, groups: pd.Series, frac: float, seed: int = RANDOM_SEED):
    """Implement `subsample_by_groups`."""
    if frac >= 1.0:
        return df, groups
    uniq = pd.Series(groups.unique())
    keep = uniq.sample(frac=frac, random_state=seed)
    mask = groups.isin(set(keep))
    return df[mask], groups[mask]


def cv_mape(df: pd.DataFrame, groups: pd.Series, n_splits: int = N_SPLITS) -> dict:
    """Implement `cv_mape`."""
    from catboost import Pool
    from sklearn.model_selection import GroupKFold

    n = min(n_splits, groups.nunique())
    if n < 2:
        raise ValueError("Too few independent groups")
    X, y = df[FEATURES], df["log_price"]
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        model = new_model()
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        oof[te] = model.predict(X.iloc[te])
    return regression_metrics(y, oof)


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["rows"] for r in rows]
    mape = [r["mape_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.plot(n, mape, "o-", color="#2563c9", lw=2)
    for x, y in zip(n, mape):
        ax.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9
        )
    ax.set_xlabel("training rows")
    ax.set_ylabel("MAPE, % (grouped CV)")
    ax.set_title("Learning curve: does error fall as data grows?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    print(f"\nChart → {OUT_PNG}")


def main():
    df = prepare_training_data(load())
    groups_all = duplicate_groups(df)
    print(f"Total clean rows: {len(df)}, independent groups: {groups_all.nunique()}\n")

    rows = []
    for frac in FRACTIONS:
        part, groups = subsample_by_groups(df, groups_all, frac)
        m = cv_mape(part, groups)
        rows.append({"frac": frac, "rows": len(part), **m})
        print(
            f"  {frac * 100:5.0f}%  rows={len(part):5d}  "
            f"MAPE={m['mape_pct']:5.2f}%  R²(log)={m['r2_log']:.3f}  "
            f"MAE={m['mae_tenge'] / 1e6:.2f}M"
        )

    last_gain = rows[-2]["mape_pct"] - rows[-1]["mape_pct"]
    first_gain = rows[0]["mape_pct"] - rows[1]["mape_pct"]
    print(f"\nGain from the first data increase: {first_gain:+.2f} MAPE points")
    print(f"Gain from the last data increase:   {last_gain:+.2f} MAPE points")
    if abs(last_gain) < 0.5:
        print("→ The curve has plateaued: more rows of the SAME kind add little.")
        print("  Improve feature signal—trim, condition, and photos—not only volume.")
    else:
        print("→ Error is still falling; additional collection remains useful.")
    plot(rows)


if __name__ == "__main__":
    main()
