# -*- coding: utf-8 -*-
"""Implementation for the `kz.report.ml_dashboard` module."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from kz.ml.train_price_model import (
    FEATURES,
    grouped_oof_predictions,
    load,
    load_artifact,
    prepare_training_data,
)

OUT_PNG = "data/eda/ml_dashboard.png"
AGE_ORDER = ["0-3", "4-7", "8-12", "13-20", "21+"]

plt.rcParams.update(
    {
        "figure.facecolor": "#12141a",
        "axes.facecolor": "#191c24",
        "axes.edgecolor": "#3a3f4d",
        "axes.labelcolor": "#e6e6e6",
        "text.color": "#e6e6e6",
        "xtick.color": "#aab",
        "ytick.color": "#aab",
        "grid.color": "#2a2e3a",
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "font.size": 10,
    }
)
C_OK, C_BAD, C_ACC = "#4fa3ff", "#ff5d5d", "#ffd166"


def main():
    clean = prepare_training_data(load()).reset_index(drop=True)
    X, y = clean[FEATURES], clean["log_price"]

    oof, baseline_oof, _base_oof = grouped_oof_predictions(clean)
    clean["oof_log"] = oof
    clean["ape"] = np.abs(np.exp(oof) - np.exp(y)) / np.exp(y) * 100

    r2 = r2_score(y, oof)
    mae = mean_absolute_error(np.exp(y), np.exp(oof))
    mape = float(clean["ape"].mean())
    final, _ = load_artifact()
    baseline_mape = float((np.abs(np.exp(baseline_oof) - np.exp(y)) / np.exp(y) * 100).mean())

    print(f"Price model (grouped out-of-fold, {len(X)} clean vehicles):")
    print(f"  R²(log) = {r2:.3f}   MAPE = {mape:.1f}%   MAE = {mae / 1e6:.2f}M ₸")
    print(f"  baseline MAPE = {baseline_mape:.1f}%")

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Price model · R²={r2:.3f} · MAPE={mape:.1f}%  (grouped OOF, {len(X)} vehicles)",
        fontsize=14,
        fontweight="bold",
    )

    a = ax[0, 0]
    a.scatter(np.exp(y), np.exp(oof), s=8, alpha=0.3, color=C_OK)
    lim = [np.exp(y).min(), np.exp(y).max()]
    a.plot(lim, lim, color=C_ACC, ls="--", lw=1.5, label="ideal (estimate = actual)")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_title("Prediction versus actual")
    a.set_xlabel("actual, ₸")
    a.set_ylabel("prediction, ₸")
    a.legend(fontsize=8)

    a = ax[0, 1]
    imp = sorted(zip(FEATURES, final.get_feature_importance()), key=lambda t: t[1])[-12:]
    a.barh([f for f, _ in imp], [v for _, v in imp], color=C_OK, alpha=0.85)
    a.set_title("Feature importance: what moves the price")

    a = ax[1, 0]
    g = clean.groupby("age_bucket")["ape"].mean().reindex(AGE_ORDER)
    a.bar(g.index.astype(str), g.values, color=C_BAD, alpha=0.85)
    a.axhline(mape, color=C_ACC, ls="--", lw=1.2, label=f"mean {mape:.0f}%")
    a.set_title("MAPE by vehicle age")
    a.set_xlabel("vehicle age, years")
    a.set_ylabel("MAPE, %")
    a.legend(fontsize=8)

    a = ax[1, 1]
    resid = y.values - oof
    a.hist(resid, bins=60, color=C_OK, alpha=0.85)
    a.axvline(0, color=C_ACC, ls="--", lw=1.2)
    a.set_title("Residual log(actual) − log(prediction): 0 = exact")
    a.set_xlabel("log residual")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nDashboard → {OUT_PNG}")


if __name__ == "__main__":
    main()
