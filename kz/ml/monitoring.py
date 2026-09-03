# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.monitoring` module."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path("data/eda/monitoring_history.csv")
N_BINS = 10
PSI_WATCH, PSI_ALERT = 0.10, 0.25

HISTORY_COLS = [
    "checked_at",
    "training_rows",
    "current_rows",
    "max_psi",
    "max_psi_feature",
    "n_watch",
    "n_alert",
    "model_created",
    "model_mape_pct",
]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = N_BINS) -> float:
    """Implement `psi`."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) < bins:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Implement `categorical_psi`."""
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    eps = 1e-6
    ev = np.clip(e.reindex(idx).fillna(0).to_numpy(), eps, None)
    av = np.clip(a.reindex(idx).fillna(0).to_numpy(), eps, None)
    return float(np.sum((av - ev) * np.log(av / ev)))


def level(value: float) -> str:
    if not np.isfinite(value):
        return "no data"
    if value > PSI_ALERT:
        return "MAJOR shift"
    if value > PSI_WATCH:
        return "noticeable shift"
    return "stable"


def compare(training: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Implement `compare`."""
    from kz.ml.train_price_model import CAT_FEATURES, NUM_FEATURES

    rows = []
    for f in NUM_FEATURES:
        if f in training.columns and f in current.columns:
            rows.append(
                {
                    "feature": f,
                    "type": "numeric",
                    "psi": psi(training[f].to_numpy(), current[f].to_numpy()),
                }
            )
    for f in CAT_FEATURES:
        if f in training.columns and f in current.columns:
            rows.append(
                {
                    "feature": f,
                    "type": "categorical",
                    "psi": categorical_psi(training[f], current[f]),
                }
            )
    out = pd.DataFrame(rows)
    out["level"] = out["psi"].map(level)
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def training_snapshot() -> pd.DataFrame:
    """Implement `training_snapshot`."""
    from kz.ml.train_price_model import load, load_artifact, prepare_training_data

    _, meta = load_artifact()
    n = int(meta.get("training_rows", 0))
    df = prepare_training_data(load())
    if "scraped_at" in df.columns:
        df = df.sort_values("scraped_at")
    return df.head(n) if 0 < n < len(df) else df


def append_history(row: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    fresh = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        if fresh:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in HISTORY_COLS})


def main():
    from kz.ml.train_price_model import load, load_artifact, prepare_training_data

    _, meta = load_artifact()
    current = prepare_training_data(load())
    train = training_snapshot()
    trained_on = int(meta.get("training_rows", 0))
    fresh_rows = len(current) - trained_on

    print(f"Model trained at {meta.get('created_at_utc', '?')[:16]} on {trained_on} rows")
    print(f"Current clean layer: {len(current)} rows ({fresh_rows:+d})\n")

    if fresh_rows <= 0:
        print("Nothing to compare: the model was trained on the current data.")
        print("Drift compares training data with NEW observations, so run this")
        print("check before retraining rather than after it.")
        print("\nNo measurement was appended to history; a zero here is not evidence.")
        return

    table = compare(train, current)
    print("Feature-distribution shift (PSI):")
    for _, r in table.iterrows():
        mark = "  ←" if r["psi"] > PSI_WATCH else ""
        print(f"  {r['feature']:20} {r['psi']:6.3f}   {r['level']}{mark}")

    n_watch = int((table["psi"] > PSI_WATCH).sum())
    n_alert = int((table["psi"] > PSI_ALERT).sum())
    top = table.iloc[0]

    val = meta.get("validation", {}).get("grouped_cv", {}).get("model", {})
    append_history(
        {
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_rows": meta.get("training_rows"),
            "current_rows": len(current),
            "max_psi": round(float(top["psi"]), 4),
            "max_psi_feature": top["feature"],
            "n_watch": n_watch,
            "n_alert": n_alert,
            "model_created": meta.get("created_at_utc", "")[:16],
            "model_mape_pct": round(float(val.get("mape_pct", float("nan"))), 2),
        }
    )

    print(f"\nSummary: {n_watch} features show noticeable shift; {n_alert} show major shift.")
    if n_alert:
        print("→ Retraining is due: python -m kz.ops.run_all --ml")
    elif n_watch:
        print("→ Still acceptable, but continue monitoring.")
    else:
        print("→ Data are stable relative to the training sample.")
    print(f"\nMonitoring history → {HISTORY}")
    print(
        "History reveals a trend: one measurement cannot separate random "
        "variation from gradual divergence."
    )


if __name__ == "__main__":
    main()
