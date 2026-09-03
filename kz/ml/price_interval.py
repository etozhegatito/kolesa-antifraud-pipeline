# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.price_interval` module."""

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
from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    code_fingerprint,
    coerce_features,
    duplicate_groups,
    load,
    new_model,
    prepare_training_data,
)

TARGET_COVERAGE = 0.80
N_SPLITS = 5


#


GROUP_EDGES = [0, 5e6, 10e6, 20e6, float("inf")]
GROUP_NAMES = ["<5M", "5-10M", "10-20M", "20M+"]


MIN_GROUP = 200

MODEL_DIR = Path("data/models")
LOWER_PATH = MODEL_DIR / "price_interval_lower.cbm"
UPPER_PATH = MODEL_DIR / "price_interval_upper.cbm"
META_PATH = MODEL_DIR / "price_interval.metadata.json"
SCHEMA_VERSION = 2


def quantile_levels(target: float = TARGET_COVERAGE) -> tuple[float, float]:
    """Implement `quantile_levels`."""
    tail = (1.0 - target) / 2.0
    return tail, 1.0 - tail


def conformity(y_log, lo_log, hi_log) -> np.ndarray:
    """Implement `conformity`."""
    y = np.asarray(y_log, dtype=float)
    return np.maximum(np.asarray(lo_log, dtype=float) - y, y - np.asarray(hi_log, dtype=float))


def conformal_offset(scores: np.ndarray, target: float = TARGET_COVERAGE) -> float:
    """Implement `conformal_offset`."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, target * (n + 1) / n)
    return float(np.quantile(scores, level, method="higher"))


def tail_offsets(y_log, lo_log, hi_log, target: float = TARGET_COVERAGE) -> tuple[float, float]:
    """Implement `tail_offsets`."""
    y = np.asarray(y_log, dtype=float)
    tail = (1.0 - target) / 2.0

    def edge(scores: np.ndarray) -> float:
        scores = scores[np.isfinite(scores)]
        n = len(scores)
        if n == 0:
            return 0.0
        level = min(1.0, (1.0 - tail) * (n + 1) / n)
        return float(np.quantile(scores, level, method="higher"))

    return edge(np.asarray(lo_log, dtype=float) - y), edge(y - np.asarray(hi_log, dtype=float))


def group_of(price: np.ndarray) -> np.ndarray:
    """Implement `group_of`."""
    return np.clip(
        np.searchsorted(GROUP_EDGES, np.asarray(price, dtype=float), side="right") - 1,
        0,
        len(GROUP_NAMES) - 1,
    )


def group_offsets(y_log, lo_log, hi_log, pred_price, target: float = TARGET_COVERAGE) -> dict:
    """Implement `group_offsets`."""
    g = group_of(pred_price)
    fallback = tail_offsets(y_log, lo_log, hi_log, target)
    out = {"global": list(fallback), "groups": {}}
    y = np.asarray(y_log, dtype=float)
    for i, name in enumerate(GROUP_NAMES):
        m = g == i
        if m.sum() < MIN_GROUP:
            out["groups"][name] = {
                "offsets": list(fallback),
                "n": int(m.sum()),
                "source": "global (group too small)",
            }
            continue
        out["groups"][name] = {
            "offsets": list(
                tail_offsets(y[m], np.asarray(lo_log)[m], np.asarray(hi_log)[m], target)
            ),
            "n": int(m.sum()),
            "source": "group-specific",
        }
    return out


def apply_offsets(lo_log, hi_log, offsets: dict):
    """Implement `apply_offsets`."""
    lo_log = np.asarray(lo_log, dtype=float)
    hi_log = np.asarray(hi_log, dtype=float)
    g = group_of(np.exp((lo_log + hi_log) / 2))
    d_lo = np.empty(len(lo_log))
    d_hi = np.empty(len(hi_log))
    for i, name in enumerate(GROUP_NAMES):
        pair = offsets["groups"].get(name, {}).get("offsets", offsets["global"])
        m = g == i
        d_lo[m], d_hi[m] = pair[0], pair[1]
    return lo_log - d_lo, hi_log + d_hi


def oof_bounds(clean: pd.DataFrame, target: float = TARGET_COVERAGE):
    """Implement `oof_bounds`."""
    lo_a, hi_a = quantile_levels(target)
    groups = duplicate_groups(clean)
    n = min(N_SPLITS, groups.nunique())
    if n < 2:
        raise ValueError("Not enough independent groups for calibration")

    lo = np.full(len(clean), np.nan)
    hi = np.full(len(clean), np.nan)
    X, y = clean[FEATURES], clean["log_price"]
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES)
        m_lo = new_model(loss_function=f"Quantile:alpha={lo_a}")
        m_lo.fit(pool)
        m_hi = new_model(loss_function=f"Quantile:alpha={hi_a}")
        m_hi.fit(pool)
        lo[te] = m_lo.predict(X.iloc[te])
        hi[te] = m_hi.predict(X.iloc[te])
    return lo, hi


def coverage_report(y_log, lo_log, hi_log, price) -> dict:
    """Implement `coverage_report`."""
    y = np.asarray(y_log, dtype=float)
    inside = (y >= lo_log) & (y <= hi_log)
    low, high = np.exp(lo_log), np.exp(hi_log)
    width = (high - low) / np.asarray(price, dtype=float)
    return {
        "coverage": float(inside.mean()),
        "median_width_pct": float(np.median(width) * 100),
        "mean_width_pct": float(np.mean(width) * 100),
        "below": float((y < lo_log).mean()),
        "above": float((y > hi_log).mean()),
    }


def fit(clean: pd.DataFrame, target: float = TARGET_COVERAGE, log=print):
    """Implement `fit`."""
    lo_oof, hi_oof = oof_bounds(clean, target)
    y, price = clean["log_price"], clean["price_tenge"]

    raw = coverage_report(y, lo_oof, hi_oof, price)
    log(
        f"Uncalibrated:       coverage {raw['coverage'] * 100:.1f}% "
        f"(target {target * 100:.0f}%), width {raw['median_width_pct']:.0f}%"
    )

    sym = conformal_offset(conformity(y, lo_oof, hi_oof), target)
    sym_rep = coverage_report(y, lo_oof - sym, hi_oof + sym, price)
    log(
        f"Global correction:  coverage {sym_rep['coverage'] * 100:.1f}%, "
        f"width {sym_rep['median_width_pct']:.0f}%"
    )

    mid_price = np.exp((lo_oof + hi_oof) / 2)
    offsets = group_offsets(y, lo_oof, hi_oof, mid_price, target)
    lo_cal, hi_cal = apply_offsets(lo_oof, hi_oof, offsets)
    fixed = coverage_report(y, lo_cal, hi_cal, price)
    log(
        f"Grouped correction: coverage {fixed['coverage'] * 100:.1f}%, "
        f"width {fixed['median_width_pct']:.0f}%\n"
    )

    log("Group corrections in log space (down / up):")
    for name, info in offsets["groups"].items():
        d_lo, d_hi = info["offsets"]
        log(f"  {name:<7} n={info['n']:<5} down {d_lo:+.3f}  up {d_hi:+.3f}   {info['source']}")

    lo_a, hi_a = quantile_levels(target)
    pool = Pool(clean[FEATURES], clean["log_price"], cat_features=CAT_FEATURES)
    final_lo = new_model(loss_function=f"Quantile:alpha={lo_a}")
    final_lo.fit(pool)
    final_hi = new_model(loss_function=f"Quantile:alpha={hi_a}")
    final_hi.fit(pool)
    return final_lo, final_hi, offsets, (lo_cal, hi_cal), fixed, sym_rep


def _save_model(model: CatBoostRegressor, path: Path) -> None:
    """Implement `_save_model`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem, suffix=".cbm", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        model.save_model(str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_artifact(lower, upper, metadata: dict) -> None:
    _save_model(lower, LOWER_PATH)
    _save_model(upper, UPPER_PATH)
    tmp = META_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, META_PATH)


def load_artifact():
    if not (LOWER_PATH.exists() and UPPER_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError(
            "Price-interval artifact is missing. Run: python -m kz.ml.price_interval"
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible price-interval artifact version")
    if meta.get("features") != FEATURES:
        raise ValueError("Price-interval feature schema does not match the code")
    lo, hi = CatBoostRegressor(), CatBoostRegressor()
    lo.load_model(str(LOWER_PATH))
    hi.load_model(str(UPPER_PATH))
    return lo, hi, meta


def predict_interval(X: pd.DataFrame, models=None) -> tuple[np.ndarray, np.ndarray]:
    """Implement `predict_interval`."""
    lo, hi, meta = models or load_artifact()
    prepared = coerce_features(X)[FEATURES]
    lo_log, hi_log = apply_offsets(lo.predict(prepared), hi.predict(prepared), meta["offsets"])
    low, high = np.exp(lo_log), np.exp(hi_log)
    return np.minimum(low, high), np.maximum(low, high)


def by_segment(clean: pd.DataFrame, lo_log, hi_log, log=print) -> dict:
    """Implement `by_segment`."""
    price = clean["price_tenge"].to_numpy(dtype=float)
    y = clean["log_price"].to_numpy()
    predicted = np.exp((np.asarray(lo_log) + np.asarray(hi_log)) / 2)

    out = {}
    for label, key, tag in [
        ("by predicted price", predicted, "predicted"),
        ("by actual price", price, "actual"),
    ]:
        log(f"\nCoverage {label} (target {TARGET_COVERAGE * 100:.0f}%):")
        g = group_of(key)
        out[tag] = {}
        for i, name in enumerate(GROUP_NAMES):
            m = g == i
            if m.sum() < 20:
                continue
            r = coverage_report(y[m], np.asarray(lo_log)[m], np.asarray(hi_log)[m], price[m])
            out[tag][name] = {"n": int(m.sum()), **r}
            skew = (r["below"] - r["above"]) * 100
            log(
                f"  {name:<7} n={m.sum():<5} coverage {r['coverage'] * 100:5.1f}%   "
                f"width {r['median_width_pct']:5.0f}%   "
                f"below {r['below'] * 100:4.1f}%  above {r['above'] * 100:4.1f}%"
                f"   skew {skew:+5.1f}"
            )
    log("\n  Balanced tails in the first view indicate useful calibration.")
    log("  Skew in the second view is unavoidable because grouping by the")
    log("  actual target conditions on what the model predicts; see by_segment.")
    return out


def main():
    clean = prepare_training_data(load()).reset_index(drop=True)
    print(f"Calibration rows: {len(clean)}   coverage target: {TARGET_COVERAGE * 100:.0f}%\n")

    lower, upper, offsets, (lo_cal, hi_cal), overall, sym = fit(clean)
    segments = by_segment(clean, lo_cal, hi_cal)

    crossed = int((lo_cal > hi_cal).sum())
    print(
        f"\nCrossed bounds (lower above upper): {crossed} of "
        f"{len(clean)} ({crossed / len(clean) * 100:.2f}%). Endpoints are "
        f"reordered at inference."
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_sha256": code_fingerprint(__file__, _tpm.__file__),
        "features": FEATURES,
        "target_coverage": TARGET_COVERAGE,
        "quantile_levels": list(quantile_levels()),
        "calibration": "mondrian_cqr_asymmetric_tails_grouped_oof",
        "calibration_rows": int(len(clean)),
        "group_edges": GROUP_EDGES[1:-1],
        "min_group": MIN_GROUP,
        "offsets": offsets,
        "crossed_bounds": crossed,
        "oof": overall,
        "oof_symmetric_global": sym,
        "segments": segments,
    }
    save_artifact(lower, upper, metadata)

    print(
        f"\nResult: the interval covers {overall['coverage'] * 100:.1f}% of vehicles "
        f"against a {TARGET_COVERAGE * 100:.0f}% target, with median width "
        f"{overall['median_width_pct']:.0f}% of price."
    )
    print(
        "Coverage cannot be improved by merely shifting point predictions; "
        "that is why it is evaluated separately from MAPE."
    )
    print(f"\nArtifacts → {LOWER_PATH.name}, {UPPER_PATH.name}, {META_PATH.name}")


if __name__ == "__main__":
    main()
