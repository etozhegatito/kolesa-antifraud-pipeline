# -*- coding: utf-8 -*-
"""Implementation for the `kz.report.evaluate_detector` module."""

import pathlib as _p

_expected = "evaluate_detector.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import math
from pathlib import Path

import pandas as pd

from kz.core.db import get_engine

LABELS_CSV = "data/manual_labels.csv"
LINE = "─" * 64
MIN_FOR_METRICS = 20


def load_labeled() -> pd.DataFrame:
    """Implement `load_labeled`."""
    clean = pd.read_sql(
        "SELECT ad_id, is_suspicious, suspicion_reasons FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    if not Path(LABELS_CSV).exists():
        return pd.DataFrame()
    lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
    lab["verdict"] = lab["verdict"].astype("string").str.strip().str.lower()

    lab = lab.drop_duplicates("ad_id", keep="last")
    lab = lab[lab["verdict"].isin(["fraud", "legit"])]
    optional = [
        c
        for c in ["sampling_stratum", "stratum_population", "stratum_sample_size"]
        if c in lab.columns
    ]
    out = clean.merge(lab[["ad_id", "verdict", *optional]], on="ad_id", how="inner")

    queue = Path("data/eda/labeling_queue.csv")
    if queue.exists() and "sampling_stratum" not in out.columns:
        q = pd.read_csv(queue, dtype={"ad_id": str})
        keep = [
            c
            for c in ["ad_id", "sampling_stratum", "stratum_population", "stratum_sample_size"]
            if c in q.columns
        ]
        out = out.merge(q[keep], on="ad_id", how="left")
    return out


def confusion(df: pd.DataFrame) -> dict:
    is_fraud = df["verdict"] == "fraud"
    flagged = df["is_suspicious"] == 1
    return {
        "TP": int((is_fraud & flagged).sum()),
        "FP": int((~is_fraud & flagged).sum()),
        "FN": int((is_fraud & ~flagged).sum()),
        "TN": int((~is_fraud & ~flagged).sum()),
    }


def weighted_confusion(df: pd.DataFrame) -> dict | None:
    """Implement `weighted_confusion`."""
    required = {"stratum_population", "stratum_sample_size"}
    if not required.issubset(df.columns):
        return None
    population = pd.to_numeric(df["stratum_population"], errors="coerce")
    sample = pd.to_numeric(df["stratum_sample_size"], errors="coerce")
    if population.isna().any() or sample.isna().any() or (sample <= 0).any():
        return None
    weight = population / sample
    is_fraud = df["verdict"] == "fraud"
    flagged = df["is_suspicious"] == 1
    return {
        "TP": float(weight[is_fraud & flagged].sum()),
        "FP": float(weight[~is_fraud & flagged].sum()),
        "FN": float(weight[is_fraud & ~flagged].sum()),
        "TN": float(weight[~is_fraud & ~flagged].sum()),
    }


def control_bound_report(df: pd.DataFrame) -> str:
    """Implement `control_bound_report`."""
    ctrl = (
        df[df.get("sampling_stratum") == "random_control"]
        if "sampling_stratum" in df.columns
        else df.iloc[0:0]
    )
    n_ctrl = len(ctrl)
    lines = ["\n► What zero confirmed fraud means"]
    if n_ctrl == 0:
        lines.append("  The control stratum is unlabeled, so missed cases cannot be estimated.")
        return "\n".join(lines)

    bound = 3 / n_ctrl
    pop = ctrl["stratum_population"].iloc[0] if "stratum_population" in ctrl else float("nan")
    lines.append(f"  The control sample contains {n_ctrl} listings and no confirmed fraud.")
    lines.append(
        f"  Rule of three: the population fraud rate is below {bound:.1%} "
        f"with approximately 95% confidence."
    )
    if pop == pop:
        lines.append(
            f"  Applied to {int(pop)} unflagged listings, this is at most "
            f"{int(bound * pop)} fraud cases; the point estimate is zero."
        )
    lines.append("  Interpretation: the sample gives the detector little fraud to catch.")
    lines.append("  Low precision can therefore mean the rules found explainable anomalies,")
    lines.append("  not that the rule implementation is necessarily defective.")
    return "\n".join(lines)


def _prf(c: dict) -> tuple[float, float, float]:
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if math.isfinite(precision) and math.isfinite(recall):
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    else:
        f1 = float("nan")
    return precision, recall, f1


def main():
    df = load_labeled()

    print(LINE)
    print("DETECTOR QUALITY (is_suspicious versus manual verdicts)")
    print(LINE)

    if df.empty:
        print("\nNo manual verdicts are available, so metrics cannot be computed.")
        print("Next steps:")
        print("  1. open data/eda/labeling_queue.csv (highest-priority rows first)")
        print("  2. inspect each URL and set verdict = fraud / legit / unknown")
        print("  3. copy labeled rows to data/manual_labels.csv")
        print("  4. run clean.py to merge verdicts, then run this report again")
        print(
            f"\nA stable estimate requires at least {MIN_FOR_METRICS} "
            "fraud/legit verdicts (currently 0)."
        )
        print(LINE)
        return

    n = len(df)
    n_fraud = int((df["verdict"] == "fraud").sum())
    c = confusion(df)
    precision, recall, f1 = _prf(c)

    print(f"\nLabeled: {n} (fraud: {n_fraud}, legit: {n - n_fraud})")
    if n < MIN_FOR_METRICS:
        print(
            f"WARNING: too few for a stable estimate (need at least {MIN_FOR_METRICS}); "
            "the following values are provisional"
        )
    print("  Recall applies only to the labeled sample. Label random_control rows")
    print("  from labeling_queue.csv to estimate misses in the wider population.")

    print("\n► Confusion matrix")
    print("                 flagged   not flagged")
    print(f"  actual fraud      {c['TP']:>4}       {c['FN']:>4}   ← FN = missed fraud")
    print(f"  actual legit      {c['FP']:>4}       {c['TN']:>4}   ← FP = false alarm")

    print("\n► Metrics")
    print(f"  precision = {precision:.1%}   (share of flagged rows that are fraud)")
    print(f"  recall    = {recall:.1%}   (share of labeled fraud that was caught)")
    print(f"  F1        = {f1:.1%}   (harmonic balance of precision and recall)")

    if n_fraud == 0:
        print(control_bound_report(df))

    weighted = weighted_confusion(df)
    if weighted is not None:
        wp, wr, wf = _prf(weighted)
        print("\n► Population estimate weighted by sampling strata")
        print(f"  precision = {wp:.1%}   recall = {wr:.1%}   F1 = {wf:.1%}")
        print("  This extrapolates to the full snapshot with inverse-probability weights;")
        print("  it is not the raw metric of an enriched review queue.")
    else:
        print("\n  Population estimate is unavailable: legacy verdicts lack the sampling")
        print("  metadata introduced by the new three-stratum queue.")

    print("\n► Precision by rule (labeled rows only)")
    flagged = df[df["is_suspicious"] == 1].copy()
    if flagged.empty:
        print("  (no labeled listing was flagged)")
    else:
        rows = []
        reasons = flagged.assign(r=flagged["suspicion_reasons"].str.split("|")).explode("r")
        for reason, g in reasons.groupby("r"):
            if not reason:
                continue
            fr = int((g["verdict"] == "fraud").sum())
            rows.append((reason, fr, len(g), fr / len(g)))
        for reason, fr, tot, p in sorted(rows, key=lambda x: -x[3]):
            print(f"  {reason:<24} {fr}/{tot}  precision={p:.0%}")
    print(LINE)


if __name__ == "__main__":
    main()
