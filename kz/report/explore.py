# -*- coding: utf-8 -*-
"""Implementation for the `kz.report.explore` module."""

import pathlib as _p

_expected = "explore.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import csv

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kz.core.db import get_engine

OUT_PNG = "data/eda/dashboard.png"
OUT_SUSP = "data/eda/suspicious_sorted.csv"
CONTROL_SAMPLE_SIZE = 50


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
C_OK, C_BAD, C_ACC, C_INFO = "#4fa3ff", "#ff5d5d", "#ffd166", "#9b7bff"


def add_iqr_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `add_iqr_flags`."""
    df = df.copy()

    def fences(s: pd.Series) -> pd.DataFrame:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return pd.DataFrame(
            {
                "iqr_low": q1 - 1.5 * iqr,
                "iqr_high": q3 + 1.5 * iqr,
            },
            index=s.index,
        )

    g = df.groupby("age_bucket", observed=True)["log_price"]
    df[["iqr_low", "iqr_high"]] = g.apply(fences).reset_index(level=0, drop=True)
    df["iqr_outlier"] = np.select(
        [df["log_price"] < df["iqr_low"], df["log_price"] > df["iqr_high"]],
        ["low", "high"],
        default="",
    )

    df["both_detectors_low"] = (
        df["suspicion_reasons"].str.contains("price_anomaly_low", na=False)
        & (df["iqr_outlier"] == "low")
    ).astype(int)
    return df


def _write_csv(path: str, df: pd.DataFrame) -> None:
    """Implement `_write_csv`."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(df.columns))
        writer.writeheader()
        for row in df.to_dict("records"):
            writer.writerow({k: ("" if pd.isna(v) else v) for k, v in row.items()})


def select_labeling_rows(
    df: pd.DataFrame,
    residual_mask: pd.Series | None = None,
    control_n: int = CONTROL_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Implement `select_labeling_rows`."""
    work = df.copy()
    residual_mask = (
        residual_mask.reindex(work.index, fill_value=False).astype(bool)
        if residual_mask is not None
        else pd.Series(False, index=work.index)
    )
    rule = work["is_suspicious"].eq(1)
    residual = (~rule) & residual_mask
    control = (~rule) & (~residual)

    work["sampling_stratum"] = np.select(
        [rule, residual],
        ["rule_positive", "residual_candidate"],
        default="random_control",
    )
    population = work["sampling_stratum"].value_counts().to_dict()

    if "verdict" in work.columns:
        verdict = work["verdict"].astype("string").str.strip().str.lower()
        unresolved = ~verdict.isin(["fraud", "legit", "unknown"])
    else:
        unresolved = pd.Series(True, index=work.index)

    pos = work[unresolved & rule].sort_values(
        ["both_detectors_low", "price_z"], ascending=[False, True]
    )
    res = work[unresolved & residual].sort_values("residual_gap", ascending=False)
    controls = work[unresolved & control].copy()

    controls["_sample_key"] = pd.util.hash_pandas_object(controls["ad_id"].astype(str), index=False)
    controls = controls.nsmallest(min(control_n, len(controls)), "_sample_key")
    controls = controls.drop(columns="_sample_key")

    q = pd.concat([pos, res, controls], ignore_index=True)
    q["stratum_population"] = q["sampling_stratum"].map(population).astype(int)
    sample_counts = q["sampling_stratum"].value_counts().to_dict()
    q["stratum_sample_size"] = q["sampling_stratum"].map(sample_counts).astype(int)
    return q


def export_labeling_queue(df: pd.DataFrame):
    """Implement `export_labeling_queue`."""
    work = df.copy()
    work["residual_gap"] = np.nan
    residual_mask = pd.Series(False, index=work.index)
    try:
        from kz.ml.residual_detector import (
            AGE_MAX,
            MIN_SUPPORT,
            load_floor_artifact,
            score_floor,
        )
        from kz.ml.train_price_model import FEATURES
        from kz.transform.price_basis import is_training_eligible

        model, metadata = load_floor_artifact()
        floor = score_floor(model, metadata, work[FEATURES])
        work["residual_gap"] = floor - np.log(work["price_tenge"])
        eligible = work.get("price_basis", pd.Series("ambiguous", index=work.index)).map(
            is_training_eligible
        )
        clean = work[(work["is_suspicious"] == 0) & eligible]
        support = clean.groupby(["brand", "model"]).size()
        sup = pd.Series(
            [int(support.get((b, m), 0)) for b, m in zip(work["brand"], work["model"])],
            index=work.index,
        )
        residual_mask = (
            work["residual_gap"].gt(0) & sup.ge(MIN_SUPPORT) & work["age"].le(AGE_MAX) & eligible
        )
    except FileNotFoundError:
        pass

    q = select_labeling_rows(work, residual_mask)
    cols = [
        "sampling_stratum",
        "stratum_population",
        "stratum_sample_size",
        "ad_id",
        "url",
        "title",
        "year",
        "price_tenge",
        "mileage_km",
        "price_z",
        "residual_gap",
        "suspicion_reasons",
    ]
    for extra in [
        "price_basis",
        "customs_cleared",
        "steering",
        "damage_keywords",
        "seller_comment",
    ]:
        if extra in q.columns:
            cols.append(extra)
    q = q[cols]
    if "seller_comment" in q.columns:
        q["seller_comment"] = q["seller_comment"].fillna("").str[:150]
    q["verdict"] = ""
    q["comment"] = ""
    out = "data/eda/labeling_queue.csv"
    _write_csv(out, q)
    counts = q["sampling_stratum"].value_counts()
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"Labeling queue ({len(q)}: {summary}) → {out}")


def console_report(df: pd.DataFrame):
    line = "─" * 72
    print(line)
    print(
        f"Listings: {len(df)}   Suspicious: {df.is_suspicious.sum()} "
        f"({df.is_suspicious.mean():.1%})   "
        f"Two-detector agreement: {df.both_detectors_low.sum()}"
    )
    print(line)

    print("\n► Suspicion reasons:")
    reasons = (
        df.loc[df.is_suspicious == 1, "suspicion_reasons"].str.split("|").explode().value_counts()
    )
    print(reasons.to_string())

    if "info_flags" in df.columns:
        info = (
            df["info_flags"]
            .fillna("")
            .replace("", np.nan)
            .dropna()
            .str.split("|")
            .explode()
            .value_counts()
        )
        if len(info):
            print("\n► Informational flags, including exculpatory context:")
            print(info.to_string())

    if "customs_cleared" in df.columns:
        enr_n = df["customs_cleared"].notna().sum()
        customs_not_cleared = df["customs_cleared"].eq("Нет").sum()
        right_hand_drive = df.get("steering", pd.Series()).eq("Справа").sum()
        damage_terms = (df.get("damage_keywords", pd.Series()).fillna("") != "").sum()
        print(
            f"\n► Enrichment: {enr_n}/{len(df)} "
            f"({enr_n / len(df):.0%}); customs clearance 'No': "
            f"{customs_not_cleared}, right-hand drive: {right_hand_drive}, "
            f"damage terms: {damage_terms}"
        )

    print("\n► Top 15 unusually cheap listings, ordered by price_z:")
    cols = [
        "ad_id",
        "title",
        "year",
        "price_tenge",
        "mileage_km",
        "price_z",
        "z_group_level",
        "iqr_outlier",
        "views_count",
        "url",
    ]
    susp = df[df.is_suspicious == 1].sort_values("price_z").head(15)
    print(susp[cols].to_string(index=False))

    print("\n► Median price by age bucket:")
    med = (
        df.groupby("age_bucket", observed=True)["price_tenge"].agg(["count", "median"]).astype(int)
    )
    print(med.to_string())
    print(line)


def build_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(
        "Kolesa.kz · Almaty — data quality and anomalies", fontsize=15, fontweight="bold", y=0.99
    )

    ok = df[df.is_suspicious == 0]
    bad = df[df.is_suspicious == 1]

    ax = axes[0, 0]
    ax.hist(
        df.price_tenge,
        bins=np.logspace(np.log10(df.price_tenge.min()), np.log10(df.price_tenge.max()), 40),
        color=C_OK,
        alpha=0.85,
    )
    ax.set_xscale("log")
    ax.axvline(
        df.price_tenge.median(),
        color=C_ACC,
        ls="--",
        lw=1.5,
        label=f"median {df.price_tenge.median() / 1e6:.1f}M",
    )
    ax.axvline(
        df.price_tenge.mean(),
        color=C_BAD,
        ls="--",
        lw=1.5,
        label=f"mean {df.price_tenge.mean() / 1e6:.1f}M",
    )
    ax.set_title("Prices on log scale: mean above median indicates a right tail")
    ax.set_xlabel("price, ₸")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    order = ["0-3", "4-7", "8-12", "13-20", "21+"]
    data = [df.loc[df.age_bucket == b, "log_price"].dropna() for b in order]
    bp = ax.boxplot(
        data,
        tick_labels=order,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=3, markerfacecolor=C_BAD, alpha=0.6),
    )
    for box in bp["boxes"]:
        box.set(facecolor=C_OK, alpha=0.55)
    for med_l in bp["medians"]:
        med_l.set(color=C_ACC, lw=2)
    ax.set_title("log(price) by age: box = IQR, points = outliers")
    ax.set_xlabel("vehicle age, years")
    ax.set_ylabel("ln(price)")

    ax = axes[0, 2]
    ax.scatter(ok.year, ok.price_tenge, s=12, alpha=0.45, color=C_OK, label="clean")
    ax.scatter(
        bad.year, bad.price_tenge, s=34, alpha=0.95, color=C_BAD, marker="x", label="suspicious"
    )
    agree = df[df.both_detectors_low == 1]
    ax.scatter(
        agree.year,
        agree.price_tenge,
        s=130,
        facecolors="none",
        edgecolors=C_ACC,
        lw=1.6,
        label="both detectors",
    )
    ax.set_yscale("log")
    ax.set_title("Model year × price")
    ax.set_xlabel("year")
    ax.set_ylabel("price, ₸ (log)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    reasons = (
        df.loc[df.is_suspicious == 1, "suspicion_reasons"].str.split("|").explode().value_counts()
    )
    ax.barh(reasons.index[::-1], reasons.values[::-1], color=C_BAD, alpha=0.85)
    ax.set_title("Suspicion reasons")

    ax = axes[1, 1]
    na = (
        df[["mileage_km", "description", "body_type", "condition", "labels", "engine_volume"]]
        .isna()
        .mean()
        * 100
    ).sort_values()
    ax.barh(na.index, na.values, color=C_INFO, alpha=0.85)
    ax.set_title("Missing values, % (mileage is MNAR)")
    ax.set_xlabel("%")

    ax = axes[1, 2]
    okm, badm = ok.dropna(subset=["mileage_km"]), bad.dropna(subset=["mileage_km"])
    ax.scatter(okm.mileage_km, okm.price_tenge, s=12, alpha=0.4, color=C_OK)
    ax.scatter(badm.mileage_km, badm.price_tenge, s=34, alpha=0.95, color=C_BAD, marker="x")
    ax.set_xscale("symlog")
    ax.set_yscale("log")
    ax.set_title("Mileage × price (suspicious = ✕)")
    ax.set_xlabel("mileage, km")
    ax.set_ylabel("price, ₸")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nDashboard → {OUT_PNG}")


def main():
    df = pd.read_sql("SELECT * FROM clean_data", get_engine())

    age_order = ["0-3", "4-7", "8-12", "13-20", "21+"]
    df["age_bucket"] = pd.Categorical(df["age_bucket"], categories=age_order, ordered=True)
    df = add_iqr_flags(df)
    console_report(df)

    susp = df[df.is_suspicious == 1].sort_values(
        ["both_detectors_low", "price_z"], ascending=[False, True]
    )
    _write_csv(OUT_SUSP, susp)
    print(f"Sorted flags → {OUT_SUSP}")

    export_labeling_queue(df)

    build_dashboard(df)


if __name__ == "__main__":
    main()
