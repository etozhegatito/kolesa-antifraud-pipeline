# -*- coding: utf-8 -*-
"""Разброс MAPE и честная локализация ошибки по возрасту и цене.

Модуль не обучает модель и не ходит в сеть. Он читает построчные прогнозы
того же grouped OOF, который ``train_price_model`` использовал для основной
метрики, и отвечает на два вопроса:

1. Насколько MAPE может плавать просто из-за состава выборки?
2. Проблема действительно в машинах старше пяти лет или прежде всего в цене?

Bootstrap пересэмплирует целые группы перезаливов. Одна и та же машина под
несколькими ad_id поэтому не притворяется несколькими независимыми объектами.

Запуск: python -m kz.ml.mape_stability
Выход: data/eda/mape_stability.json + mape_stability_segments.csv
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from kz.ml.train_price_model import OOF_DIAGNOSTICS_PATH, RANDOM_SEED

REPORT_PATH = Path("data/eda/mape_stability.json")
SEGMENTS_PATH = Path("data/eda/mape_stability_segments.csv")
BOOTSTRAP_REPEATS = 2000
AGE_LABELS = ["0-5", "6-10", "11-20", "21+"]
PRICE_LABELS = ["<5M", "5M+"]


def grouped_bootstrap_mape(
    ape_pct: np.ndarray,
    groups: np.ndarray,
    n_boot: int = BOOTSTRAP_REPEATS,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int | list[float]]:
    """MAPE и его ДИ при bootstrap целых независимых машин.

    Это неопределённость состава выборки, а не случайного seed CatBoost.
    Production CV намеренно детерминирован, чтобы новые замеры были сравнимы.
    """
    ape = np.asarray(ape_pct, dtype=float)
    group_values = pd.Series(groups, dtype="string")
    valid = np.isfinite(ape) & group_values.notna().to_numpy()
    ape = ape[valid]
    group_values = group_values[valid]
    if len(ape) == 0:
        raise ValueError("Нельзя посчитать MAPE на пустом сегменте")

    group_codes, _ = pd.factorize(group_values, sort=False)
    n_groups = int(group_codes.max()) + 1
    sizes = np.bincount(group_codes, minlength=n_groups).astype(float)
    sums = np.bincount(group_codes, weights=ape, minlength=n_groups)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.integers(0, n_groups, n_groups)
        values[i] = sums[sampled].sum() / sizes[sampled].sum()

    lo, p05, p95, hi = np.percentile(values, [2.5, 5, 95, 97.5])
    return {
        "n": int(len(ape)),
        "independent_groups": n_groups,
        "mape_pct": float(ape.mean()),
        "median_ape_pct": float(np.median(ape)),
        "bootstrap_std_pct_points": float(values.std(ddof=1)),
        "bootstrap_90_interval": [float(p05), float(p95)],
        "bootstrap_95_ci": [float(lo), float(hi)],
        "bootstrap_repeats": int(n_boot),
    }


def _metric_row(
    df: pd.DataFrame,
    segment_type: str,
    segment: str,
    n_boot: int,
) -> dict:
    result = grouped_bootstrap_mape(
        df["absolute_percentage_error_pct"].to_numpy(dtype=float),
        df["duplicate_group"].astype(str).to_numpy(),
        n_boot=n_boot,
    )
    return {"segment_type": segment_type, "segment": segment, **result}


def build_report(
    oof: pd.DataFrame,
    n_boot: int = BOOTSTRAP_REPEATS,
) -> tuple[dict, pd.DataFrame]:
    """Строит общий, возрастной, ценовой и перекрёстный срезы."""
    required = {
        "duplicate_group", "age", "actual_price_tenge",
        "absolute_percentage_error_pct",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF-отчёту не хватает колонок: {', '.join(missing)}")

    work = oof.copy()
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["actual_price_tenge"] = pd.to_numeric(
        work["actual_price_tenge"], errors="coerce"
    )
    work["absolute_percentage_error_pct"] = pd.to_numeric(
        work["absolute_percentage_error_pct"], errors="coerce"
    )
    work = work[
        work["age"].notna()
        & work["actual_price_tenge"].gt(0)
        & work["absolute_percentage_error_pct"].notna()
    ].copy()
    if work.empty:
        raise ValueError("После проверки типов в OOF-отчёте не осталось строк")

    work["age_segment"] = pd.cut(
        work["age"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=AGE_LABELS,
        include_lowest=True,
    )
    work["price_segment"] = pd.cut(
        work["actual_price_tenge"],
        bins=[0, 5_000_000, np.inf],
        labels=PRICE_LABELS,
        include_lowest=True,
    )

    rows = [_metric_row(work, "overall", "all", n_boot)]
    for age in AGE_LABELS:
        part = work[work["age_segment"] == age]
        if not part.empty:
            rows.append(_metric_row(part, "age", age, n_boot))
    for price in PRICE_LABELS:
        part = work[work["price_segment"] == price]
        if not part.empty:
            rows.append(_metric_row(part, "price", price, n_boot))
    for age in AGE_LABELS:
        for price in PRICE_LABELS:
            part = work[
                (work["age_segment"] == age)
                & (work["price_segment"] == price)
            ]
            if not part.empty:
                rows.append(_metric_row(part, "age_x_price", f"{age} | {price}", n_boot))

    segments = pd.DataFrame(rows)
    total_n = float(segments.iloc[0]["n"])
    total_mape = float(segments.iloc[0]["mape_pct"])
    segments["share_rows_pct"] = segments["n"] / total_n * 100
    # MAPE — среднее всех APE. Поэтому n_segment / n_total * MAPE_segment
    # даёт точный вклад сегмента в общую метрику, а не эвристику важности.
    segments["mape_contribution_pct_points"] = (
        segments["n"] / total_n * segments["mape_pct"]
    )
    segments["share_total_error_pct"] = (
        segments["mape_contribution_pct_points"] / total_mape * 100
    )
    records = segments.to_dict(orient="records")
    nested = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(OOF_DIAGNOSTICS_PATH),
        "definition": (
            "Grouped bootstrap of OOF absolute percentage errors; whole "
            "duplicate/relist groups are resampled together."
        ),
        "interpretation": (
            "Confidence intervals describe sample-composition uncertainty, "
            "not CatBoost seed variation."
        ),
        "overall": records[0],
        "by_age": [r for r in records if r["segment_type"] == "age"],
        "by_price": [r for r in records if r["segment_type"] == "price"],
        "by_age_and_price": [
            r for r in records if r["segment_type"] == "age_x_price"
        ],
    }
    return nested, segments


def save_report(report: dict, segments: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = REPORT_PATH.with_suffix(".json.tmp")
    tmp_csv = SEGMENTS_PATH.with_suffix(".csv.tmp")
    tmp_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    segments.to_csv(tmp_csv, index=False)
    os.replace(tmp_json, REPORT_PATH)
    os.replace(tmp_csv, SEGMENTS_PATH)


def _print_table(title: str, rows: pd.DataFrame) -> None:
    print(f"\n{title}")
    for row in rows.itertuples(index=False):
        lo, hi = row.bootstrap_95_ci
        print(
            f"  {row.segment:<12} n={row.n:<5} "
            f"MAPE={row.mape_pct:>5.2f}%  "
            f"95% ДИ [{lo:.2f}; {hi:.2f}]  "
            f"доля всей ошибки={row.share_total_error_pct:>4.1f}%"
        )


def main() -> None:
    if not OOF_DIAGNOSTICS_PATH.exists():
        raise FileNotFoundError(
            f"Нет {OOF_DIAGNOSTICS_PATH}. Сначала: "
            "python -m kz.ml.train_price_model"
        )
    oof = pd.read_csv(OOF_DIAGNOSTICS_PATH, dtype={"duplicate_group": str})
    report, segments = build_report(oof)
    save_report(report, segments)

    overall = segments.iloc[0]
    lo, hi = overall["bootstrap_95_ci"]
    print("Изменчивость grouped OOF MAPE при смене состава выборки:")
    print(
        f"  MAPE={overall['mape_pct']:.2f}%  "
        f"bootstrap SD={overall['bootstrap_std_pct_points']:.2f} п.п.  "
        f"95% ДИ [{lo:.2f}; {hi:.2f}]"
    )
    _print_table("По возрасту:", segments[segments.segment_type == "age"])
    _print_table("По цене:", segments[segments.segment_type == "price"])
    _print_table(
        "Возраст × цена:", segments[segments.segment_type == "age_x_price"]
    )
    print(f"\nJSON → {REPORT_PATH}")
    print(f"CSV  → {SEGMENTS_PATH}")


if __name__ == "__main__":
    main()
