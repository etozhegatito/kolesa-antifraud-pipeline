# -*- coding: utf-8 -*-
"""Analyse the fixed below-5M review pilot against blinded OOF errors.

The labels are diagnostic evidence, not model features.  This report keeps the
three sampling sources separate because 30 of the 50 listings were selected
for large errors on purpose.  Pooling them into a market-wide MAPE or condition
prevalence estimate would therefore be misleading.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kz.report.price_review import (
    DATA_ISSUES,
    EVIDENCE_SOURCES,
    PRICE_VALIDITY,
    VEHICLE_STATES,
    journal_by_id,
    load_pilot,
)

REPORT_DIR = Path("data/eda")
REPORT_CSV = REPORT_DIR / "price_review_analysis.csv"
REPORT_JSON = REPORT_DIR / "price_review_analysis.json"
REPORT_HTML = REPORT_DIR / "price_review_analysis.html"

CONFIRMED_NON_COMPARABLE = frozenset(
    {"cash_uncleared", "credit_or_down_payment", "parts_price", "exchange_or_placeholder"}
)
MATERIAL_CONDITION = frozenset({"repair_needed", "non_running", "wreck", "parts"})
ANALYSIS_VERSION = 1


def join_reviews(pilot: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach exactly one valid review to every fixed-pilot listing."""
    if pilot["ad_id"].astype(str).duplicated().any():
        raise ValueError("The fixed price-review pilot contains duplicate ad_id values")
    if labels.empty:
        raise ValueError("The price-review journal is empty")

    work = labels.copy()
    work["ad_id"] = work["ad_id"].astype(str)
    if work["ad_id"].duplicated().any():
        raise ValueError("The price-review journal contains duplicate ad_id values")

    expected = {
        "vehicle_state": set(VEHICLE_STATES),
        "price_validity": set(PRICE_VALIDITY),
        "evidence_source": set(EVIDENCE_SOURCES),
        "data_issue": set(DATA_ISSUES),
    }
    for column, allowed in expected.items():
        invalid = set(work[column].dropna().astype(str)) - allowed
        if invalid:
            raise ValueError(f"Invalid {column} values: {sorted(invalid)}")

    keep = [
        "ad_id",
        "vehicle_state",
        "price_validity",
        "evidence_source",
        "data_issue",
        "comment",
        "labeled_at",
        "selection_source",
        "dataset_split",
        "annotator",
        "label_version",
    ]
    joined = pilot.copy()
    joined["ad_id"] = joined["ad_id"].astype(str)
    joined = joined.merge(
        work[[column for column in keep if column in work]],
        on="ad_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
    )

    missing = joined.loc[joined["vehicle_state"].isna(), "ad_id"].tolist()
    if missing:
        raise ValueError(f"Missing price-review labels for {len(missing)} pilot listings")

    for column in ("selection_source", "dataset_split"):
        labelled = f"{column}_label"
        if labelled not in joined:
            continue
        mismatch = joined[labelled].notna() & joined[column].ne(joined[labelled])
        if mismatch.any():
            raise ValueError(f"Journal {column} does not match the fixed pilot")
        joined = joined.drop(columns=labelled)
    return joined


def add_error_fields(rows: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable OOF residual fields and coarse diagnosis groups."""
    out = rows.copy()
    actual = pd.to_numeric(out["price_tenge"], errors="raise")
    predicted = pd.to_numeric(out["routed_pred_tenge"], errors="raise")
    if (actual <= 0).any():
        raise ValueError("Price-review analysis requires positive target prices")

    out["signed_percentage_error_pct"] = (predicted - actual) / actual * 100
    out["absolute_error_tenge"] = (predicted - actual).abs()
    out["condition_group"] = np.select(
        [
            out["vehicle_state"].eq("normal"),
            out["vehicle_state"].eq("cosmetic"),
            out["vehicle_state"].isin(MATERIAL_CONDITION),
        ],
        ["normal", "cosmetic", "material_condition"],
        default="unclear",
    )
    out["target_group"] = np.select(
        [
            out["price_validity"].eq("comparable_cash"),
            out["price_validity"].isin(CONFIRMED_NON_COMPARABLE),
        ],
        ["comparable", "non_comparable"],
        default="unclear",
    )
    return out


def category_summary(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return descriptive error statistics for one review dimension."""
    return (
        rows.groupby(column, dropna=False)
        .agg(
            n=("ad_id", "size"),
            mape_pct=("absolute_percentage_error_pct", "mean"),
            median_ape_pct=("absolute_percentage_error_pct", "median"),
            mean_signed_error_pct=("signed_percentage_error_pct", "mean"),
            median_absolute_error_tenge=("absolute_error_tenge", "median"),
        )
        .reset_index()
        .sort_values(["mape_pct", "n"], ascending=[False, False])
        .reset_index(drop=True)
    )


def analyse(rows: pd.DataFrame) -> dict:
    """Build a serialisable report while preserving the sampling caveat."""
    summaries = {}
    for column in (
        "selection_source",
        "dataset_split",
        "vehicle_state",
        "condition_group",
        "price_validity",
        "target_group",
        "evidence_source",
        "data_issue",
    ):
        summaries[column] = category_summary(rows, column).to_dict(orient="records")

    random_sources = rows[
        rows["selection_source"].isin({"random_local_audit", "random_cheap_control"})
    ]
    top_columns = [
        "ad_id",
        "brand",
        "model",
        "year",
        "price_tenge",
        "routed_pred_tenge",
        "absolute_percentage_error_pct",
        "signed_percentage_error_pct",
        "vehicle_state",
        "price_validity",
        "evidence_source",
        "data_issue",
        "selection_source",
    ]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "rows": int(len(rows)),
        "confirmed_non_comparable_rows": int(
            rows["price_validity"].isin(CONFIRMED_NON_COMPARABLE).sum()
        ),
        "material_condition_rows": int(rows["vehicle_state"].isin(MATERIAL_CONDITION).sum()),
        "random_source_rows": int(len(random_sources)),
        "random_source_mape_pct": float(random_sources["absolute_percentage_error_pct"].mean()),
        "sampling_warning": (
            "Thirty listings were selected for large OOF errors and every listing required "
            "locally available photos. Pooled pilot figures are diagnostic and do not estimate "
            "the full below-5M market."
        ),
        "summaries": summaries,
        "top_errors": rows.nlargest(10, "absolute_percentage_error_pct")[top_columns].to_dict(
            orient="records"
        ),
    }


def _fmt(value: object, column: str) -> str:
    if pd.isna(value):
        return "—"
    if column == "n":
        return str(int(value))
    if column.endswith("_tenge"):
        return f"₸{float(value):,.0f}"
    if column.endswith("_pct"):
        return f"{float(value):.2f}%"
    return html.escape(str(value))


def _table(title: str, records: list[dict]) -> str:
    if not records:
        return f"<section><h2>{html.escape(title)}</h2><p>No rows.</p></section>"
    columns = list(records[0])
    head = "".join(
        f"<th>{html.escape(column.replace('_', ' ').title())}</th>" for column in columns
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_fmt(row.get(column), column)}</td>" for column in columns)
        + "</tr>"
        for row in records
    )
    return (
        f"<section><h2>{html.escape(title)}</h2><div class='table-wrap'><table>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>"
    )


def render_html(report: dict) -> str:
    """Render a dependency-free local report for browser inspection."""
    summaries = report["summaries"]
    parts = next(
        (row for row in summaries["price_validity"] if row["price_validity"] == "parts_price"),
        None,
    )
    parts_text = (
        f"{int(parts['n'])} parts-price rows have {parts['mape_pct']:.1f}% mean APE."
        if parts
        else "No parts-price row was found."
    )
    sections = "".join(
        [
            _table("Selection source", summaries["selection_source"]),
            _table("Vehicle state", summaries["vehicle_state"]),
            _table("Price validity", summaries["price_validity"]),
            _table("Evidence source", summaries["evidence_source"]),
            _table("Data quality", summaries["data_issue"]),
            _table("Largest OOF errors", report["top_errors"]),
        ]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Below-5M review analysis</title><style>
:root{{--bg:#0b1020;--card:#131b2e;--line:#2b3853;--text:#eaf0ff;--muted:#9fb0ce;--accent:#78a6ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1220px;margin:auto;padding:36px 22px 70px}}h1{{font-size:clamp(28px,5vw,48px);margin:0 0 8px}}
p{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.card,section,.warning{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}}
.card b{{display:block;font-size:28px;color:var(--accent)}}.warning{{border-color:#80652a;color:#ffd77a}}
section{{margin-top:16px}}h2{{margin-top:0}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>Below-5M review analysis</h1>
<p>The fixed human-review pilot is joined to grouped out-of-fold predictions.</p>
<div class="cards"><div class="card"><span>Reviewed</span><b>{report["rows"]}</b></div>
<div class="card"><span>Confirmed non-comparable</span><b>{report["confirmed_non_comparable_rows"]}</b></div>
<div class="card"><span>Material condition</span><b>{report["material_condition_rows"]}</b></div>
<div class="card"><span>Random-source MAPE</span><b>{report["random_source_mape_pct"]:.2f}%</b></div></div>
<div class="warning"><b>Interpretation boundary.</b> {html.escape(report["sampling_warning"])}</div>
<p><b>Strongest actionable finding:</b> {html.escape(parts_text)} Explicitly incomplete
vehicles should leave price training before adding a speculative CV feature.</p>{sections}</main></body></html>"""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    pilot = load_pilot()
    labels = pd.DataFrame(journal_by_id().values())
    rows = add_error_fields(join_reviews(pilot, labels))
    report = analyse(rows)

    export_columns = [
        "ad_id",
        "brand",
        "model",
        "year",
        "age",
        "price_tenge",
        "routed_pred_tenge",
        "absolute_percentage_error_pct",
        "signed_percentage_error_pct",
        "absolute_error_tenge",
        "vehicle_state",
        "condition_group",
        "price_validity",
        "target_group",
        "evidence_source",
        "data_issue",
        "selection_source",
        "dataset_split",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows[export_columns].to_csv(REPORT_CSV, index=False)
    _atomic_text(REPORT_JSON, json.dumps(report, ensure_ascii=False, indent=2))
    _atomic_text(REPORT_HTML, render_html(report))

    print(f"Reviewed pilot: {report['rows']} listings")
    print(f"Confirmed non-comparable targets: {report['confirmed_non_comparable_rows']}")
    print(f"Material-condition listings: {report['material_condition_rows']}")
    print(f"Random-source MAPE: {report['random_source_mape_pct']:.2f}%")
    print(f"HTML report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
