# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.survival` module."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from kz.core.db import get_engine

OUT_PNG = Path("data/eda/survival.png")
HORIZON = 21
MIN_EVENTS_PER_FEATURE = 10
EVENT_STATUSES = ("archived", "deleted")

_MON = {
    "янв": 1,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "май": 5,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}


def parse_posted(s, today: date | None = None):
    """Implement `parse_posted`."""
    m = re.match(r"\s*(\d{1,2})\s+([а-яё]+)", str(s).lower())
    if not m:
        return None
    day, mon = int(m.group(1)), _MON.get(m.group(2)[:3])
    if not mon:
        return None
    today = today or date.today()
    try:
        parsed = date(today.year, mon, day)
    except ValueError:
        return None
    return date(today.year - 1, mon, day) if parsed > today else parsed


def build_lifespans(cd: pd.DataFrame, st: pd.DataFrame, sg: pd.DataFrame) -> pd.DataFrame:
    """Implement `build_lifespans`."""
    d = cd.merge(st, on="ad_id", how="left").merge(sg, on="ad_id", how="left")
    d["start"] = pd.to_datetime(d["posted_date"].map(parse_posted))
    d["event"] = d["status"].isin(EVENT_STATUSES).astype(int)

    d["end"] = pd.to_datetime(np.where(d["event"] == 1, d["checked_at"], d["last_seen"]))
    d["days"] = (d["end"] - d["start"]).dt.days
    d = d[d["days"].notna() & (d["days"] >= 0) & d["price_tenge"].notna()]
    return d.reset_index(drop=True)


def load_survival() -> pd.DataFrame:
    """Implement `load_survival`."""
    eng = get_engine()

    cd = pd.read_sql("SELECT * FROM clean_data", eng, dtype={"ad_id": str})

    st = pd.read_sql("SELECT ad_id, checked_at FROM ad_status", eng, dtype={"ad_id": str})
    sg = pd.read_sql(
        "SELECT ad_id, MAX(seen_date) AS last_seen FROM sightings GROUP BY ad_id",
        eng,
        dtype={"ad_id": str},
    )
    return build_lifespans(cd, st, sg)


def add_price_position(d: pd.DataFrame) -> pd.DataFrame:
    """Implement `add_price_position`."""
    from kz.ml.train_price_model import coerce_features, FEATURES, load_artifact

    model, _ = load_artifact()
    X = coerce_features(d.copy())[FEATURES]
    fair = np.exp(model.predict(X))
    d = d.copy()
    d["fair_price"] = fair
    d["price_ratio"] = d["price_tenge"] / fair

    d["price_group"] = pd.cut(
        d["price_ratio"],
        [0, 0.9, 1.1, np.inf],
        labels=["below market", "near market", "above market"],
    )
    return d


def kaplan_meier(d: pd.DataFrame, log=print):
    """Implement `kaplan_meier`."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter()
    km.fit(d["days"], d["event"], label="all listings")
    log("\nShare of listings still on the market:")
    for t in (3, 7, 14, HORIZON):
        s = float(km.survival_function_at_times(t).iloc[0])
        log(f"  after {t:2d} days — {s * 100:5.1f}%   ({100 - s * 100:.1f}% left the market)")
    med = km.median_survival_time_
    log(
        "\nMedian listing lifetime: "
        + (
            f"{med:.0f} days"
            if np.isfinite(med)
            else "not reached; fewer than half the listings left during observation"
        )
    )
    return km


def by_price_group(d: pd.DataFrame, log=print):
    """Implement `by_price_group`."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    log("\nMarket exit by price group (share gone by day 14):")
    curves = {}
    for name, g in d.groupby("price_group", observed=True):
        if g["event"].sum() < 5:
            log(f"  {name:16} events {int(g['event'].sum())}; too few")
            continue
        km = KaplanMeierFitter().fit(g["days"], g["event"], label=str(name))
        s14 = float(km.survival_function_at_times(14).iloc[0])
        log(
            f"  {str(name):16} n={len(g):5d}  events={int(g['event'].sum()):4d}  "
            f"gone by day 14 {100 - s14 * 100:5.1f}%"
        )
        curves[str(name)] = km

    res = multivariate_logrank_test(d["days"], d["price_group"], d["event"])
    log(
        f"\nLog-rank test across curves: p = {res.p_value:.4f}"
        + (
            "  — statistically significant"
            if res.p_value < 0.05
            else "  — not statistically significant"
        )
    )
    return curves


def cox_model(d: pd.DataFrame, log=print):
    """Implement `cox_model`."""
    from lifelines import CoxPHFitter

    feats = ["price_ratio", "age", "photos_count", "is_vip"]
    n_events = int(d["event"].sum())
    allowed = n_events // MIN_EVENTS_PER_FEATURE
    if allowed < len(feats):
        feats = feats[: max(1, allowed)]
        log(
            f"\n(features limited to {len(feats)}: {n_events} events; "
            f"require at least {MIN_EVENTS_PER_FEATURE} events per feature)"
        )

    work = d[feats + ["days", "event"]].dropna().copy()
    cph = CoxPHFitter().fit(work, duration_col="days", event_col="event")
    log("\nCox model hazard ratios (above 1 means a faster market exit):")
    s = cph.summary
    for name in s.index:
        hr, p = s.loc[name, "exp(coef)"], s.loc[name, "p"]
        mark = "  ← significant" if p < 0.05 else ""
        log(f"  {name:16} HR={hr:5.2f}   p={p:.4f}{mark}")
    return cph


def plot(km, curves, path: Path = OUT_PNG) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    km.plot_survival_function(ax=ax[0], color="#2563c9")
    ax[0].set_title("Share of listings still on market")
    ax[0].set_xlabel("days since publication")
    ax[0].set_ylabel("still listed")
    ax[0].set_xlim(0, HORIZON)
    ax[0].grid(alpha=0.3)
    for _label, c in curves.items():
        c.plot_survival_function(ax=ax[1])
    ax[1].set_title("By price group")
    ax[1].set_xlabel("days since publication")
    ax[1].set_xlim(0, HORIZON)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\nChart → {path}")


def verified_bracket(d: pd.DataFrame, log=print) -> None:
    """Implement `verified_bracket`."""
    from lifelines import KaplanMeierFitter

    from kz.core.db import get_engine

    st = pd.read_sql("SELECT ad_id FROM ad_status", get_engine(), dtype={"ad_id": str})
    checked = d[d["ad_id"].isin(set(st["ad_id"]))]
    if len(checked) < 50 or len(checked) == len(d):
        return

    log("\nShare gone by day 14 is a bracket, not a point estimate:")
    for name, sub in [("all listings", d), ("checked listings only", checked)]:
        km = KaplanMeierFitter().fit(sub["days"], sub["event"])
        s14 = float(km.survival_function_at_times(14).iloc[0])
        log(f"  {name:24} n={len(sub):5}  gone {100 - s14 * 100:5.1f}%")
    log("  The first bound is low because unchecked rows count as active;")
    log("  the second is high because missing listings were checked first.")


def main():
    from kz.core import freshness as fr

    d = load_survival()

    state = fr.measure()
    fr.report(state)
    for w in fr.stale_warnings(state):
        print(f"  ⚠ {w}")

    print(
        f"\nListings: {len(d)}   events: {int(d.event.sum())}   "
        f"censored: {int((1 - d.event).sum())}"
    )
    print(
        f"Event share {d.event.mean() * 100:.1f}% — most listings remain active, "
        "which is why survival analysis is required instead of ordinary regression."
    )

    km = kaplan_meier(d)
    verified_bracket(d)
    d = add_price_position(d)
    curves = by_price_group(d)
    try:
        cox_model(d)
    except Exception as e:  # noqa: BLE001 -- intentional exception
        print(f"\nCox model did not converge: {e}")
    plot(km, curves)
    print(
        f"\nEvidence boundary: the observation window is about {int(d.days.max())} days. "
        f"The data do not support conclusions beyond {HORIZON} days."
    )


if __name__ == "__main__":
    main()
