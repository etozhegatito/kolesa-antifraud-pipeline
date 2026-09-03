# -*- coding: utf-8 -*-
"""Generate evidence-based photo-quality suggestions for vehicle listings."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


WORSE_THAN = 0.20


MIN_PHOTOS = 2


def load_photo_signals() -> pd.DataFrame:
    """Aggregate CLIP and image-quality signals by listing."""
    from kz.ml.photo_clip import load as load_clip

    clip = load_clip()
    agg = (
        clip.groupby("ad_id")
        .agg(
            clip_dirty=("clip_dirty", "mean"),
            clip_rusty=("clip_rusty", "mean"),
            clip_studio=("clip_studio", "mean"),
            n_photos=("clip_dirty", "size"),
        )
        .reset_index()
    )

    try:
        from kz.ml.photo_features import load_quality

        agg = agg.merge(load_quality(), on="ad_id", how="left")
    except Exception:  # noqa: BLE001 -- intentional exception
        pass
    return agg


def thresholds(df: pd.DataFrame, cols: list[str], worse_than: float = WORSE_THAN) -> dict:
    """Derive corpus-relative thresholds for advice signals."""
    low_is_bad = {"img_brightness", "img_sharpness", "img_contrast"}
    out = {}
    for c in cols:
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) < 50:
            continue
        q = worse_than if c in low_is_bad else 1 - worse_than
        out[c] = float(np.quantile(v, q))
    return out


def advise(row: pd.Series, cuts: dict) -> list[str]:
    """Return photo suggestions supported by measured image signals."""
    out = []
    pct = int(WORSE_THAN * 100)

    if row.get("n_photos", 0) < MIN_PHOTOS:
        return ["Too few photos are available to assess image quality."]

    if "img_brightness" in cuts and row.get("img_brightness", np.inf) < cuts["img_brightness"]:
        out.append(
            f"The photos are darker than {100 - pct}% of listings. "
            f"Retake them in daylight or a well-lit location."
        )
    if "img_sharpness" in cuts and row.get("img_sharpness", np.inf) < cuts["img_sharpness"]:
        out.append(
            f"The photos are less sharp than {100 - pct}% of listings. "
            f"Clean the lens and stabilize the camera."
        )
    if "clip_dirty" in cuts and row.get("clip_dirty", -np.inf) > cuts["clip_dirty"]:
        out.append(
            "The vehicle appears dirty in the photos. Washing it before "
            "a new photo session is a low-cost listing improvement."
        )
    if "clip_rusty" in cuts and row.get("clip_rusty", -np.inf) > cuts["clip_rusty"]:
        out.append(
            "The photos show signs of corrosion. If present, disclose it "
            "in the description so a low price has a clear explanation."
        )
    return out


def validate(log=print) -> None:
    """Check observational associations without making causal claims."""
    from kz.core.db import get_engine

    from kz.ml.survival import parse_posted

    sig = load_photo_signals()
    cd = pd.read_sql(
        "SELECT ad_id, views_count, price_tenge, age, photos_count, "
        "posted_date, scraped_at FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    d = sig.merge(cd, on="ad_id", how="inner")
    d["views_count"] = pd.to_numeric(d.views_count, errors="coerce")

    start = pd.to_datetime(d.posted_date.map(parse_posted))
    seen = pd.to_datetime(d.scraped_at, errors="coerce")
    d["days_up"] = (seen - start).dt.days.clip(lower=1)
    d["views_per_day"] = d.views_count / d.days_up

    d = d[d.views_per_day.notna() & d.price_tenge.notna() & d.days_up.notna()]
    if len(d) < 200:
        log(f"Listings with photos and views: {len(d)} — too few to evaluate")
        return

    cols = ["img_brightness", "img_sharpness", "clip_dirty", "clip_rusty"]
    cuts = thresholds(d, cols)
    d["advice_n"] = d.apply(lambda r: len(advise(r, cuts)), axis=1)

    log(f"Listings with photos and views: {len(d)}")
    log(f"Thresholds (corpus percentile {int(WORSE_THAN * 100)}%):")
    for c, v in cuts.items():
        log(f"   {c:16} {v:.3f}")

    log(
        f"\nMedian listing age: {d.days_up.median():.0f} days "
        f"(from {d.days_up.min():.0f} to {d.days_up.max():.0f})"
    )

    band = pd.cut(
        d.price_tenge, [0, 5e6, 10e6, 20e6, np.inf], labels=["<5M", "5-10M", "10-20M", "20M+"]
    )
    for metric, title in [
        ("views_count", "TOTAL views (not normalized)"),
        ("views_per_day", "views PER DAY"),
    ]:
        log(f"\nMedian: {title}")
        log(f"   {'group':8} {'no advice':>12} {'has advice':>12} {'difference':>9}")
        for name in band.cat.categories:
            m = np.asarray(band == name)
            good = d.loc[m & (d.advice_n == 0), metric]
            bad = d.loc[m & (d.advice_n > 0), metric]
            if len(good) < 20 or len(bad) < 20:
                continue
            g, b = float(good.median()), float(bad.median())
            log(f"   {name:8} {g:12.1f} {b:12.1f} {(b - g) / g * 100:8.1f}%")

    log("\nMedian listing age by advice group:")
    for label, sub in [("no advice", d[d.advice_n == 0]), ("has advice", d[d.advice_n > 0])]:
        log(f"   {label:14} {sub.days_up.median():5.0f} days  (n={len(sub)})")

    log(
        "\nCAUSAL CHECK FAILED: this is observational, not experimental. "
        "Photo effects would require randomly changing photos for a treatment group."
    )


def main():
    if "--validate" in sys.argv:
        validate()
        return

    sig = load_photo_signals()
    cuts = thresholds(sig, ["img_brightness", "img_sharpness", "clip_dirty", "clip_rusty"])
    sig["advice"] = sig.apply(lambda r: advise(r, cuts), axis=1)
    n = int((sig.advice.map(len) > 0).sum())
    print(f"Listings with photos: {len(sig)}")
    print(f"Listings with suggestions: {n} ({n / len(sig) * 100:.0f}%)\n")

    counts = pd.Series([a.split(".")[0] for lst in sig.advice for a in lst]).value_counts()
    print("Most frequent suggestions:")
    for text, k in counts.items():
        print(f"  {k:5}  {text}")

    print("\nExample listings with suggestions:")
    for _, r in sig[sig.advice.map(len) > 0].head(3).iterrows():
        print(f"\n  {r.ad_id} ({int(r.n_photos)} photos):")
        for a in r.advice:
            print(f"    • {a}")


if __name__ == "__main__":
    main()
