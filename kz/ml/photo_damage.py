# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.photo_damage` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kz.core.db import get_engine
from kz.ml.photo_clip import load as load_clip_scores
from kz.ml.photo_clip import load_embeddings
from kz.report.photo_labels import NEEDS_REVIEW, read_journal

CHEAP_EDGE = 5_000_000
N_SPLITS = 5
SEED = 42
BOOTSTRAPS = 1000
POSITIVE = {"damaged", "wreck"}
USED = POSITIVE | {"intact"}


def groups_from_hashes(ad_ids, hashes) -> np.ndarray:
    """Implement `groups_from_hashes`."""
    ads = np.asarray([str(x) for x in ad_ids])
    hs = np.asarray([str(x) if x is not None else "" for x in hashes])
    parent = {ad: ad for ad in np.unique(ads)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_by_hash = {}
    for ad, phash in zip(ads, hs, strict=True):
        if not phash:
            continue
        if phash in first_by_hash:
            union(ad, first_by_hash[phash])
        else:
            first_by_hash[phash] = ad
    return np.asarray([find(ad) for ad in ads])


def visual_groups(d: pd.DataFrame) -> np.ndarray:
    """Implement `visual_groups`."""
    import imagehash
    from PIL import Image, ImageOps

    hashes = []
    cache = {}
    for raw in d.get("path", pd.Series("", index=d.index)).fillna(""):
        path = str(raw)
        if path not in cache:
            try:
                with Image.open(Path(path)) as img:
                    cache[path] = str(imagehash.phash(ImageOps.exif_transpose(img)))
            except (OSError, ValueError):
                cache[path] = ""
        hashes.append(cache[path])
    return groups_from_hashes(d.ad_id.to_numpy(), hashes)


def load_labelled() -> tuple[pd.DataFrame, np.ndarray]:
    """Implement `load_labelled`."""
    _, rows = read_journal()
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise RuntimeError("photo_labels journal is empty")
    labels = labels.drop_duplicates(["ad_id", "position"], keep="last")
    if "review_status" in labels:
        labels = labels[labels.review_status.fillna("") != NEEDS_REVIEW]
    labels = labels[labels.label.isin(USED)].copy()
    if "dataset_split" not in labels:
        labels["dataset_split"] = "train"
    labels["dataset_split"] = labels.dataset_split.fillna("").replace("", "train")
    labels["ad_id"] = labels.ad_id.astype(str)
    labels["position"] = pd.to_numeric(labels.position, errors="coerce")
    labels = labels.dropna(subset=["position"])
    labels["position"] = labels.position.astype(int)

    idx, emb = load_embeddings()
    idx = idx.reset_index(drop=True)
    idx["embedding_row"] = idx.index
    idx["ad_id"] = idx.ad_id.astype(str)
    idx["position"] = idx.position.astype(int)
    d = labels.merge(
        idx[["ad_id", "position", "embedding_row"]], on=["ad_id", "position"], how="inner"
    )

    cars = pd.read_sql(
        "SELECT ad_id, age, price_tenge FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    d = d.merge(cars, on="ad_id", how="left")
    d["age"] = pd.to_numeric(d.age, errors="coerce")
    d["price_tenge"] = pd.to_numeric(d.price_tenge, errors="coerce")
    d = d.dropna(subset=["age", "price_tenge"])
    d["target"] = d.label.isin(POSITIVE).astype(int)
    return d.reset_index(drop=True), emb


def _splits(d: pd.DataFrame):
    groups = visual_groups(d)
    independent = pd.DataFrame({"group": groups, "target": d.target}).groupby("group").target.max()
    positive_groups = int(independent.sum())
    negative_groups = int((independent == 0).sum())
    n = min(N_SPLITS, positive_groups, negative_groups)
    if n < 2:
        raise RuntimeError("At least two independent visual groups per class are required")
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    dummy = np.zeros((len(d), 1))
    return list(cv.split(dummy, d.target.to_numpy(), groups))


def oof_scores(d: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Implement `oof_scores`."""
    y = d.target.to_numpy()
    photo = emb[d.embedding_row.to_numpy()]
    table = np.column_stack(
        [
            d.age.to_numpy(dtype=float),
            np.log(d.price_tenge.to_numpy(dtype=float)),
        ]
    )
    out = d[["ad_id", "target"]].copy()
    out["table"] = np.nan
    out["photo"] = np.nan
    out["combined"] = np.nan

    for tr, te in _splits(d):
        table_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        photo_model = LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced")
        combined_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"),
        )
        table_model.fit(table[tr], y[tr])
        photo_model.fit(photo[tr], y[tr])
        joined = np.column_stack([table, photo])
        combined_model.fit(joined[tr], y[tr])
        out.loc[te, "table"] = table_model.predict_proba(table[te])[:, 1]
        out.loc[te, "photo"] = photo_model.predict_proba(photo[te])[:, 1]
        out.loc[te, "combined"] = combined_model.predict_proba(joined[te])[:, 1]
    return out


def per_ad(frame_scores: pd.DataFrame) -> pd.DataFrame:
    """Implement `per_ad`."""
    return frame_scores.groupby("ad_id", as_index=False).agg(
        target=("target", "max"),
        table=("table", "max"),
        photo=("photo", "max"),
        combined=("combined", "max"),
    )


def auc_ci(y, score, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    """Implement `auc_ci`."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return auc, float(lo), float(hi)


def average_precision_ci(y, score, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    """Implement `average_precision_ci`."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    value = float(average_precision_score(y, score))
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(average_precision_score(y[idx], score[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return value, float(lo), float(hi)


def auc_delta_ci(y, candidate, baseline, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    """Implement `auc_delta_ci`."""
    y = np.asarray(y, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    delta = float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline))
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(
                roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], baseline[idx])
            )
    lo, hi = np.percentile(values, [2.5, 97.5])
    return delta, float(lo), float(hi)


def zero_shot_per_ad(d: pd.DataFrame) -> pd.DataFrame:
    scores = load_clip_scores()[["ad_id", "position", "clip_damaged"]].copy()
    scores["ad_id"] = scores.ad_id.astype(str)
    scores["position"] = scores.position.astype(int)
    keys = d[["ad_id", "position", "target"]].merge(scores, on=["ad_id", "position"], how="inner")
    return keys.groupby("ad_id", as_index=False).agg(
        target=("target", "max"), zero_shot=("clip_damaged", "max")
    )


def main() -> None:
    d, emb = load_labelled()
    d = d[d.price_tenge < CHEAP_EDGE].reset_index(drop=True)
    audit_ads = d.loc[d.dataset_split == "audit", "ad_id"].nunique()
    d = d[d.dataset_split != "audit"].reset_index(drop=True)
    n_pos_ads = d.loc[d.target == 1, "ad_id"].nunique()
    n_neg_ads = d.loc[d.target == 0, "ad_id"].nunique()
    print(f"Cheap segment: {len(d)} frames, {n_pos_ads} damaged and {n_neg_ads} intact listings")
    if audit_ads:
        print(
            f"Audit holdout: {audit_ads} new listings excluded from "
            "training/CV; separate evaluation starts once both classes are present"
        )

    ads = per_ad(oof_scores(d, emb))
    zero = zero_shot_per_ad(d)
    ads = ads.merge(zero[["ad_id", "zero_shot"]], on="ad_id", how="left")
    print("\nOOF metrics by independent listing (95% bootstrap CI):")
    for col, name in [
        ("table", "age + price"),
        ("zero_shot", "CLIP zero-shot"),
        ("photo", "CLIP vector, manual labels"),
        ("combined", "tabular + CLIP vector"),
    ]:
        work = ads[["target", col]].dropna()
        auc, lo, hi = auc_ci(work.target, work[col])
        ap, ap_lo, ap_hi = average_precision_ci(work.target, work[col])
        print(
            f"  {name:27} ROC-AUC {auc:.3f} [{lo:.3f}; {hi:.3f}]  "
            f"PR-AUC {ap:.3f} [{ap_lo:.3f}; {ap_hi:.3f}]"
        )
    print("\nPaired difference versus age + price:")
    for col, name in [("photo", "CLIP vector"), ("combined", "tabular + CLIP")]:
        work = ads[["target", "table", col]].dropna()
        delta, lo, hi = auc_delta_ci(work.target, work[col], work.table)
        print(f"  {name:27} ΔAUC {delta:+.3f}  [{lo:+.3f}; {hi:+.3f}]")
    print(
        "\nProduct gate: the photo or combined model must beat age + price, "
        "and the confidence interval for the difference must exclude zero."
    )


if __name__ == "__main__":
    main()
