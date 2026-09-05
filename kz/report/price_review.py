# -*- coding: utf-8 -*-
"""Human review for the hard below-5M price segment.

This journal deliberately does not feed the price model.  Its first job is to
measure *why* inexpensive listings are difficult: physical condition, a
non-comparable advertised amount, bad structured data, or genuinely normal
market variation.  Only a later automated text/photo feature may enter the
model, and only after grouped and temporal validation.

The pilot cohort is fixed before completed rows are removed.  Otherwise every
page refresh would replace a reviewed listing and silently turn a 50-listing
pilot into an unlimited moving queue.  It contains:

* 30 old vehicles with the largest grouped-OOF errors;
* 10 random inexpensive controls;
* 10 random audit listings selected before error ranking.

Only already-downloaded photos are shown.  Manual review therefore performs no
network requests and cannot consume or bypass the Kolesa request budget.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kz.collect.photo_fetch import PHOTO_DIR, local_path
from kz.ml.train_price_model import (
    OOF_DIAGNOSTICS_PATH,
    duplicate_groups,
    load,
    prepare_training_data,
)

_DIR = os.environ.get("KZ_LABELS_DIR", "data")
LABELS_CSV = str(Path(_DIR) / "price_review_labels.csv")
LABELS_PREV = str(Path(_DIR) / "price_review_labels.prev.csv")
PILOT_CSV = str(Path(_DIR) / "price_review_pilot.csv")

PILOT_SIZE = 50
AUDIT_SIZE = 10
HIGH_ERROR_SIZE = 30
RANDOM_CONTROL_SIZE = 10
OLD_AGE_MIN = 21
CHEAP_PRICE_MAX = 5_000_000
MIN_LOCAL_PHOTOS = 3
LABEL_VERSION = "1"
PILOT_VERSION = "1"

VEHICLE_STATES = {
    "normal": "No material problem is visible or disclosed",
    "cosmetic": "Rust, scratches, paint wear, or other cosmetic deterioration",
    "repair_needed": "Local impact or mechanical repair is needed",
    "non_running": "The vehicle does not run or needs towing",
    "wreck": "Major crash damage affects a whole assembly",
    "parts": "Dismantled vehicle or donor/parts-only listing",
    "unclear": "The available evidence is insufficient",
}

PRICE_VALIDITY = {
    "comparable_cash": "A normal cash price for the advertised vehicle",
    "cash_uncleared": "Cash price without customs clearance",
    "credit_or_down_payment": "Credit-only amount or down payment",
    "parts_price": "Price refers to parts or a donor vehicle",
    "exchange_or_placeholder": "Exchange, placeholder, or deliberately conditional amount",
    "ambiguous": "The meaning of the advertised amount is unclear",
}

EVIDENCE_SOURCES = {
    "both": "Supported by both description and photos",
    "photos": "Supported by photos only",
    "text": "Supported by description/structured text only",
    "neither": "No special condition evidence is present",
    "unclear": "Evidence source cannot be determined",
}

DATA_ISSUES = {
    "none": "Structured fields look usable",
    "wrong_specs": "Year, model, engine, mileage, or another field looks wrong",
    "missing_critical_details": "Critical information is missing",
    "duplicate_or_repost": "Possible duplicate/repost identity issue",
    "unclear": "Data quality cannot be determined",
}

HEADER = [
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

PILOT_COLUMNS = [
    "ad_id",
    "url",
    "title",
    "brand",
    "model",
    "price_tenge",
    "year",
    "mileage_km",
    "engine_volume",
    "engine_type",
    "transmission",
    "body_type",
    "condition",
    "city",
    "description",
    "photos_count",
    "views_count",
    "posted_date",
    "labels",
    "is_vip",
    "has_monthly_price",
    "scraped_at",
    "customs_cleared",
    "drive",
    "steering",
    "color",
    "generation",
    "page_mileage_km",
    "damage_keywords",
    "seller_comment",
    "kolesa_avg_price",
    "page_status_badge",
    "text_full",
    "price_basis",
    "age",
    "is_mileage_missing",
    "is_description_missing",
    "duplicate_group",
    "routed_pred_tenge",
    "base_pred_tenge",
    "baseline_pred_tenge",
    "absolute_percentage_error_pct",
    "dataset_split",
    "selection_source",
    "photo_positions",
    "pilot_version",
]


def split_for_ad(ad_id: str) -> str:
    """Return a deterministic 20% audit split before any model ranking."""
    bucket = int(hashlib.sha256(str(ad_id).encode()).hexdigest()[:8], 16) % 100
    return "audit" if bucket < 20 else "discovery"


def align_oof(training: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Attach saved OOF diagnostics only when row identity is provably exact.

    The historical diagnostics intentionally contain no listing text or URL.
    Their row order matches the training cohort, but relying on order without
    checking it would risk showing one vehicle's error beside another vehicle.
    Duplicate-group, actual-price, and age equality make that failure loud.
    """
    train = training.reset_index(drop=True).copy()
    report = oof.reset_index(drop=True).copy()
    if len(train) != len(report):
        raise ValueError(
            "The OOF diagnostics do not match the current training cohort; "
            "run `python -m kz.ops.run_all --ml` before price review."
        )

    expected_groups = duplicate_groups(train).astype(str).reset_index(drop=True)
    actual_groups = report["duplicate_group"].astype(str).reset_index(drop=True)
    same_groups = expected_groups.equals(actual_groups)
    same_price = np.allclose(
        pd.to_numeric(train["price_tenge"], errors="coerce"),
        pd.to_numeric(report["actual_price_tenge"], errors="coerce"),
        equal_nan=True,
    )
    same_age = np.allclose(
        pd.to_numeric(train["age"], errors="coerce"),
        pd.to_numeric(report["age"], errors="coerce"),
        equal_nan=True,
    )
    if not (same_groups and same_price and same_age):
        raise ValueError(
            "The OOF diagnostics are stale or reordered; refusing to build a "
            "price-review queue with uncertain listing identity."
        )

    train["duplicate_group"] = expected_groups
    for col in (
        "routed_pred_tenge",
        "base_pred_tenge",
        "baseline_pred_tenge",
        "absolute_percentage_error_pct",
    ):
        train[col] = pd.to_numeric(report[col], errors="coerce").to_numpy()
    return train


def load_candidates() -> pd.DataFrame:
    """Load inexpensive train-eligible listings with local photo evidence."""
    from kz.core.db import get_engine

    if not OOF_DIAGNOSTICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {OOF_DIAGNOSTICS_PATH}; run `python -m kz.ops.run_all --ml` first."
        )
    training = prepare_training_data(load())
    oof = pd.read_csv(OOF_DIAGNOSTICS_PATH, dtype={"duplicate_group": str})
    rows = align_oof(training, oof)
    rows["ad_id"] = rows["ad_id"].astype(str)
    rows["age"] = pd.to_numeric(rows["age"], errors="coerce")
    rows = rows[pd.to_numeric(rows["price_tenge"], errors="coerce") < CHEAP_PRICE_MAX].copy()

    photos = pd.read_sql(
        "SELECT ad_id, position FROM photos ORDER BY ad_id, position",
        get_engine(),
        dtype={"ad_id": str},
    )
    galleries: dict[str, list[dict]] = {}
    for r in photos.itertuples(index=False):
        path = local_path(str(r.ad_id), int(r.position))
        if not path.is_file():
            continue
        galleries.setdefault(str(r.ad_id), []).append(
            {
                "position": int(r.position),
                "path": str(path),
                "src": f"/photos/{path.relative_to(PHOTO_DIR)}",
            }
        )
    rows["photos"] = rows["ad_id"].map(galleries)
    rows = rows[rows["photos"].map(lambda value: isinstance(value, list) and bool(value))].copy()
    # One cover image can hide exactly the defect we are trying to discover.
    # Require several local viewpoints rather than inviting a confident
    # listing-level condition label from one seller-selected angle.
    rows = rows[rows["photos"].map(len) >= MIN_LOCAL_PHOTOS].copy()
    rows["dataset_split"] = rows["ad_id"].map(split_for_ad)
    return rows.reset_index(drop=True)


def select_pilot(candidates: pd.DataFrame, limit: int = PILOT_SIZE) -> pd.DataFrame:
    """Select one deterministic, fixed pilot cohort.

    Audit rows are sampled before OOF-error ranking.  Random controls are then
    taken outside the high-error set, which lets the later analysis distinguish
    causes of model failure from ordinary prevalence in the cheap segment.
    """
    if candidates.empty or limit <= 0:
        return candidates.head(0).copy()

    n_audit = min(AUDIT_SIZE, limit)
    audit_pool = candidates[candidates["dataset_split"] == "audit"]
    audit = audit_pool.sample(n=min(n_audit, len(audit_pool)), random_state=1905).copy()
    # This is random only inside the already-downloaded-photo population.  It
    # must not be used to claim prevalence for every cheap listing until photo
    # acquisition itself is randomized.
    audit["selection_source"] = "random_local_audit"

    remaining = candidates[
        (candidates["dataset_split"] != "audit") & ~candidates["ad_id"].isin(audit["ad_id"])
    ].copy()
    high_pool = remaining[remaining["age"] >= OLD_AGE_MIN]
    high = high_pool.nlargest(
        min(HIGH_ERROR_SIZE, max(0, limit - len(audit))), "absolute_percentage_error_pct"
    ).copy()
    high["selection_source"] = "old_high_oof_error"

    random_pool = remaining[~remaining["ad_id"].isin(high["ad_id"])]
    random_n = min(RANDOM_CONTROL_SIZE, max(0, limit - len(audit) - len(high)))
    random = random_pool.sample(n=min(random_n, len(random_pool)), random_state=4207).copy()
    random["selection_source"] = "random_cheap_control"

    selected = pd.concat([audit, high, random], ignore_index=True)
    if len(selected) < limit:
        used = set(selected["ad_id"])
        fill = candidates[~candidates["ad_id"].isin(used)].sample(
            n=min(limit - len(selected), len(candidates) - len(used)), random_state=7301
        )
        fill = fill.copy()
        fill["selection_source"] = "pilot_fill"
        selected = pd.concat([selected, fill], ignore_index=True)

    # Mix strata so the annotator cannot infer that the next listing is meant
    # to be a failure case or a control from its position in the queue.
    return selected.head(limit).sample(frac=1.0, random_state=117).reset_index(drop=True)


def save_pilot(pilot: pd.DataFrame) -> None:
    """Persist the fixed cohort before target-policy or model changes.

    The manifest contains only already-local facts and photo positions.  It is
    deliberately separate from the mutable review journal so a later clean or
    retrain cannot replace completed cases with a new high-error queue.
    """
    if pilot.empty:
        raise ValueError("Cannot persist an empty price-review pilot")
    work = pilot.copy()
    work["ad_id"] = work["ad_id"].astype(str)
    if work["ad_id"].duplicated().any():
        raise ValueError("Cannot persist a price-review pilot with duplicate ad_id values")
    if len(work) > PILOT_SIZE:
        raise ValueError(f"Price-review pilot exceeds the fixed {PILOT_SIZE}-listing limit")

    work["photo_positions"] = work["photos"].map(
        lambda gallery: ",".join(str(int(photo["position"])) for photo in gallery)
    )
    work["pilot_version"] = PILOT_VERSION
    for column in PILOT_COLUMNS:
        if column not in work:
            work[column] = ""

    path = Path(PILOT_CSV)
    if path.exists():
        existing = pd.read_csv(path, dtype={"ad_id": str})
        if existing["ad_id"].astype(str).tolist() != work["ad_id"].tolist():
            raise ValueError("Refusing to replace the existing fixed price-review pilot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    work[PILOT_COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_pilot() -> pd.DataFrame:
    """Load the durable cohort and rebuild galleries from local files only."""
    path = Path(PILOT_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed price-review pilot: {path}")
    rows = pd.read_csv(path, dtype={"ad_id": str, "duplicate_group": str})
    if rows.empty or rows["ad_id"].duplicated().any():
        raise ValueError("The fixed price-review pilot is empty or contains duplicate ad_id values")
    versions = set(rows["pilot_version"].fillna("").astype(str))
    if versions != {PILOT_VERSION}:
        raise ValueError(f"Unsupported price-review pilot versions: {sorted(versions)}")

    def gallery(row: pd.Series) -> list[dict]:
        value = row.get("photo_positions")
        raw = "" if pd.isna(value) else str(value)
        out = []
        for token in raw.split(","):
            if not token.strip():
                continue
            position = int(token)
            path = local_path(str(row["ad_id"]), position)
            if path.is_file():
                out.append(
                    {
                        "position": position,
                        "path": str(path),
                        "src": f"/photos/{path.relative_to(PHOTO_DIR)}",
                    }
                )
        return out

    rows["photos"] = rows.apply(gallery, axis=1)
    return rows


def load_pilot() -> pd.DataFrame:
    """Return the durable pilot, creating it once from aligned OOF rows."""
    if Path(PILOT_CSV).exists():
        return read_pilot()
    pilot = select_pilot(load_candidates())
    save_pilot(pilot)
    return read_pilot()


_snapshot_done = False


def _snapshot_once() -> None:
    global _snapshot_done
    if _snapshot_done or not Path(LABELS_CSV).exists():
        _snapshot_done = True
        return
    shutil.copyfile(LABELS_CSV, LABELS_PREV)
    _snapshot_done = True


def read_journal() -> tuple[list[str], list[dict]]:
    """Read raw strings so identifiers and manual values are never coerced."""
    path = Path(LABELS_CSV)
    if not path.exists():
        return list(HEADER), []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or HEADER), list(reader)


def _write_journal(rows: list[dict]) -> None:
    Path(LABELS_CSV).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(LABELS_CSV + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HEADER})
    os.replace(tmp, LABELS_CSV)


def save_review(
    ad_id: str,
    vehicle_state: str,
    price_validity: str,
    evidence_source: str,
    data_issue: str = "none",
    comment: str = "",
    *,
    selection_source: str,
    dataset_split: str,
    annotator: str = "sanzhar",
) -> None:
    """Validate and atomically insert or update one listing-level review."""
    if vehicle_state not in VEHICLE_STATES:
        raise ValueError(f"Unknown vehicle_state: {vehicle_state!r}")
    if price_validity not in PRICE_VALIDITY:
        raise ValueError(f"Unknown price_validity: {price_validity!r}")
    if evidence_source not in EVIDENCE_SOURCES:
        raise ValueError(f"Unknown evidence_source: {evidence_source!r}")
    if data_issue not in DATA_ISSUES:
        raise ValueError(f"Unknown data_issue: {data_issue!r}")
    if dataset_split not in {"discovery", "audit"}:
        raise ValueError(f"Unknown dataset_split: {dataset_split!r}")

    _snapshot_once()
    _, rows = read_journal()
    record = {
        "ad_id": str(ad_id),
        "vehicle_state": vehicle_state,
        "price_validity": price_validity,
        "evidence_source": evidence_source,
        "data_issue": data_issue,
        "comment": str(comment).strip(),
        "labeled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_source": selection_source,
        "dataset_split": dataset_split,
        "annotator": annotator,
        "label_version": LABEL_VERSION,
    }
    index = next((i for i, row in enumerate(rows) if str(row.get("ad_id")) == str(ad_id)), None)
    if index is None:
        rows.append(record)
    else:
        rows[index] = record
    _write_journal(rows)


def stats() -> dict[str, int]:
    """Return durable journal counts, including all state categories."""
    _, rows = read_journal()
    counts = {key: 0 for key in VEHICLE_STATES}
    for row in rows:
        state = row.get("vehicle_state")
        if state in counts:
            counts[state] += 1
    counts["reviewed"] = sum(counts.values())
    return counts


def journal_by_id() -> dict[str, dict]:
    """Return the latest durable record for each listing."""
    _, rows = read_journal()
    return {str(row.get("ad_id")): row for row in rows if row.get("ad_id")}
