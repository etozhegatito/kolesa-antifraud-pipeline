# -*- coding: utf-8 -*-
"""Photo-damage labelling queue and journal.

The under-5M-tenge segment produces most of the price error because old-vehicle
condition is absent from listing-table fields. Full-frame zero-shot CLIP did
not solve the problem: local dents disappear among road and sky. Tiling raised
damage AUC from 0.776 to 0.827 but reduced rust AUC, confirming that impact is
a local signal. Bounding boxes therefore provide cleaner supervision.

Coordinates are stored relative to image size in the 0..1 range, so browser
resizing cannot invalidate them. Crops are not saved as new images: immutable
source photos stay separate from editable annotations.

The journal follows the same safety rules as ``data/manual_labels.csv``:
existing frame rows are updated in place, writes are atomic, and a recovery
snapshot is created before the first mutation.

Run ``python -m kz.report.photo_labels`` to inspect the queue or add ``--stats``
to print current annotation statistics. Use ``python -m kz.web`` for labelling.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# The environment override is a safety boundary, not just configuration.
# Manual server tests must never write into the real annotation journal.
#
#     KZ_LABELS_DIR=/tmp/scratch python -m kz.web
#
# Always use a scratch directory for manual or automated UI checks.
_DIR = os.environ.get("KZ_LABELS_DIR", "data")
LABELS_CSV = str(Path(_DIR) / "photo_labels.csv")
LABELS_PREV = str(Path(_DIR) / "photo_labels.prev.csv")
LABELS_REVIEW_BACKUP = str(Path(_DIR) / "photo_labels.pre_definition_review.csv")

HEADER = [
    "ad_id",
    "position",
    "path",
    "label",
    "x1",
    "y1",
    "x2",
    "y2",
    "comment",
    "labeled_at",
    "selection_source",
    "dataset_split",
    "annotator",
    "label_version",
    "boxes_json",
    "review_status",
]

# New listings receive a deterministic split from ad_id. Legacy labels cannot
# become a holdout retroactively because they already influenced experiments.
AUDIT_PERCENT = 20
AUDIT_PER_QUEUE = 60
LABEL_VERSION = "3"
MAX_BOXES_PER_FRAME = 20
NEEDS_REVIEW = "needs_review"
REVIEWED = "reviewed"


def split_for_ad(ad_id: str) -> str:
    """Return a stable train/audit split independent of row order."""
    bucket = int(hashlib.sha256(str(ad_id).encode()).hexdigest()[:8], 16) % 100
    return "audit" if bucket < AUDIT_PERCENT else "train"


# ``unclear`` prevents an annotator from forcing ambiguous frames into an
# incorrect confident class.
LABELS = {
    "damaged": "local impact, dent, crease, or broken part; draw a box",
    "wreck": "major crash with a destroyed front/rear assembly",
    "parts": "dismantled vehicle or removed major component",
    "intact": "no impact or dent; rust and scuffs still belong here",
    "unclear": "cannot determine because of darkness, angle, or crop",
}

# The interface uses these exact keys. A former broad translation caused
# definition drift by including rust and scuffs in `damaged`, although the
# target class means local impact, dents, or broken parts.
#
# The damaged/wreck boundary is operational rather than subjective:
#
#   one local box is meaningful  → damaged
#   a local box is meaningless   → wreck; evidence is the whole frame
#
# Separate classes keep a small dataset homogeneous and preserve future
# options: classes can be merged later, but mixed labels cannot be separated
# after collection. `parts` also uses whole-frame evidence.
#
# Rust is labelled `intact` for this particular impact task and recorded in the
# comment. Zero-shot CLIP already detects rust (historical AUC 0.881), while
# impact remains the missing signal. Tiling improves local dents but harms the
# global rust signal, so combining them would weaken both tasks.
#
# Pure random sampling would yield only a few positives at roughly 1% prevalence.
# The queue therefore combines likely positives with negative controls.
CONTROL_PER_POSITIVE = 2


def queue(limit: int = 400) -> pd.DataFrame:
    """Build a queue from likely positives plus negative controls.

    Text and site badges enrich the positive yield. Random controls keep the
    training set and evaluation from containing only flagged vehicles.
    """
    from kz.core.db import get_engine
    from kz.ml.photo_clip import load_embeddings

    idx, _ = load_embeddings()
    cd = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge, price_tenge, age FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    cd["suspect"] = (cd.damage_keywords.fillna("").str.len() > 0) | cd.page_status_badge.fillna(
        "-"
    ).str.contains("вар|ход|залож", case=False)
    d = idx.merge(cd, on="ad_id", how="left")
    d["suspect"] = d.suspect.fillna(False)

    done = {(r["ad_id"], str(r["position"])) for r in read_journal()[1]}
    d = d[~d.apply(lambda r: (r.ad_id, str(r.position)) in done, axis=1)]

    d["dataset_split"] = d.ad_id.map(split_for_ad)
    d["selection_source"] = np.where(d.suspect, "text_or_badge", "random_control")

    # Select audit rows before model ranking or text prioritization so the audit
    # remains random rather than becoming another active-learning slice.
    audit_pool = d[d.dataset_split == "audit"].copy()
    n_audit = min(AUDIT_PER_QUEUE, limit, len(audit_pool))
    audit = audit_pool.sample(n=n_audit, random_state=29) if n_audit else audit_pool.head(0)
    audit["selection_source"] = "random_audit"

    train = _mark_candidates(d[d.dataset_split == "train"].copy())
    remaining = max(0, limit - len(audit))
    pos = train[train.suspect]
    per_pos = CONTROL_PER_POSITIVE if _negatives_so_far() < ENOUGH_NEGATIVES else 0
    n_ctrl = min(
        len(train[~train.suspect]),
        max(0, remaining - len(pos)),
        (len(pos) * per_pos) if per_pos else CONTROL_WHEN_ENOUGH,
    )
    ctrl = train[~train.suspect].sample(n=n_ctrl, random_state=42) if n_ctrl else train.head(0)
    out = pd.concat([audit, pos, ctrl]).head(limit)
    # Mix strata to reduce annotator expectation bias.
    out = out.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return _body_first(out)


# Rank and sample by independent listings rather than frames because grouped
# validation treats multiple photos of one vehicle as one unit.
RANK_TOP_ADS = 120
FRAMES_PER_AD = 2

# Once several hundred negatives exist, reserve only a small control sample so
# queue capacity shifts toward scarce positives without losing prevalence checks.
CONTROL_WHEN_ENOUGH = 60
ENOUGH_NEGATIVES = 200


def _negatives_so_far() -> int:
    """Count verified negative frames used to size the next control sample."""
    return stats()["intact"]


def _mark_candidates(d: pd.DataFrame) -> pd.DataFrame:
    """Take top-ranked listings and at most a few frames from each.

    Text and badges prioritize what a seller wrote; ranking adds how the photo
    looks. Aggregate with the maximum score so one convincing frame is enough
    and listings with many photos do not win merely by volume. The final queue
    remains shuffled to reduce annotator expectation bias.
    """
    from kz.ml.photo_clip import load_damage_rank

    rank = load_damage_rank()
    if rank is None or d.empty:
        return d
    rank = rank.copy()
    rank["position"] = rank.position.astype(int)
    m = d.merge(rank, on=["ad_id", "position"], how="left")
    m["damage_rank"] = m.damage_rank.fillna(-1.0)

    by_ad = m.groupby("ad_id").damage_rank.max().nlargest(RANK_TOP_ADS)
    pick = (
        m[m.ad_id.isin(by_ad.index)]
        .sort_values("damage_rank", ascending=False)
        .groupby("ad_id")
        .head(FRAMES_PER_AD)
        .index
    )
    newly_ranked = pick[~m.loc[pick, "suspect"].to_numpy()]
    m.loc[pick, "suspect"] = True
    m.loc[newly_ranked, "selection_source"] = "model_rank"
    return m.drop(columns=["damage_rank"])


def _body_first(q: pd.DataFrame) -> pd.DataFrame:
    """Move likely no-body frames to the end without discarding them.

    Interiors, engine bays, wheel close-ups, and documents cannot show body
    impact and account for about 12% of images. The threshold is imperfect, so
    exclusion would risk silently losing useful exterior frames. Stable sorting
    preserves the random order within both sections.
    """
    from kz.ml.photo_clip import NO_BODY_THRESHOLD, load_no_body

    nb = load_no_body()
    if nb is None or q.empty:
        return q
    m = q.merge(nb, on=["ad_id", "position"], how="left")
    m["_tail"] = (m["clip_no_body"].fillna(-1.0) > NO_BODY_THRESHOLD).astype(int)
    m = m.sort_values("_tail", kind="stable")
    return m.drop(columns=["_tail", "clip_no_body"]).reset_index(drop=True)


# Journal

_snapshot_done = False


def _snapshot_once() -> None:
    """Save one recovery snapshot before the process's first mutation."""
    global _snapshot_done
    if _snapshot_done or not Path(LABELS_CSV).exists():
        _snapshot_done = True
        return
    shutil.copyfile(LABELS_CSV, LABELS_PREV)
    _snapshot_done = True


def read_journal() -> tuple[list[str], list[dict]]:
    """Read raw journal values with ``csv`` to preserve integer strings."""
    p = Path(LABELS_CSV)
    if not p.exists():
        return list(HEADER), []
    with p.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or HEADER), list(r)


def write_journal(header: list[str], rows: list[dict]) -> None:
    """Write atomically through a temporary file and replacement."""
    Path(LABELS_CSV).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(LABELS_CSV + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in header})
    os.replace(tmp, LABELS_CSV)


def is_training_label(row: dict) -> bool:
    """Return whether a label is cleared for CV training and evaluation."""
    return str(row.get("review_status") or "") != NEEDS_REVIEW


def mark_legacy_damaged_for_review() -> int:
    """Non-destructively send legacy ``damaged`` labels back for review.

    A former broad class name allowed rust, dirt, and scuffs into the impact
    class. Do not guess corrections or delete work: add a status, create a
    dedicated backup, and exclude disputed rows until manual review.
    """
    header, rows = read_journal()
    pending = [
        r
        for r in rows
        if r.get("label") == "damaged" and not str(r.get("review_status") or "").strip()
    ]
    if not pending:
        return 0
    source = Path(LABELS_CSV)
    backup = Path(LABELS_REVIEW_BACKUP)
    if source.exists() and not backup.exists():
        shutil.copyfile(source, backup)
    header = list(dict.fromkeys([*header, *HEADER]))
    for row in pending:
        row["review_status"] = NEEDS_REVIEW
    write_journal(header, rows)
    return len(pending)


def _normalise_boxes(boxes) -> list[tuple[float, float, float, float]]:
    """Validate relative boxes and return coordinates in the 0..1 range."""
    if boxes is None:
        return []
    if not isinstance(boxes, (list, tuple)):
        raise ValueError("boxes must be a list of bounding boxes")
    if len(boxes) > MAX_BOXES_PER_FRAME:
        raise ValueError(f"Too many boxes; maximum is {MAX_BOXES_PER_FRAME}")

    out = []
    for raw in boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"A box must contain four coordinates: {raw!r}")
        try:
            x1, y1, x2, y2 = (float(v) for v in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Box coordinates must be numeric: {raw!r}") from e
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Box is outside the image or inverted: {raw}")
        out.append((x1, y1, x2, y2))
    return out


def boxes_from_row(row: dict) -> list[tuple[float, float, float, float]]:
    """Read all row boxes, treating legacy x1..y2 as a one-box list."""
    payload = str(row.get("boxes_json") or "").strip()
    if payload:
        try:
            return _normalise_boxes(json.loads(payload))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid boxes_json for {row.get('path')}: {e}") from e
    if row.get("x1") not in (None, ""):
        return _normalise_boxes([[row.get(k) for k in ("x1", "y1", "x2", "y2")]])
    return []


def save_label(
    ad_id: str,
    position,
    path: str,
    label: str,
    box=None,
    boxes=None,
    comment: str = "",
    selection_source: str = "manual",
    dataset_split: str = "train",
    annotator: str | None = None,
) -> None:
    """Insert or update one frame label without creating duplicates.

    ``boxes`` contains relative ``(x1, y1, x2, y2)`` tuples. ``box`` is a
    backward-compatible single-box argument. JSON stores every box, while
    legacy x1..y2 columns mirror the first one for older research scripts.
    """
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label!r}")
    if box is not None and boxes is not None:
        raise ValueError("Pass either box or boxes, not both")
    frame_boxes = _normalise_boxes(
        boxes if boxes is not None else ([box] if box is not None else [])
    )
    if label == "damaged" and not frame_boxes:
        raise ValueError("The damaged label requires at least one box")
    # Boxes are allowed with every label and required only for `damaged`.
    # Preserving a rust region on an `intact` frame avoids silently discarding
    # manual work and supports future localization analysis.
    _snapshot_once()
    header, rows = read_journal()
    # Migrate schema only on an intentional write. Importing this module never
    # rewrites the existing journal.
    header = list(dict.fromkeys([*header, *HEADER]))
    key = (str(ad_id), str(position))
    if dataset_split not in {"train", "audit"}:
        raise ValueError(f"Unknown dataset_split: {dataset_split!r}")
    first = frame_boxes[0] if frame_boxes else None
    row = {
        "ad_id": str(ad_id),
        "position": str(position),
        "path": path,
        "label": label,
        "x1": f"{first[0]:.4f}" if first else "",
        "y1": f"{first[1]:.4f}" if first else "",
        "x2": f"{first[2]:.4f}" if first else "",
        "y2": f"{first[3]:.4f}" if first else "",
        "comment": comment,
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
        "selection_source": selection_source,
        "dataset_split": dataset_split,
        "annotator": annotator or os.environ.get("KZ_ANNOTATOR", "sanzhar"),
        "label_version": LABEL_VERSION,
        "boxes_json": (
            json.dumps([[round(v, 4) for v in b] for b in frame_boxes], separators=(",", ":"))
            if frame_boxes
            else ""
        ),
        # Any new or repeated manual decision follows v3 and clears review.
        "review_status": REVIEWED,
    }
    for i, r in enumerate(rows):
        if (r.get("ad_id"), str(r.get("position"))) == key:
            rows[i] = row
            break
    else:
        rows.append(row)
    write_journal(header, rows)


def labelled_frames() -> list[dict]:
    """Return completed frames for in-page review and correction.

    The work queue intentionally excludes them, but the clickable counters must
    still make previous decisions editable.
    """
    _, rows = read_journal()
    out = []
    for r in rows:
        if r.get("label") not in LABELS:
            continue
        rec = {
            "ad_id": r.get("ad_id", ""),
            "position": int(r.get("position") or 0),
            "path": r.get("path", ""),
            "label": r["label"],
            "comment": r.get("comment", ""),
            "selection_source": r.get("selection_source", "legacy"),
            "dataset_split": r.get("dataset_split", "train") or "train",
            "annotator": r.get("annotator", ""),
            "label_version": r.get("label_version", "1") or "1",
            "review_status": r.get("review_status", ""),
        }
        boxes = boxes_from_row(r)
        if boxes:
            rec["boxes"] = [list(b) for b in boxes]
            rec |= dict(zip(("x1", "y1", "x2", "y2"), boxes[0]))
        out.append(rec)
    return out


def stats() -> dict:
    """Count labelled frames and independent listings.

    UI progress uses frames; grouped validation uses listings because five
    photos of one vehicle are not five independent observations.
    """
    _, rows = read_journal()
    out = dict.fromkeys(LABELS, 0)
    out["damage_boxes"] = 0
    ads = {label: set() for label in LABELS}
    for r in rows:
        if r.get("label") in out:
            out[r["label"]] += 1
            ads[r["label"]].add(str(r.get("ad_id", "")))
            if r["label"] == "damaged":
                out["damage_boxes"] += len(boxes_from_row(r))
    out["total"] = len(rows)
    out["ads_total"] = len(set().union(*ads.values()))
    for label in LABELS:
        out[f"{label}_ads"] = len(ads[label])
    out["positive_ads"] = len(ads["damaged"] | ads["wreck"])
    review_rows = [r for r in rows if r.get("review_status") == NEEDS_REVIEW]
    out["needs_review"] = len(review_rows)
    verified_positive = {
        str(r.get("ad_id", ""))
        for r in rows
        if r.get("label") in {"damaged", "wreck"} and is_training_label(r)
    }
    out["verified_positive_ads"] = len(verified_positive)
    audit_rows = [r for r in rows if r.get("dataset_split") == "audit"]
    out["audit_frames"] = len(audit_rows)
    out["audit_ads"] = len({str(r.get("ad_id", "")) for r in audit_rows})
    return out


def main():
    if "--mark-legacy-review" in sys.argv:
        changed = mark_legacy_damaged_for_review()
        print(
            f"Marked needs_review: {changed}. Nothing was deleted; backup: {LABELS_REVIEW_BACKUP}"
        )
        return
    if "--stats" in sys.argv:
        s = stats()
        print(f"Labelled frames: {s['total']}")
        for k, desc in LABELS.items():
            print(f"  {k:9} {s[k]:4} frames, {s[f'{k}_ads']:3} listings   {desc}")
        need = 200 - s["verified_positive_ads"]
        print(
            f"\nIndependent damaged/wreck listings: {s['positive_ads']} total, "
            f"{s['verified_positive_ads']} verified for CV."
        )
        if s["needs_review"]:
            print(
                f"Needs review: {s['needs_review']} frames are excluded from "
                "training and metrics until relabelled."
            )
        print(f"Local damage boxes: {s['damage_boxes']}")
        print(
            f"New random audit holdout: {s['audit_ads']} listings, "
            f"{s['audit_frames']} frames (legacy labels are never moved into it)."
        )
        print(
            "Target for a stable local evaluation is about 200: "
            f"{'enough' if need <= 0 else f'{need} more listings'}"
        )
        for k in ("parts", "wreck"):
            if s[k]:
                print(f"{k}: {s[k]} — keep as a separate class during training or merge later")
        return

    q = queue()
    print(f"Frames in queue: {len(q)}   (flagged listings: {int(q.suspect.sum())})")
    print(f"Already labelled: {stats()['total']}")
    print("\nOpen labelling:  python -m kz.web  →  http://127.0.0.1:8000/damage")


if __name__ == "__main__":
    main()
