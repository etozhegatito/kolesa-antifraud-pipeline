# -*- coding: utf-8 -*-
"""Verdict journal: the only module that writes manual anomaly labels.

The journal is never rebuilt from a changing queue or truncated. Verdicts
remain valid after a listing leaves the candidate set. Writes are atomic, a
single recovery snapshot is taken before the first mutation, and the standard
``csv`` module preserves integer strings that Pandas round-trips as ``50.0``.
"""

import csv
import os
import shutil
from pathlib import Path

import pandas as pd

_DIR = os.environ.get("KZ_LABELS_DIR", "data")
LABELS_CSV = str(Path(_DIR) / "manual_labels.csv")
# One recovery point from before the current process's first mutation.
LABELS_PREV = str(Path(_DIR) / "manual_labels.prev.csv")

VERDICTS = ("fraud", "legit", "unknown")


# Sampling strata belong in the journal, not only in the disposable work queue.
# Without this metadata, completed controls cannot be used to estimate misses.
STRATUM_COLS = ["sampling_stratum", "stratum_population"]

BASE_HEADER = [
    "ad_id",
    "url",
    "title",
    "year",
    "price_tenge",
    "mileage_km",
    "suspicion_reasons",
    "seller_comment",
    "verdict",
    "comment",
]


def journal_header() -> list[str]:
    """Use the journal's own column order and append missing stratum fields."""
    head = None
    if Path(LABELS_CSV).exists():
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            head = next(csv.reader(f), None)
    head = list(head) if head else list(BASE_HEADER)
    for c in STRATUM_COLS:
        if c not in head:
            head.append(c)
    return head


def _cell(v) -> str:
    """Format a CSV cell: missing becomes empty and integers lose ``.0``."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v)


_snapshot_done = False


def _snapshot_once() -> None:
    """Save one recovery snapshot before the process's first journal write."""
    global _snapshot_done
    if _snapshot_done:
        return
    _snapshot_done = True
    if Path(LABELS_CSV).exists():
        shutil.copyfile(LABELS_CSV, LABELS_PREV)


def read_journal() -> tuple[list[str], list[dict]]:
    """Read journal rows as dictionaries without reformatting stored values."""
    if not Path(LABELS_CSV).exists():
        return journal_header(), []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [dict(x) for x in r]
        return list(r.fieldnames or journal_header()), rows


def write_journal(header: list[str], rows: list[dict]) -> None:
    """Write atomically through a temporary file to prevent truncation."""
    Path(LABELS_CSV).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(LABELS_CSV) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in header})
    os.replace(tmp, LABELS_CSV)


def upsert_verdict(ad_id: str, verdict: str, comment: str, facts: dict) -> None:
    """Insert a verdict or update the existing row for the same ``ad_id``.

    Repeated clicks must not create contradictory duplicate rows. The first
    row keeps its position, later duplicates are removed, and the recovery
    snapshot preserves the previous journal state.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    _snapshot_once()
    header, rows = read_journal()
    aid = str(ad_id)
    same = [r for r in rows if str(r.get("ad_id", "")) == aid]
    if same:
        target = same[0]  # update the first row in place
        keep = set(id(r) for r in same[1:])  # remove later duplicates
        rows = [r for r in rows if id(r) not in keep]
    else:
        target = {c: "" for c in header}
        target.update({c: _cell(facts.get(c)) for c in header if c in facts})
        target["ad_id"] = aid
        rows.append(target)
    target["verdict"] = verdict
    target["comment"] = comment or ""
    write_journal(header, rows)


def dedupe_journal() -> tuple[int, int]:
    """Collapse duplicates to one row per listing.

    The last non-empty verdict wins while the first row keeps its position.
    Return ``(rows_before, rows_after)``.
    """
    header, rows = read_journal()
    before = len(rows)
    _snapshot_once()
    order, best = [], {}
    for r in rows:
        aid = str(r.get("ad_id", ""))
        if aid not in best:
            order.append(aid)
            best[aid] = dict(r)
            continue
        # A non-empty verdict overrides; an empty value never erases a decision.
        if str(r.get("verdict", "")).strip():
            best[aid]["verdict"] = r["verdict"]
            best[aid]["comment"] = r.get("comment", "")
    out = [best[a] for a in order]
    write_journal(header, out)
    return before, len(out)


def journal_facts(rows: pd.DataFrame) -> dict:
    """Map ``ad_id`` to descriptive fields stored with a journal verdict."""
    out = {}
    for _, r in rows.iterrows():
        out[str(r["ad_id"])] = {
            "sampling_stratum": r.get("stratum") or "",
            "url": r.get("url") or f"https://kolesa.kz/a/show/{r['ad_id']}",
            "title": f"{r.get('brand') or ''} {r.get('model') or ''}".strip(),
            "year": r.get("year"),
            "price_tenge": r.get("price_tenge"),
            "mileage_km": r.get("mileage_km"),
            "suspicion_reasons": r.get("suspicion_reasons"),
            "seller_comment": r.get("seller_comment"),
        }
    return out
