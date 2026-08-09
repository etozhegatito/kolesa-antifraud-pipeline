# -*- coding: utf-8 -*-
"""Row counts per table, plus deltas between two moments.

Why this exists: an Airflow task that just says "success" tells you nothing.
What you actually want to know after a run is how much data arrived — how many
new ads were parsed, how many rows reached the database. So the collect DAG
takes a snapshot before the network steps and prints the delta afterwards.

The snapshot lives in logs/ as JSON, so it survives between tasks: each Airflow
task is a separate process and cannot pass Python objects to the next one.

Usage:
    python -m kz.ops.db_stats            print current counts
    python -m kz.ops.db_stats --save     save a snapshot (before a run)
    python -m kz.ops.db_stats --diff     compare with the snapshot, print delta
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from kz.core.db import get_engine

SNAPSHOT_FILE = "logs/.db_snapshot.json"

# Pipeline layers in the order data flows through them, so the report reads
# top-down: raw input first, derived tables last.
TABLES = [
    ("raw_ads", "ads collected"),
    ("sightings", "price observations"),
    ("photos", "photo links"),
    ("ad_status", "status checks"),
    ("enriched", "pages enriched"),
    ("photo_hashes", "photo hashes"),
    ("clean_data", "clean rows"),
    ("photo_duplicates", "duplicate photo pairs"),
]


def table_counts() -> dict[str, int]:
    """Current row count per table. Missing tables are reported as 0 rather
    than raising: photo_duplicates and clean_data only appear after the first
    full run, and a stats helper should not be the thing that breaks."""
    counts = {}
    with get_engine().connect() as conn:
        for name, _ in TABLES:
            try:
                counts[name] = int(conn.execute(
                    text(f"SELECT COUNT(*) FROM {name}")).scalar())
            except Exception:
                counts[name] = 0
    return counts


def save_snapshot(path: str = SNAPSHOT_FILE) -> dict:
    counts = table_counts()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "taken_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
    }, indent=2), encoding="utf-8")
    return counts


def load_snapshot(path: str = SNAPSHOT_FILE) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def format_counts(counts: dict[str, int], before: dict[str, int] | None = None) -> str:
    """Aligned table of counts, with a delta column when a baseline is given."""
    lines = []
    width = max(len(label) for _, label in TABLES)
    for name, label in TABLES:
        now = counts.get(name, 0)
        if before is None:
            lines.append(f"  {label:<{width}}  {now:>7,}")
            continue
        delta = now - before.get(name, 0)
        mark = f"{delta:+,}" if delta else "no change"
        lines.append(f"  {label:<{width}}  {now:>7,}   {mark:>12}")
    return "\n".join(lines).replace(",", " ")


def main():
    if "--save" in sys.argv:
        counts = save_snapshot()
        print("Baseline snapshot saved. Current state:")
        print(format_counts(counts))
        return

    counts = table_counts()
    if "--diff" in sys.argv:
        snap = load_snapshot()
        if not snap:
            print("No baseline snapshot found, showing current state only.")
            print(format_counts(counts))
            return
        print(f"Changes since {snap['taken_at_utc'][:16]} UTC:")
        print(format_counts(counts, snap["counts"]))
        total = sum(counts.values()) - sum(snap["counts"].values())
        print(f"\n  Total rows added: {total:+,}".replace(",", " "))
        return

    print("Current row counts:")
    print(format_counts(counts))


if __name__ == "__main__":
    main()
