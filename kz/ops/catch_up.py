# -*- coding: utf-8 -*-
"""Implementation for the `kz.ops.catch_up` module."""

import pathlib as _p

_expected = "catch_up.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named {_p.Path(__file__).name}."
    )

import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta

import fcntl

import pandas as pd

from kz.core import pacing
from kz.core.db import get_engine

LINE = "─" * 64


DEFAULT_KOLESA_BUDGET = 200


DEFAULT_CDN_BUDGET = 1200
DAILY_BUDGET = {
    "kolesa": int(os.environ.get("KOLESA_BUDGET", DEFAULT_KOLESA_BUDGET)),
    "cdn": int(os.environ.get("CDN_BUDGET", DEFAULT_CDN_BUDGET)),
}
BUDGET_FILE = "logs/.catch_up_budget.json"


RISK_ZONES = [
    (100, "low", "repeatedly exercised in this project without a restriction"),
    (200, "normal", "default operating range; no restriction was observed at this level"),
    (
        270,
        "elevated",
        "approaches the observed restriction: the IP was blocked near 270 on 2026-07-23",
    ),
    (10**9, "high", "exceeds an observed restriction threshold; do not proceed"),
]


def risk_zone(n: int):
    """Implement `risk_zone`."""
    for limit, label, note in RISK_ZONES:
        if n <= limit:
            return label, note
    return RISK_ZONES[-1][1], RISK_ZONES[-1][2]


def eta_minutes(n: int, lo: float = 4.0, hi: float = 8.0) -> float:
    """Implement `eta_minutes`."""
    return n * (pacing.mean_pause(lo, hi) + 3.0) / 60


def parse_budget(argv) -> int | None:
    """Implement `parse_budget`."""
    for i, a in enumerate(argv):
        raw = None
        if a == "--budget" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif a.startswith("--budget="):
            raw = a.split("=", 1)[1]
        if raw is not None:
            try:
                n = int(raw)
            except ValueError:
                raise SystemExit(f"--budget must be an integer; received {raw!r}") from None
            if n < 1:
                raise SystemExit("--budget must be >= 1")
            return n
    return None


def print_risk_help(current: int):
    """Implement `print_risk_help`."""
    print("\nKolesa requests in a rolling 24-hour window (--budget N):")
    prev = 0
    for limit, label, note in RISK_ZONES:
        rng = f"{prev + 1}–{limit}" if limit < 10**8 else f"{prev + 1}+"
        print(f"  {rng:<10} {label:<14} — {note}")
        prev = limit
    label, _ = risk_zone(current)
    print(
        f"\nCurrent ceiling: {current} ({label}); "
        f"a full allocation takes approximately {eta_minutes(current):.0f} minutes."
    )
    print("Set it with: python -m kz.ops.catch_up --run --backfill --budget 300")
    print(
        "The parser and catch_up share this counter. Manual Kolesa browsing "
        "is not counted, so do not run it alongside full collection."
    )


# test_catch_up_chunk_sizes_match_jobs.

CHUNK_MAX = {"status": 20, "enrich": 20, "backfill": 20, "photo": 300}


STATUS_STALE_DAYS = 2
STATUS_RECHECK_DAYS = 7


BUDGET_SCHEMA_VERSION = 2
BUDGET_WINDOW_HOURS = 24
BUDGET_EVENT_KEEP_HOURS = 48
BUDGET_KEEP_DAYS = 7


class BudgetStateError(RuntimeError):
    """Implementation of `BudgetStateError`."""


def _now() -> datetime:
    """Implement `_now`."""
    return datetime.now().astimezone()


def _read_state() -> dict:
    """Implement `_read_state`."""
    try:
        d = json.loads(_p.Path(BUDGET_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": BUDGET_SCHEMA_VERSION, "days": {}, "events": []}
    except (OSError, ValueError) as e:
        raise BudgetStateError(
            f"Budget file {BUDGET_FILE} is corrupt or unreadable; "
            f"network access is blocked to prevent an accidental quota reset: {e}"
        ) from e
    if not isinstance(d, dict):
        raise BudgetStateError(
            f"Budget file {BUDGET_FILE} must contain a JSON object; network access is blocked"
        )
    if isinstance(d.get("days"), dict):
        days = d["days"]
    elif d.get("date"):
        days = {d["date"]: {"kolesa": int(d.get("kolesa", 0)), "cdn": int(d.get("cdn", 0))}}
    else:
        days = {}
    events = d.get("events") if isinstance(d.get("events"), list) else []
    return {"schema_version": int(d.get("schema_version", 1)), "days": days, "events": events}


def _read_days() -> dict:
    """Implement `_read_days`."""
    return _read_state()["days"]


def _parse_event_time(raw, fallback_tz) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=fallback_tz)


def _migrate_legacy_state(state: dict, now: datetime) -> dict:
    """Implement `_migrate_legacy_state`."""
    if int(state.get("schema_version", 1)) >= BUDGET_SCHEMA_VERSION:
        return state
    today = now.date()
    yesterday = today - timedelta(days=1)
    migrated = []
    for raw_day, used in state.get("days", {}).items():
        try:
            day = datetime.strptime(raw_day, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if day == today:
            at = now
        elif day == yesterday:
            at = datetime.combine(day, dt_time(23, 59, 59), tzinfo=now.tzinfo)
        else:
            continue
        for host in ("kolesa", "cdn"):
            cost = int((used or {}).get(host, 0))
            if cost > 0:
                migrated.append({"at": at.isoformat(), "host": host, "cost": cost, "legacy": True})
    state["events"] = migrated
    state["schema_version"] = BUDGET_SCHEMA_VERSION
    return state


def _active_events(state: dict, now: datetime) -> list[dict]:
    cutoff = now - timedelta(hours=BUDGET_WINDOW_HOURS)
    active = []
    for event in state.get("events", []):
        at = _parse_event_time(event.get("at"), now.tzinfo)
        try:
            cost = int(event.get("cost", 0))
        except (TypeError, ValueError):
            continue
        if at is None or cost <= 0 or event.get("host") not in {"kolesa", "cdn"}:
            continue
        if at > cutoff:
            clean = {"at": at.isoformat(), "host": event["host"], "cost": cost}
            if event.get("legacy"):
                clean["legacy"] = True
            active.append(clean)
    return active


def _rolling_used(state: dict, now: datetime) -> dict:
    used = {"kolesa": 0, "cdn": 0}
    for event in _active_events(state, now):
        used[event["host"]] += event["cost"]
    return used


def _write_state(state: dict, now: datetime | None = None):
    now = now or _now()
    days = dict(state.get("days", {}))
    for old in sorted(days)[:-BUDGET_KEEP_DAYS]:
        days.pop(old)
    event_cutoff = now - timedelta(hours=BUDGET_EVENT_KEEP_HOURS)
    events = []
    for event in state.get("events", []):
        at = _parse_event_time(event.get("at"), now.tzinfo)
        if at is not None and at > event_cutoff:
            events.append(event)
    target = _p.Path(BUDGET_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(
                {"schema_version": BUDGET_SCHEMA_VERSION, "days": days, "events": events},
                out,
                sort_keys=True,
            )
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, target)
    finally:
        _p.Path(tmp_name).unlink(missing_ok=True)


def _write_days(days: dict):
    """Implement `_write_days`."""
    _write_state({"schema_version": 1, "days": days, "events": []})


@contextmanager
def _budget_lock():
    """Implement `_budget_lock`."""
    lock_path = _p.Path(str(BUDGET_FILE) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_budget_used() -> dict:
    """Implement `load_budget_used`."""
    with _budget_lock():
        now = _now()
        state = _read_state()
        was_legacy = int(state.get("schema_version", 1)) < BUDGET_SCHEMA_VERSION
        state = _migrate_legacy_state(state, now)
        if was_legacy:
            _write_state(state, now)
        return _rolling_used(state, now)


def save_budget_used(used: dict):
    """Implement `save_budget_used`."""
    with _budget_lock():
        now = _now()
        state = _migrate_legacy_state(_read_state(), now)
        state["events"] = [
            {"at": now.isoformat(), "host": host, "cost": int(used[host])}
            for host in ("kolesa", "cdn")
            if int(used[host]) > 0
        ]
        state["days"][now.date().isoformat()] = {
            "kolesa": int(used["kolesa"]),
            "cdn": int(used["cdn"]),
        }
        _write_state(state, now)


def charge_budget(host: str, cost: int) -> dict:
    """Implement `charge_budget`."""
    with _budget_lock():
        now = _now()
        state = _migrate_legacy_state(_read_state(), now)
        key = now.date().isoformat()
        cur = state["days"].setdefault(key, {"kolesa": 0, "cdn": 0})
        cur[host] = int(cur.get(host, 0)) + int(cost)
        state["events"].append({"at": now.isoformat(), "host": host, "cost": int(cost)})
        _write_state(state, now)
        return _rolling_used(state, now)


def reserve_budget(host: str, cost: int, limit: int) -> dict | None:
    """Implement `reserve_budget`."""
    with _budget_lock():
        now = _now()
        state = _migrate_legacy_state(_read_state(), now)
        used = _rolling_used(state, now)
        if int(used.get(host, 0)) + int(cost) > int(limit):
            return None
        key = now.date().isoformat()
        cur = state["days"].setdefault(key, {"kolesa": 0, "cdn": 0})
        cur[host] = int(cur.get(host, 0)) + int(cost)
        state["events"].append({"at": now.isoformat(), "host": host, "cost": int(cost)})
        _write_state(state, now)
        used[host] += int(cost)
        return used


def compute_gaps() -> dict:
    """Implement `compute_gaps`."""
    eng = get_engine()
    g = {}

    last_seen = pd.read_sql(
        "SELECT ad_id, MAX(seen_date) AS seen FROM sightings GROUP BY ad_id",
        eng,
        dtype={"ad_id": str},
    )
    st = pd.read_sql("SELECT ad_id, status, checked_at FROM ad_status", eng, dtype={"ad_id": str})
    ls = last_seen.merge(st, on="ad_id", how="left")
    today = pd.Timestamp.today().normalize()
    seen_days = (today - pd.to_datetime(ls["seen"])).dt.days
    checked_days = (today - pd.to_datetime(ls["checked_at"])).dt.days
    terminal = ls["status"].isin(["archived", "deleted"])
    recently_checked = checked_days < STATUS_RECHECK_DAYS  # NaN<7 → False
    g["status"] = int(((~terminal) & (seen_days >= STATUS_STALE_DAYS) & (~recently_checked)).sum())

    clean_ids = set(pd.read_sql("SELECT ad_id FROM clean_data", eng, dtype={"ad_id": str})["ad_id"])
    enr = pd.read_sql(
        "SELECT ad_id, kolesa_avg_price, page_status_badge, http_status FROM enriched",
        eng,
        dtype={"ad_id": str},
    )
    g["enrich"] = len(clean_ids - set(enr["ad_id"]))

    ok = enr[enr["http_status"] == 200]
    g["backfill"] = int((ok["kolesa_avg_price"].isna() | ok["page_status_badge"].isna()).sum())
    g["enriched_total"] = int(len(ok))

    photos = pd.read_sql("SELECT url FROM photos", eng)
    photos = photos[photos["url"].fillna("").str.startswith("http")]
    hashed = set(pd.read_sql("SELECT url FROM photo_hashes", eng)["url"])
    g["photo"] = int((~photos["url"].isin(hashed)).sum())

    return g


KOLESA = [
    ("listing statuses (check_status)", "kz.collect.check_status", "status"),
    ("detail enrichment (enrich)", "kz.collect.enrich", "enrich"),
    ("avgPrice + badge (backfill)", "kz.collect.backfill_avgprice", "backfill"),
]
CDN = [("photo hashes (photo_dedup)", "kz.collect.photo_dedup", "photo")]
OFFLINE = [("clean layer (clean)", "kz.transform.clean"), ("report (explore)", "kz.report.explore")]


VALUE_JOBS = [j for j in KOLESA if j[2] in ("enrich", "backfill")]


BACKFILL_JOBS = [j for j in KOLESA if j[2] == "backfill"]


def is_429_line(line: str) -> bool:
    """Implement `is_429_line`."""
    normalized = line.lower()
    return "429" in normalized and (
        "paus" in normalized
        or "consecutive" in normalized
        or "пауза" in normalized
        or "подряд" in normalized
    )


def count_429() -> int:
    """Implement `count_429`."""
    n = 0
    for f in glob.glob("logs/*.log"):
        try:
            n += sum(
                is_429_line(ln)
                for ln in _p.Path(f).read_text(encoding="utf-8", errors="ignore").splitlines()
            )
        except OSError:
            pass
    return n


def run(script: str) -> int:
    print(f"\n{'═' * 60}\n▶ {script}\n{'═' * 60}")
    return subprocess.run([sys.executable, "-m", script]).returncode


def next_action(gap_before: int, gap_after: int, rc: int, saw_new_429: bool) -> str:
    """Implement `next_action`."""
    if gap_after == 0:
        return "done"
    if saw_new_429:
        return "rate_limited"
    if rc != 0:
        return "breaker"
    if gap_after >= gap_before:
        return "stuck"
    return "continue"


def budget_allows(
    host: str, key: str, gap_before: int, used: dict, run_spent: dict | None = None
) -> bool:
    """Implement `budget_allows`."""
    cost = min(CHUNK_MAX[key], gap_before)
    if used[host] + cost > DAILY_BUDGET[host]:
        return False
    return run_spent is None or run_spent[host] + cost <= DAILY_BUDGET[host]


def run_one_chunk(
    name: str, script: str, key: str, host: str, used: dict, run_spent: dict | None = None
) -> str:
    """Implement `run_one_chunk`."""
    gap_before = compute_gaps()[key]
    if gap_before == 0:
        return "done"

    used.update(load_budget_used())
    if not budget_allows(host, key, gap_before, used, run_spent):
        return "budget"

    cost = min(CHUNK_MAX[key], gap_before)
    reserved = reserve_budget(host, cost, DAILY_BUDGET[host])
    if reserved is None:
        used.update(load_budget_used())
        return "budget"
    used.update(reserved)
    if run_spent is not None:
        run_spent[host] += cost
    before_429 = count_429()
    print(
        f"\n  {name}: {gap_before} remaining; {host} budget "
        f"{used[host]}/{DAILY_BUDGET[host]}; running a chunk (about {cost} requests)..."
    )
    rc = run(script)
    saw_429 = count_429() > before_429
    gap_after = compute_gaps()[key]
    action = next_action(gap_before, gap_after, rc, saw_429)
    print(
        f"  {name}: gap {gap_before} → {gap_after}; {host} budget {used[host]}/{DAILY_BUDGET[host]}"
    )
    return "progress" if action == "continue" else action


def drain_host(
    jobs, host: str, used: dict, until_done: bool, run_spent: dict | None = None
) -> bool:
    """Implement `drain_host`."""
    blocked = set()
    while True:
        progressed = False
        for name, script, key in jobs:
            if key in blocked:
                continue
            outcome = run_one_chunk(name, script, key, host, used, run_spent)
            if outcome in ("rate_limited", "breaker"):
                print(
                    f"\nWARNING: {name}: {outcome}; stopping jobs for host '{host}' "
                    "to protect the shared IP."
                )
                return True
            if outcome == "progress":
                progressed = True
                continue
            blocked.add(key)
            if outcome == "done":
                print(f"OK: {name}: no gaps remain")
            elif outcome == "stuck":
                print(
                    f"WARNING: {name}: the chunk did not reduce the gap; "
                    "remaining rows cannot be filled (404/no data), skipping"
                )
            elif outcome == "budget":
                print(
                    f"PAUSED: {name}: the rolling 24-hour '{host}' quota is nearly exhausted "
                    f"({used[host]}/{DAILY_BUDGET[host]}); wait for the window to free up"
                )
        if not until_done or not progressed or len(blocked) == len(jobs):
            return False


def report(g: dict, title: str):
    print(f"\n{LINE}\n{title}\n{LINE}")
    labels = {
        "status": "statuses to check",
        "enrich": "not enriched",
        "backfill": "avgPrice/badge missing",
        "photo": "photos not hashed",
    }
    for k in ["status", "enrich", "backfill", "photo"]:
        mark = "—" if g[k] == 0 else str(g[k])
        extra = (
            f"  (of {g['enriched_total']} enriched rows)"
            if k == "backfill" and g.get("enriched_total")
            else ""
        )
        print(f"  {labels[k]:<28} {mark}{extra}")
    print(LINE)


def run_gapped_jobs(until_done: bool = False, kolesa_jobs=None, do_cdn: bool = True):
    """Implement `run_gapped_jobs`."""
    used = load_budget_used()
    run_spent = {"kolesa": 0, "cdn": 0}
    t0 = time.time()

    kolesa_jobs = KOLESA if kolesa_jobs is None else kolesa_jobs
    kolesa_aborted = drain_host(kolesa_jobs, "kolesa", used, until_done, run_spent)
    if do_cdn:
        drain_host(CDN, "cdn", used, until_done, run_spent)
    if kolesa_aborted:
        print(
            "\n(Kolesa jobs stopped after a site signal. The CDN is a separate host, "
            "so its backlog can still be processed.)"
        )

    for _name, script in OFFLINE:
        run(script)

    print(f"\ncatch_up completed in {(time.time() - t0) / 60:.1f} minutes")
    print(
        f"  rolling 24-hour budget: kolesa {used['kolesa']}/{DAILY_BUDGET['kolesa']}, "
        f"CDN {used['cdn']}/{DAILY_BUDGET['cdn']}"
    )
    report(compute_gaps(), "REMAINING AFTER THE RUN")


def main():
    until_done = "--until-done" in sys.argv
    backfill_only = "--backfill" in sys.argv
    values = "--values" in sys.argv and not backfill_only

    cli_budget = parse_budget(sys.argv)
    if cli_budget is not None:
        DAILY_BUDGET["kolesa"] = cli_budget

    g = compute_gaps()
    report(g, "CURRENT GAPS")

    used = load_budget_used()
    label, note = risk_zone(DAILY_BUDGET["kolesa"])
    print(
        f"Rolling request budget (used in the past 24 hours): "
        f"kolesa {used['kolesa']}/{DAILY_BUDGET['kolesa']} [{label}], "
        f"CDN {used['cdn']}/{DAILY_BUDGET['cdn']}"
    )
    if label in ("elevated", "high"):
        print(f"  ⚠ {label}: {note}")
    print_risk_help(DAILY_BUDGET["kolesa"])

    if backfill_only:
        kolesa_jobs, do_cdn, net = BACKFILL_JOBS, False, g["backfill"]
    elif values:
        kolesa_jobs, do_cdn, net = VALUE_JOBS, False, g["enrich"] + g["backfill"]
    else:
        kolesa_jobs, do_cdn = KOLESA, True
        net = g["status"] + g["enrich"] + g["backfill"] + g["photo"]
    if net == 0:
        print("\nNothing remains to process in the selected mode.")
        return

    if backfill_only:
        print(
            f"\n--backfill mode: fill only avgPrice and badges for enriched rows "
            f"({g['backfill']} of {g['enriched_total']}); skip completed rows."
        )
        print("New enrichment, statuses, and photos are not touched.")
        doable = min(g["backfill"], max(0, DAILY_BUDGET["kolesa"] - used["kolesa"]))
        print(
            f"Fits in the remaining rolling budget: {doable} "
            f"of {g['backfill']} (approximately {eta_minutes(doable):.0f} minutes)."
        )
    elif values:
        print("\n--values mode: enrichment plus avgPrice/badge backfill,")
        print("without status or photo jobs; intended for quick anomaly-review preparation.")
    if until_done:
        print("--until-done: use the remaining 24-hour quota round-robin; stop on 429 or budget.")
    elif not (values or backfill_only):
        print("\nOne pass over all jobs within budget. Focused modes: --values / --backfill.")

    flags = (
        (" --until-done" if until_done else "")
        + (" --backfill" if backfill_only else (" --values" if values else ""))
        + (f" --budget {cli_budget}" if cli_budget is not None else "")
    )
    if "--run" in sys.argv:
        run_gapped_jobs(until_done, kolesa_jobs, do_cdn)
        return
    if not sys.stdin.isatty():
        print(f"\nRun with: python -m kz.ops.catch_up --run{flags}")
        return
    ans = input("\nRun catch-up now? [y/N] ").strip().lower()
    if ans in ("y", "yes"):
        run_gapped_jobs(until_done, kolesa_jobs, do_cdn)
    else:
        print(f"Not started. Run later with: python -m kz.ops.catch_up --run{flags}")


if __name__ == "__main__":
    main()
