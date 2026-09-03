# -*- coding: utf-8 -*-
"""Measure data age so reports can distinguish current evidence from stale data.

The pipeline runs on a laptop rather than continuously. Old lifecycle checks or
gaps in sightings can bias survival and market summaries. This module reports
age facts; each consumer decides whether to warn or suppress a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo


LOCAL_TIMEZONE = ZoneInfo("Asia/Almaty")


def local_date_from_utc_iso(value: str) -> date:
    """Return the artifact date in the market timezone.

    Metadata is stored in UTC. Convert before taking ``.date()`` so an early
    Almaty run is not shown as the previous day. Treat legacy naive values as UTC.
    """
    created = datetime.fromisoformat(value)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(LOCAL_TIMEZONE).date()


@dataclass
class Freshness:
    """Age and coverage facts at the time of measurement."""

    last_collect: date | None
    collect_days: int
    span_days: int
    last_status_check: date | None
    ads_total: int
    ads_status_checked: int
    model_created: date | None

    @property
    def data_age_days(self) -> int | None:
        if self.last_collect is None:
            return None
        return (date.today() - self.last_collect).days

    @property
    def status_age_days(self) -> int | None:
        if self.last_status_check is None:
            return None
        return (date.today() - self.last_status_check).days

    @property
    def model_age_days(self) -> int | None:
        if self.model_created is None:
            return None
        return (date.today() - self.model_created).days

    @property
    def status_coverage(self) -> float:
        """Share of listings whose status has been checked at least once."""
        return self.ads_status_checked / self.ads_total if self.ads_total else 0.0

    @property
    def collect_regularity(self) -> float:
        """Share of calendar days with collection; 1.0 means every day."""
        return self.collect_days / self.span_days if self.span_days else 0.0


def measure() -> Freshness:
    """Measure each freshness fact directly from the database."""
    import pandas as pd

    from kz.core.db import get_engine

    eng = get_engine()

    def scalar(sql: str):
        return pd.read_sql(sql, eng).iloc[0, 0]

    days = pd.read_sql("SELECT DISTINCT seen_date FROM sightings", eng)
    seen = pd.to_datetime(days["seen_date"]).dt.date.tolist() if len(days) else []

    last_check = scalar("SELECT MAX(checked_at) FROM ad_status")
    model_created = None
    try:
        from kz.ml.train_price_model import load_artifact

        _, meta = load_artifact()
        model_created = local_date_from_utc_iso(meta["created_at_utc"])
    except Exception:  # noqa: BLE001 — artifact is optional
        pass

    return Freshness(
        last_collect=max(seen) if seen else None,
        collect_days=len(seen),
        span_days=(max(seen) - min(seen)).days + 1 if seen else 0,
        last_status_check=pd.to_datetime(last_check).date() if last_check else None,
        ads_total=int(scalar("SELECT COUNT(*) FROM clean_data")),
        ads_status_checked=int(scalar("SELECT COUNT(*) FROM ad_status")),
        model_created=model_created,
    )


def report(f: Freshness, log=print) -> None:
    """Print the freshness preamble used by reports."""

    def age(n, what):
        if n is None:
            return f"{what}: never"
        if n == 0:
            return f"{what}: today"
        return f"{what}: {n} days ago"

    log("Data freshness:")
    log(
        f"  {age(f.data_age_days, 'last collection')}, "
        f"{age(f.status_age_days, 'status checks')}, "
        f"{age(f.model_age_days, 'model training')}"
    )
    log(
        f"  status checked for {f.ads_status_checked} of {f.ads_total} "
        f"listings ({f.status_coverage * 100:.0f}%)"
    )
    log(
        f"  collection ran on {f.collect_days} of {f.span_days} days "
        f"({f.collect_regularity * 100:.0f}% of the period)"
    )


def stale_warnings(f: Freshness, status_days: int = 7, coverage: float = 0.5) -> list[str]:
    """Return conclusions weakened by data age or coverage.

    Thresholds are deliberately permissive; warnings explain uncertainty rather
    than blocking every report.
    """
    out = []
    if f.status_age_days is not None and f.status_age_days > status_days:
        out.append(
            f"Statuses were last checked {f.status_age_days} days ago. Listings "
            f"removed since then may still appear active, inflating lifetimes."
        )
    if f.status_coverage < coverage:
        out.append(
            f"Only {f.status_coverage * 100:.0f}% of listings have a status check. "
            f"Others are active by default rather than by observation."
        )
    if f.collect_regularity < 0.5 and f.span_days > 7:
        out.append(
            f"Collection ran on {f.collect_regularity * 100:.0f}% of days. Price "
            f"and sighting histories contain gaps, so event dates are imprecise."
        )
    return out
