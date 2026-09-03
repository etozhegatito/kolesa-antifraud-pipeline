# -*- coding: utf-8 -*-
"""Implementation for the `kz.collect.photo_fetch` module."""

import pathlib as _p

_expected = "photo_fetch.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named {_p.Path(__file__).name}."
    )

import csv
import io
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps

from kz.collect.enrich import HEADERS
from kz.core import pacing
from kz.core.db import get_engine
from kz.ops.catch_up import DAILY_BUDGET, charge_budget, load_budget_used

PHOTO_DIR = Path("data/photos")
MANIFEST = PHOTO_DIR / "manifest.csv"
LOG_FILE = "logs/photo_fetch.log"

MAX_PER_RUN = 300
MAX_SIDE = 768
JPEG_QUALITY = 85
DELAY_RANGE = (0.8, 2.0)
BREAK_EVERY = 120


MAX_CONSECUTIVE_FAILS = 10

MANIFEST_COLS = [
    "ad_id",
    "position",
    "url",
    "path",
    "http_status",
    "width",
    "height",
    "bytes",
    "fetched_at",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def local_path(ad_id: str, position: int) -> Path:
    """Implement `local_path`."""
    return PHOTO_DIR / str(ad_id)[:2] / f"{ad_id}_{position}.jpg"


def load_manifest() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST, dtype={"ad_id": str})
    return pd.DataFrame(columns=MANIFEST_COLS)


def append_manifest(rows: list[dict]) -> None:
    """Implement `append_manifest`."""
    if not rows:
        return
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fresh = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if fresh:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLS})


PERMANENT_STATUSES = {200, 404, 410}


def live_hosts(urls) -> set[str]:
    """Implement `live_hosts`."""
    import socket

    hosts = {u.split("/")[2] for u in urls}
    alive = set()
    for h in sorted(hosts):
        try:
            socket.getaddrinfo(h, 443)
            alive.add(h)
        except OSError:
            log.warning(f"Host {h} does not resolve; skipping its URLs")
    return alive


def pick_targets(limit: int, covers_only: bool = True, complete_only: bool = False) -> pd.DataFrame:
    """Implement `pick_targets`."""
    ph = pd.read_sql("SELECT ad_id, position, url FROM photos", get_engine(), dtype={"ad_id": str})
    ph = ph[ph["url"].fillna("").str.startswith("http")]
    if covers_only:
        ph = ph[ph["position"] == ph.groupby("ad_id")["position"].transform("min")]

    man = load_manifest()
    if complete_only and len(man):
        started = set(man.loc[man["http_status"] == 200, "ad_id"])
        ph = ph[ph["ad_id"].isin(started)]
    if len(man):
        done = set(man.loc[man["http_status"].isin(PERMANENT_STATUSES), "url"])
        ph = ph[~ph["url"].isin(done)]
    ph = ph[~ph.apply(lambda r: local_path(r.ad_id, r.position).exists(), axis=1)]

    alive = live_hosts(ph["url"])
    ph = ph[ph["url"].str.split("/").str[2].isin(alive)]
    return ph.sort_values(["position", "ad_id"]).head(limit)


def save_image(content: bytes, dest: Path) -> tuple[int, int, int]:
    """Implement `save_image`."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(content)))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return img.width, img.height, dest.stat().st_size


def main():
    limit = MAX_PER_RUN
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    covers_only = "--all-positions" not in sys.argv

    complete_only = "--complete" in sys.argv
    if complete_only:
        covers_only = False

    used = load_budget_used()
    left = DAILY_BUDGET["cdn"] - used["cdn"]
    if left <= 0:
        log.info(
            f"The rolling CDN budget is exhausted ({used['cdn']}/{DAILY_BUDGET['cdn']}); "
            "resume when the window frees up."
        )
        return
    limit = min(limit, left)

    targets = pick_targets(limit, covers_only, complete_only)
    if targets.empty:
        log.info("Nothing to download: every image is present or marked unavailable.")
        return
    log.info(
        f"Queued {len(targets)} images "
        f"({'cover photos' if covers_only else 'all positions'}); "
        f"CDN budget {used['cdn']}/{DAILY_BUDGET['cdn']}"
    )

    session = requests.Session()
    rows, ok, fails, streak = [], 0, 0, 0
    for i, r in enumerate(targets.itertuples(index=False), 1):
        dest = local_path(r.ad_id, r.position)
        try:
            resp = session.get(r.url, headers=HEADERS, timeout=20)
            status = resp.status_code
            if status == 200:
                w, h, size = save_image(resp.content, dest)
                rows.append(
                    dict(
                        ad_id=r.ad_id,
                        position=r.position,
                        url=r.url,
                        path=str(dest),
                        http_status=200,
                        width=w,
                        height=h,
                        bytes=size,
                        fetched_at=datetime.now().isoformat(timespec="seconds"),
                    )
                )
                ok += 1
                streak = 0
            else:
                rows.append(
                    dict(
                        ad_id=r.ad_id,
                        position=r.position,
                        url=r.url,
                        path="",
                        http_status=status,
                        width="",
                        height="",
                        bytes="",
                        fetched_at=datetime.now().isoformat(timespec="seconds"),
                    )
                )
                fails += 1
                streak += 1
        except Exception as e:  # noqa: BLE001 -- intentional exception
            log.warning(f"{r.ad_id}/{r.position}: {e}")
            rows.append(
                dict(
                    ad_id=r.ad_id,
                    position=r.position,
                    url=r.url,
                    path="",
                    http_status=-1,
                    width="",
                    height="",
                    bytes="",
                    fetched_at=datetime.now().isoformat(timespec="seconds"),
                )
            )
            fails += 1
            streak += 1

        if streak >= MAX_CONSECUTIVE_FAILS:
            log.error("Stopped after too many consecutive failures; resume later.")
            break
        if i % 50 == 0:
            append_manifest(rows)
            rows = []
            log.info(f"  {i}/{len(targets)}: downloaded {ok}, failures {fails}")
        pacing.polite_sleep(i, DELAY_RANGE, log, break_every=BREAK_EVERY)

    append_manifest(rows)
    charge_budget("cdn", ok + fails)
    mb = sum(int(x["bytes"]) for x in rows if x.get("bytes")) / 1e6
    log.info(
        f"Completed: downloaded {ok} ({mb:.1f} MB), failures {fails}. "
        f"Files currently on disk: {sum(1 for _ in PHOTO_DIR.rglob('*.jpg'))}"
    )


if __name__ == "__main__":
    main()
