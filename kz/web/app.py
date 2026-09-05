# -*- coding: utf-8 -*-
"""Web interface for price estimation and manual review.

This module is a thin wrapper around :mod:`kz.web.service` and
:mod:`kz.report.label_cards`: it defines routes and HTML, while business
logic remains testable without starting a server.

The four pages serve different tasks:
  /estimate   vehicle estimate, range, explanation, market position, and checks;
  /label      manual review of market-anomaly candidates;
  /price-review listing-level diagnosis of the difficult below-5M segment;
  /damage     photo and bounding-box labelling for computer vision.

Run:  python -m kz.web
         → http://127.0.0.1:8000
"""

from __future__ import annotations

import html as _html
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from kz.report import label_cards
from kz.web import pages
from kz.web.service import full_estimate

# Public mode is used by the externally hosted container. Price estimation is
# read-only, while verdict labelling writes to data/manual_labels.csv: the
# ground truth used to evaluate anomaly screening. Anonymous users must not be
# able to modify it, so every review page and write endpoint is disabled.
PUBLIC_DEMO = os.getenv("KZ_PUBLIC_DEMO", "").lower() in ("1", "true", "yes")

# Per-address request limit in public mode. Thirty requests per minute is far
# above normal manual use. This is not a complete abuse-prevention system; it
# simply prevents an accidental client loop from exhausting the free instance.
RATE_LIMIT_PER_MIN = 30
RATE_WINDOW_SEC = 60

app = FastAPI(title="KZ Auto Market Intelligence", docs_url="/api/docs")

# Build labelling cards once per process: the database query is expensive and
# the candidate list changes only when the clean layer is rebuilt.
_cards_html: str | None = None
_cards_facts: dict = {}


def _cards():
    global _cards_html, _cards_facts
    if _cards_html is None:
        # Use the complete review queue, not only detector positives. Without a
        # random control sample we can estimate precision but not missed cases
        # or recall. The old standalone server silently used a different queue;
        # the application now has one canonical entry point.
        rows = label_cards.load_rows(include_queue=True)
        _cards_html = label_cards.build(rows, serve_mode=True)
        _cards_facts = label_cards.journal_facts(rows)
    return _cards_html


if PUBLIC_DEMO:
    import time
    from collections import defaultdict, deque

    _hits: dict[str, deque] = defaultdict(deque)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        """Apply a per-address rolling window and discard timestamps older
        than one minute.

        The counter lives in process memory, so multiple instances each get
        their own allowance. That is sufficient for this portfolio demo; a
        distributed limit would require infrastructure such as Redis.
        """
        ip = request.client.host if request.client else "?"
        now = time.monotonic()
        q = _hits[ip]
        while q and now - q[0] > RATE_WINDOW_SEC:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return JSONResponse(
                {"error": "Too many requests; please wait one minute."}, status_code=429
            )
        q.append(now)
        return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def index():
    return pages.index_page()


@app.get("/estimate", response_class=HTMLResponse)
def estimate_form():
    return pages.estimate_page()


@app.post("/api/estimate")
async def api_estimate(request: Request):
    """Estimate a listing price from vehicle characteristics."""
    data = await request.json()
    from kz.ml.train_price_model import CAT_FEATURES, NUM_FEATURES

    car = {k: data.get(k) for k in CAT_FEATURES}
    # Convert every numeric field at the HTTP boundary. Form values arrive as
    # strings, and comparisons such as "8" < 5 otherwise fail with an opaque
    # TypeError. Derive the list from model features so newly added fields do
    # not silently bypass conversion.
    for k in list(NUM_FEATURES) + ["year"]:
        v = data.get(k)
        if v not in (None, ""):
            try:
                car[k] = float(v)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": f"Field {k} must be numeric; received {v!r}."}, status_code=400
                )
    if "year" in car and "age" not in car:
        from datetime import date

        car["age"] = date.today().year - int(car.pop("year")) + 1
    try:
        result = full_estimate(
            car,
            asking_price=float(data["asking_price"]) if data.get("asking_price") else None,
            text=str(data.get("text") or ""),
        )
    except Exception as e:  # noqa: BLE001 — return a client error
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(result)


@app.post("/api/photos/check")
async def api_photos_check(request: Request):
    """Describe uploaded photos. Deliberately does not touch the price.

    The seller form was tabular-only while the model trained on listings that
    carry photographs, which is a train/serve gap of our own making. This
    closes the plumbing without pretending the vision work is finished: every
    supervised photo claim was withdrawn in FINDINGS 28, and FINDINGS 23 found
    no incremental value over age and price.

    Files are held in memory and never written to disk. Storing seller
    photographs would create a personal-data store with no policy behind it,
    and using uploads as training data needs consent nobody has given.
    """
    from kz.web.photo_intake import MAX_BYTES_PER_FILE, MAX_FILES, analyse

    form = await request.form()
    uploads = form.getlist("photos")
    if len(uploads) > MAX_FILES:
        return JSONResponse(
            {"error": f"At most {MAX_FILES} files per request."}, status_code=400
        )

    files: list[tuple[str, bytes]] = []
    for item in uploads:
        read = getattr(item, "read", None)
        if read is None:
            continue
        blob = await read()
        if len(blob) > MAX_BYTES_PER_FILE:
            return JSONResponse(
                {
                    "error": f"{getattr(item, 'filename', 'file')} exceeds "
                             f"{MAX_BYTES_PER_FILE // (1024 * 1024)} MB."
                },
                status_code=400,
            )
        files.append((getattr(item, "filename", "file"), blob))

    return JSONResponse(analyse(files).as_dict())


@app.get("/photos/{path:path}")
def photo(path: str):
    """Return a locally downloaded photo.

    Files are served from disk because one Kolesa CDN host was retired and
    external URLs for 39% of the affected cards no longer resolve.
    """
    from fastapi.responses import FileResponse
    from kz.collect.photo_fetch import PHOTO_DIR

    target = (PHOTO_DIR / path).resolve()
    # Prevent path traversal via ../ segments.
    if PHOTO_DIR.resolve() not in target.parents or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target, media_type="image/jpeg")


_damage_queue: list | None = None
_price_review_cohort = None


def _damage_rows():
    """Build the damage-labelling queue once per process.

    Loading CLIP vectors and querying the database are expensive. The queue
    changes only after a label is written, and the write path updates the cache.
    """
    global _damage_queue
    if _damage_queue is None:
        from kz.report.photo_labels import queue

        q = queue()
        _damage_queue = [
            {
                "ad_id": str(r.ad_id),
                "position": int(r.position),
                "path": str(r.path),
                "suspect": bool(r.suspect),
                "selection_source": str(r.selection_source),
                "dataset_split": str(r.dataset_split),
                "price": (f"{r.price_tenge / 1e6:.1f}M ₸" if pd_notna(r.price_tenge) else ""),
            }
            for r in q.itertuples()
        ]
    return _damage_queue


def pd_notna(v) -> bool:
    import pandas as pd

    return bool(pd.notna(v))


def _price_rows():
    """Build the fixed below-5M pilot once per process."""
    global _price_review_cohort
    if _price_review_cohort is None:
        from kz.report.price_review import load_pilot

        _price_review_cohort = load_pilot()
    return _price_review_cohort


def _price_records() -> tuple[list[dict], list[dict]]:
    """Separate the fixed pilot into pending and recoverable reviewed rows."""
    from kz.report.price_review import journal_by_id
    from kz.web.service import jsonable

    journal = journal_by_id()
    # Deliberately omit OOF predictions/errors from the browser payload.  The
    # annotator should classify observable facts without being pulled toward
    # the model's answer, and timestamps/internal columns only bloat the page.
    shown = [
        "ad_id",
        "brand",
        "model",
        "year",
        "age",
        "price_tenge",
        "mileage_km",
        "engine_volume",
        "transmission",
        "body_type",
        "photos_count",
        "price_basis",
        "description",
        "seller_comment",
        "text_full",
        "damage_keywords",
        "page_status_badge",
        "photos",
        "selection_source",
        "dataset_split",
    ]
    pending, reviewed = [], []
    available = [column for column in shown if column in _price_rows().columns]
    for record in _price_rows()[available].to_dict(orient="records"):
        aid = str(record["ad_id"])
        if aid in journal:
            record.update(journal[aid])
            reviewed.append(jsonable(record))
        else:
            pending.append(jsonable(record))
    return pending, reviewed


def _price_damage_rows(ad_id: str, preferred_position: int | None = None) -> list[dict]:
    """Expose only server-known local pilot frames to precise CV labelling."""
    matches = _price_rows()[_price_rows()["ad_id"].astype(str) == str(ad_id)]
    if matches.empty:
        return []
    row = matches.iloc[0]
    out = [
        {
            "ad_id": str(ad_id),
            "position": int(photo["position"]),
            "path": str(photo["path"]),
            "suspect": True,
            "selection_source": f"price_review:{row['selection_source']}",
            "dataset_split": "audit" if row["dataset_split"] == "audit" else "train",
            "price": f"{float(row['price_tenge']) / 1e6:.1f}M ₸",
        }
        for photo in row["photos"]
    ]
    if preferred_position is not None:
        out.sort(key=lambda item: item["position"] != preferred_position)
    return out


@app.get("/damage", response_class=HTMLResponse)
def damage_page(request: Request):
    """Render bounding-box labelling under the same restriction as /label.

    This page writes irreplaceable manual work to data/photo_labels.csv, so it
    is never exposed by the public demo.
    """
    if PUBLIC_DEMO:
        return HTMLResponse("Labelling is available only in local mode.", status_code=404)
    from kz.report.photo_labels import labelled_frames, stats
    from kz.web.damage_page import page

    # A price-review card can open its currently visible local frame directly.
    # It still writes the exact same frame-level CV journal; the query parameter
    # only changes navigation, never label semantics.
    ad_id = request.query_params.get("ad_id")
    if ad_id:
        raw_position = request.query_params.get("position")
        try:
            position = int(raw_position) if raw_position is not None else None
        except ValueError:
            return HTMLResponse("Invalid photo position.", status_code=400)
        direct = _price_damage_rows(ad_id, position)
        if not direct:
            return HTMLResponse("Listing is not in the local price-review pilot.", status_code=404)
        done = [r for r in labelled_frames() if str(r.get("ad_id")) == str(ad_id)]
        return page(direct, stats(), done)

    # Read completed labels on every request. The same process updates the
    # journal, so caching would show stale labels.
    return page(_damage_rows(), stats(), labelled_frames())


@app.post("/damage/label")
async def damage_label(request: Request):
    """Save a frame label after server-side validation.

    Browser input is untrusted and must never be allowed to corrupt the journal.
    """
    global _damage_queue
    if PUBLIC_DEMO:
        return JSONResponse({"error": "not found"}, status_code=404)
    from kz.report.photo_labels import labelled_frames, save_label, stats

    data = await request.json()
    # A frame is valid if it is queued or already labelled. Completed labels
    # remain editable, but arbitrary browser-supplied paths are rejected.
    known = {(r["ad_id"], r["position"]): r for r in _damage_rows()}
    known.update({(r["ad_id"], r["position"]): r for r in labelled_frames()})
    # Frames reached from /price-review are also trusted only when their local
    # paths belong to the fixed server-side pilot. The browser never supplies
    # or chooses a filesystem path.
    requested_ad = str(data.get("ad_id"))
    known.update({(r["ad_id"], r["position"]): r for r in _price_damage_rows(requested_ad)})
    try:
        key = (str(data.get("ad_id")), int(data.get("position", -1)))
        if key not in known:
            raise ValueError(f"Frame is not in the labelling set: {key}")
        provenance = known[key]
        save_label(
            str(data["ad_id"]),
            int(data["position"]),
            str(provenance["path"]),
            str(data.get("label", "")),
            boxes=data.get("boxes"),
            comment=str(data.get("comment") or ""),
            selection_source=str(provenance.get("selection_source") or "legacy"),
            dataset_split=str(provenance.get("dataset_split") or "train"),
            annotator=str(provenance.get("annotator") or "sanzhar"),
        )
        # Remove the frame from the cached queue immediately. Otherwise a page
        # refresh would offer the newly labelled photo again until restart.
        if _damage_queue is not None:
            _damage_queue = [r for r in _damage_queue if (r["ad_id"], r["position"]) != key]
    except Exception as e:  # noqa: BLE001 — return a client error
        return JSONResponse({"error": _html.escape(str(e))}, status_code=400)
    return JSONResponse({"ok": True, "stats": stats()})


@app.get("/price-review", response_class=HTMLResponse)
def price_review_page():
    """Render the local, blinded pilot for below-5M price failures."""
    if PUBLIC_DEMO:
        return HTMLResponse("Labelling is available only in local mode.", status_code=404)
    from kz.web.price_review_page import page

    pending, reviewed = _price_records()
    return page(pending, reviewed)


@app.post("/price-review/label")
async def price_review_label(request: Request):
    """Save one listing-level condition review after whitelist validation."""
    if PUBLIC_DEMO:
        return JSONResponse({"error": "not found"}, status_code=404)
    from kz.report.price_review import save_review, stats

    data = await request.json()
    ad_id = str(data.get("ad_id", ""))
    matches = _price_rows()[_price_rows()["ad_id"].astype(str) == ad_id]
    try:
        if matches.empty:
            raise ValueError(f"Listing is not in the fixed price-review pilot: {ad_id!r}")
        fact = matches.iloc[0]
        save_review(
            ad_id,
            str(data.get("vehicle_state", "")),
            str(data.get("price_validity", "")),
            str(data.get("evidence_source", "")),
            str(data.get("data_issue", "none")),
            str(data.get("comment", "")),
            selection_source=str(fact["selection_source"]),
            dataset_split=str(fact["dataset_split"]),
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": _html.escape(str(e))}, status_code=400)
    return JSONResponse({"ok": True, "stats": stats()})


@app.get("/label", response_class=HTMLResponse)
def label_page():
    if PUBLIC_DEMO:
        return HTMLResponse("Labelling is available only in local mode.", status_code=404)
    return _cards()


@app.post("/verdict")
async def save_verdict(request: Request):
    """Save an annotator verdict; this is the journal's only HTTP write path."""
    global _cards_html
    if PUBLIC_DEMO:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Facts validate that the browser can mutate only a card served by this
    # process. They remain reusable between clicks; rebuilding the full page
    # before every verdict would repeatedly query the database and photos.
    if not _cards_facts:
        _cards()
    data = await request.json()
    ad_id = str(data.get("ad_id", ""))
    try:
        if ad_id not in _cards_facts:
            raise ValueError(f"Unknown ad_id: {ad_id!r}")
        label_cards.upsert_verdict(
            ad_id, str(data.get("verdict", "")), str(data.get("comment", "")), _cards_facts[ad_id]
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": _html.escape(str(e))}, status_code=400)
    # The next GET must reflect the durable journal. Keep facts warm so a batch
    # of keyboard labels stays fast, but discard the rendered HTML snapshot.
    _cards_html = None
    return JSONResponse({"ok": True})


@app.get("/api/health")
def health():
    from kz.web.service import get_model

    _, meta = get_model()
    val = meta.get("validation", {}).get("grouped_cv", {}).get("model", {})
    return {
        "ok": True,
        "model_created": meta.get("created_at_utc"),
        "training_rows": meta.get("training_rows"),
        "model_mape_pct": val.get("mape_pct"),
        "public_demo": PUBLIC_DEMO,
    }
