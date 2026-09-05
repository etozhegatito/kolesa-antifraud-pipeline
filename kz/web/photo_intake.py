# -*- coding: utf-8 -*-
"""Describe seller-uploaded photos without letting them touch the price.

WHY THIS EXISTS AND WHAT IT DELIBERATELY DOES NOT DO

The price model was trained on listings that carry photographs, yet the
seller form accepts tabular fields only. That gap is a train/serve skew we
created ourselves, and closing the plumbing is worthwhile even while the
vision work is unfinished.

The price is NOT adjusted here, and no caller may adjust it from these
results. Every supervised photo claim was withdrawn in FINDINGS 28 after the
labelling definition turned out to have drifted, and FINDINGS 23 showed
full-frame CLIP adding nothing over age and price. A condition score shown
to a seller today would be an unvalidated number that most likely just
re-encodes vehicle age.

So this module reports only statements that can be checked by looking at the
picture:

  * how many files were received, and which ones are not readable images;
  * pixel size, and which frames are too small to show damage;
  * exact duplicates by content hash, and near duplicates by perceptual hash
    where imagehash is installed;
  * whether the frame shows vehicle bodywork at all.

The last one is the only learned signal used, and it is the only photo axis
this project has validated against its own hand labels: 0.986 ROC-AUC over
117 frames a reviewer marked as cabin, engine bay, wheel, or paperwork. It
answers "is the car body visible", not "is the car damaged".

NOTHING IS WRITTEN TO DISK

Uploads stay in memory for the duration of the request. Keeping seller
photographs would create a personal-data store this project has no policy
for, and turning uploads into training data requires consent that nobody has
given. When that decision is made, add storage deliberately rather than
discovering that the service has been accumulating files all along.

GRACEFUL DEGRADATION

The deployed image installs requirements-web.txt, which carries Pillow but
neither imagehash nor PyTorch: imagehash drags scipy and PyWavelets along for
roughly 80 MB to buy only perceptual near-duplicates, and PyTorch costs two
gigabytes for a single axis, against a 512 MB free-instance ceiling.

Rather than failing, each capability reports itself as unavailable and the
caller shows what it could determine. A service that silently returned fewer
findings would be worse: the public image already fell back to a fixed price
range for weeks without saying so (FINDINGS 35).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

# A frame smaller than this cannot show a dent no matter what looks at it.
# Chosen from the collected corpus rather than a round number: the smallest
# frames kolesa serves are 320 pixels wide thumbnails.
MIN_USEFUL_PIXELS = 400 * 300

# Hamming distance between perceptual hashes below which two frames are the
# same picture. The photo deduplication job uses the same threshold, so the
# seller sees the same notion of "duplicate" the pipeline uses.
PHASH_NEAR_DISTANCE = 5

# Upload guard rails. These are not security boundaries; they keep a mistaken
# drag-and-drop of a photo library from occupying the process.
MAX_FILES = 20
MAX_BYTES_PER_FILE = 12 * 1024 * 1024


@dataclass
class FrameReport:
    """What could be determined about a single uploaded file."""

    name: str
    ok: bool
    bytes: int
    width: int | None = None
    height: int | None = None
    too_small: bool = False
    duplicate_of: str | None = None
    shows_bodywork: bool | None = None
    error: str | None = None


@dataclass
class IntakeReport:
    frames: list[FrameReport] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "affects_price": False,
            "frames": [vars(f) for f in self.frames],
            "unavailable": self.unavailable,
            "notes": self.notes,
        }


def _pillow():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


# Cache for the bodywork scorer. _UNSET distinguishes "not attempted yet"
# from "attempted and unavailable", so an image without the CLIP stack pays
# the import failure once instead of on every upload.
_UNSET = object()
_body_axis_cache: object = _UNSET


def _body_axis():
    """Return a scorer for "does this frame show bodywork", or None.

    Loading CLIP costs seconds and two gigabytes of dependencies, so the
    caller gets None whenever the image stack is absent and simply omits
    that finding.

    The model is built once per process. Rebuilding it per request made each
    upload wait for weights to load, which is invisible in the deployed image
    (no torch there) and painful everywhere else — exactly the kind of defect
    that survives because the environment that would reveal it is not the
    environment that runs in production.
    """
    global _body_axis_cache
    if _body_axis_cache is not _UNSET:
        return _body_axis_cache
    _body_axis_cache = _build_body_axis()
    return _body_axis_cache


def _build_body_axis():
    try:
        import numpy as np
        import torch

        from kz.ml.photo_clip import (
            NO_BODY_THRESHOLD,
            PROMPT_PAIRS,
            _load_model,
            _text_vectors,
        )
    except ImportError:
        return None

    try:
        model, preprocess, tokenizer, device = _load_model()
    except Exception:                                # noqa: BLE001 — weights,
        # network, or device trouble. An upload check must not fail because a
        # model could not be fetched; the caller reports the axis as absent.
        return None
    positive, negative = PROMPT_PAIRS["clip_no_body"]
    axis = (
        _text_vectors(model, tokenizer, device, positive)
        - _text_vectors(model, tokenizer, device, negative)
    ).cpu().numpy()
    axis = axis / np.linalg.norm(axis)

    def score(image) -> bool:
        with torch.no_grad():
            vector = model.encode_image(preprocess(image).unsqueeze(0).to(device))
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return bool(float(vector.cpu().numpy()[0] @ axis) <= NO_BODY_THRESHOLD)

    return score


def analyse(files: list[tuple[str, bytes]]) -> IntakeReport:
    """Describe uploaded frames. Never returns anything price-related.

    `files` is a list of (filename, content) pairs already read into memory
    by the caller, which is where the size limits are enforced.
    """
    report = IntakeReport()
    if not files:
        report.notes.append("No files were received.")
        return report

    Image = _pillow()
    if Image is None:
        report.unavailable.append(
            "Pixel size and image validation need Pillow, "
            "which this deployment does not install."
        )

    body_score = _body_axis() if Image is not None else None
    if body_score is None:
        report.unavailable.append(
            "Bodywork detection needs the CLIP image stack "
            "(requirements-photos.txt), which this deployment does not install."
        )

    seen_exact: dict[str, str] = {}
    hashes: list[tuple[str, object]] = []

    for name, blob in files:
        frame = FrameReport(name=name, ok=True, bytes=len(blob))

        digest = hashlib.sha256(blob).hexdigest()
        if digest in seen_exact:
            frame.duplicate_of = seen_exact[digest]
        else:
            seen_exact[digest] = name

        if Image is None:
            report.frames.append(frame)
            continue

        try:
            image = Image.open(io.BytesIO(blob))
            image.load()
            image = image.convert("RGB")
        except Exception as exc:                     # noqa: BLE001 — user file
            frame.ok = False
            frame.error = f"not a readable image ({type(exc).__name__})"
            report.frames.append(frame)
            continue

        frame.width, frame.height = image.size
        frame.too_small = frame.width * frame.height < MIN_USEFUL_PIXELS

        if frame.duplicate_of is None:
            try:
                import imagehash

                current = imagehash.phash(image)
                for other_name, other_hash in hashes:
                    if current - other_hash <= PHASH_NEAR_DISTANCE:
                        frame.duplicate_of = other_name
                        break
                else:
                    hashes.append((name, current))
            except ImportError:
                if not any("near-duplicate" in u for u in report.unavailable):
                    report.unavailable.append(
                        "Near-duplicate matching needs imagehash, which this "
                        "deployment does not install; identical files are "
                        "still detected by content hash."
                    )

        if body_score is not None:
            try:
                frame.shows_bodywork = body_score(image)
            except Exception:                        # noqa: BLE001 — best effort
                frame.shows_bodywork = None

        report.frames.append(frame)

    report.notes.extend(_summarise(report.frames))
    return report


def _summarise(frames: list[FrameReport]) -> list[str]:
    """Plain statements about the set, each checkable by looking at it."""
    notes: list[str] = []
    usable = [f for f in frames if f.ok]
    broken = [f for f in frames if not f.ok]
    if broken:
        notes.append(
            f"{len(broken)} of {len(frames)} files could not be read as images."
        )

    duplicates = [f for f in usable if f.duplicate_of]
    if duplicates:
        notes.append(
            f"{len(duplicates)} frames repeat an earlier photo. "
            "Buyers see fewer distinct views than the count suggests."
        )

    small = [f for f in usable if f.too_small]
    if small:
        notes.append(
            f"{len(small)} frames are smaller than {MIN_USEFUL_PIXELS // 1000}k "
            "pixels, too small to show panel damage."
        )

    scored = [f for f in usable if f.shows_bodywork is not None]
    if scored:
        without = [f for f in scored if not f.shows_bodywork]
        if without:
            notes.append(
                f"{len(without)} of {len(scored)} frames do not show the "
                "vehicle body — cabin, engine bay, wheels, or documents. "
                "Buyers look for exterior views first."
            )
        distinct_body = [
            f for f in scored if f.shows_bodywork and not f.duplicate_of
        ]
        if len(distinct_body) < 3:
            notes.append(
                f"Only {len(distinct_body)} distinct exterior frames. "
                "Listings normally show several angles of the body."
            )

    notes.append(
        "These observations do not change the estimate. This project has no "
        "validated model that reads condition from photographs; see "
        "docs/FINDINGS.md sections 23 and 28."
    )
    return notes
