# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.photo_clip` module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_PATH = Path("data/models/photo_clip.npz")
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH = 32


PROMPT_PAIRS = {
    "clip_damaged": (
        [
            "a photo of a damaged car",
            "a photo of a wrecked car",
            "a car after an accident",
            "a crashed car with body damage",
        ],
        [
            "a photo of a car in good condition",
            "a clean undamaged car",
            "a well maintained car",
            "a car in excellent condition",
        ],
    ),
    "clip_rusty": (
        ["a rusty old car", "a car with rust and corrosion"],
        ["a car with clean paint", "a car with shiny bodywork"],
    ),
    "clip_dirty": (
        ["a dirty muddy car", "an unwashed car"],
        ["a freshly washed clean car", "a polished car"],
    ),
    "clip_no_body": (
        [
            "a photo of the interior of a car",
            "a car dashboard and steering wheel",
            "car seats inside the cabin",
            "a close-up of a car engine bay",
            "a close-up photo of a car wheel",
            "a photo of vehicle documents",
        ],
        [
            "a photo of a car exterior parked outside",
            "the side of a car body",
            "a car seen from the front outside",
            "a whole car photographed on the street",
        ],
    ),
    "clip_studio": (
        ["a professional dealership photo of a car in a showroom", "a studio photograph of a car"],
        ["an amateur phone photo of a car on the street", "a car parked in a yard"],
    ),
}


def _load_model():
    import open_clip
    import torch

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


def _text_vectors(model, tokenizer, device, prompts: list[str]):
    """Implement `_text_vectors`."""
    import torch

    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        v = t.mean(dim=0)
        return v / v.norm()


def score_photos(paths: list[str], log=print, keep_embeddings: bool = False):
    """Implement `score_photos`."""
    import torch
    from PIL import Image, ImageOps

    model, preprocess, tokenizer, device = _load_model()
    log(f"  device: {device}, images: {len(paths)}")

    axes = {
        name: (
            _text_vectors(model, tokenizer, device, pos),
            _text_vectors(model, tokenizer, device, neg),
        )
        for name, (pos, neg) in PROMPT_PAIRS.items()
    }

    rows = []
    embeddings = [] if keep_embeddings else None
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i : i + BATCH]
            batch = torch.stack(
                [preprocess(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in chunk]
            ).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out = {}
            for name, (pos, neg) in axes.items():
                out[name] = (feats @ pos - feats @ neg).cpu().numpy()
            if keep_embeddings:
                embeddings.append(feats.cpu().numpy().astype(np.float32))
            for j in range(len(chunk)):
                rows.append({name: float(out[name][j]) for name in axes})
            if (i // BATCH) % 10 == 0:
                log(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    scores = pd.DataFrame(rows)
    if keep_embeddings:
        return scores, np.vstack(embeddings)
    return scores


def build(log=print, all_positions: bool = True) -> pd.DataFrame:
    from kz.ml.photo_features import photo_index

    idx = photo_index(all_positions=all_positions)
    if idx.empty:
        raise SystemExit("No photos found. Run: python -m kz.collect.photo_fetch")
    pos = idx["position"].to_numpy() if all_positions else np.ones(len(idx), int)
    log(f"Photos: {len(idx)} across {idx['ad_id'].nunique()} listings")
    scores, emb = score_photos(idx["path"].tolist(), log=log, keep_embeddings=True)
    scores.insert(0, "position", pos)
    scores.insert(0, "ad_id", idx["ad_id"].to_numpy())
    CLIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    value_cols = [c for c in scores.columns if c not in ("ad_id", "position")]
    np.savez_compressed(
        CLIP_PATH,
        ad_id=scores["ad_id"].to_numpy().astype("U32"),
        position=scores["position"].to_numpy().astype(np.int16),
        scores=scores[value_cols].to_numpy().astype(np.float32),
        cols=np.array(value_cols, dtype="U32"),
        path=idx["path"].to_numpy().astype("U160"),
        emb=emb,
    )
    log(f"Saved → {CLIP_PATH}  (scores {scores[value_cols].shape}, embeddings {emb.shape})")
    return scores


def load_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    """Implement `load_embeddings`."""
    if not CLIP_PATH.exists():
        raise FileNotFoundError("Run first: python -m kz.ml.photo_clip")
    z = np.load(CLIP_PATH, allow_pickle=False)
    if "emb" not in z.files:
        raise KeyError("The artifact has no embeddings; recompute with python -m kz.ml.photo_clip")
    idx = pd.DataFrame(
        {
            "ad_id": [str(a) for a in z["ad_id"]],
            "position": z["position"],
            "path": [str(p) for p in z["path"]],
        }
    )
    return idx, z["emb"]


def load() -> pd.DataFrame:
    if not CLIP_PATH.exists():
        raise FileNotFoundError("Run first: python -m kz.ml.photo_clip")
    z = np.load(CLIP_PATH, allow_pickle=False)
    df = pd.DataFrame(z["scores"], columns=[str(c) for c in z["cols"]])
    df.insert(0, "ad_id", [str(a) for a in z["ad_id"]])

    df.insert(1, "position", z["position"] if "position" in z.files else 1)
    return df


def aggregate(per_photo: pd.DataFrame) -> pd.DataFrame:
    """Implement `aggregate`."""
    cols = [c for c in per_photo.columns if c not in ("ad_id", "position")]
    g = per_photo.groupby("ad_id")
    out = g[cols].max().add_suffix("_max").join(g[cols].mean().add_suffix("_mean"))
    cover = (
        per_photo.sort_values("position")
        .drop_duplicates("ad_id")
        .set_index("ad_id")[cols]
        .add_suffix("_cover")
    )
    out = out.join(cover)
    out["n_photos"] = g.size()
    return out.reset_index()


def validate(log=print) -> None:
    """Implement `validate`."""
    from sklearn.metrics import roc_auc_score

    from kz.core.db import get_engine

    per_photo = load()
    d = aggregate(per_photo)
    cd = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge, price_tenge, age FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    d = d.merge(cd, on="ad_id", how="left")
    d["has_damage"] = d.damage_keywords.fillna("").str.len() > 0
    d["bad_badge"] = d.page_status_badge.fillna("-").str.contains("вар|ход|залож", case=False)

    log(
        f"Scored listings: {len(d)}   frames: {len(per_photo)}   "
        f"average {len(per_photo) / max(len(d), 1):.1f} per listing\n"
    )

    bases = sorted({c for c in per_photo.columns if c not in ("ad_id", "position")})
    for flag, name in [
        ("has_damage", "damage terms in seller text"),
        ("bad_badge", "emergency/non-running badge"),
    ]:
        y = d[flag].to_numpy()
        if y.sum() < 5 or (~y).sum() < 5:
            log(f"{name}: {int(y.sum())} examples; too few for AUC\n")
            continue
        log(f"{name}: {int(y.sum())} versus {int((~y).sum())}   (AUC, 0.5 = chance)")
        log(f"   {'axis':14} {'cover':>9} {'maximum':>9} {'mean':>9}")
        for base in bases:
            cells = []
            for suffix in ("_cover", "_max", "_mean"):
                col = base + suffix
                cells.append(f"{roc_auc_score(y, d[col]):9.3f}" if col in d else f"{'—':>9}")
            log(f"   {base:14} " + " ".join(cells))
        log("")

    log("Frames 2–5 add value only when maximum and mean scores outperform")
    log("the cover score. A cover is the sales-friendly angle; signal found")
    log("only there means the damage was already visible in the first photo.\n")
    _redundancy_check(d, log=log)


def _oof_logistic_auc(X, y) -> float:
    """Implement `_oof_logistic_auc`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = min(5, int(y.sum()), int((1 - y).sum()))
    if n < 2:
        raise ValueError("OOF AUC requires at least two examples of each class")
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pred = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pred))


def _redundancy_check(d: pd.DataFrame, log=print) -> None:
    """Implement `_redundancy_check`."""
    d = d.copy()
    d["log_price"] = np.log(pd.to_numeric(d["price_tenge"], errors="coerce"))
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    probes = [c for c in ("clip_damaged_max", "clip_rusty_mean", "clip_dirty_mean") if c in d]

    log("Does CLIP add signal BEYOND vehicle age and price?")
    for flag, name in [("has_damage", "damage terms"), ("bad_badge", "emergency badge")]:
        work = d[["age", "log_price", flag] + probes].dropna()
        y = work[flag].to_numpy().astype(int)
        if y.sum() < 5:
            log(f"  {name}: {int(y.sum())} examples; too few")
            continue

        auc0 = _oof_logistic_auc(work[["age", "log_price"]], y)
        log(f"  {name} ({int(y.sum())} positives): age + price produce OOF AUC {auc0:.3f}")
        for c in probes:
            auc = _oof_logistic_auc(work[["age", "log_price", c]], y)
            log(f"     + {c:20} {auc:.3f}  ({auc - auc0:+.3f})")


def main():
    if "--no-body" in sys.argv:
        build_no_body()
        return
    if "--rank" in sys.argv:
        build_damage_rank()
        return
    if "--validate" in sys.argv:
        validate()
        return
    build()
    validate()


NO_BODY_CSV = "data/models/photo_no_body.csv"


#

#   +0,05        81%                2               0
#   +0,03        85%               15               0
#   +0,02        91%               24               1
#


NO_BODY_THRESHOLD = 0.03


def score_axis(name: str = "clip_no_body") -> pd.DataFrame:
    """Implement `score_axis`."""
    idx, emb = load_embeddings()
    model, _, tokenizer, device = _load_model()
    pos, neg = PROMPT_PAIRS[name]
    v = (
        (
            _text_vectors(model, tokenizer, device, pos)
            - _text_vectors(model, tokenizer, device, neg)
        )
        .cpu()
        .numpy()
    )
    out = idx[["ad_id", "position"]].copy()
    out[name] = emb @ (v / np.linalg.norm(v))
    return out


def build_no_body(log=print) -> pd.DataFrame:
    """Implement `build_no_body`."""
    d = score_axis("clip_no_body")
    Path(NO_BODY_CSV).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(NO_BODY_CSV, index=False)
    n = int((d.clip_no_body > NO_BODY_THRESHOLD).sum())
    log(
        f"'Body missing' axis: {n} of {len(d)} frames above threshold "
        f"{NO_BODY_THRESHOLD} ({n / len(d) * 100:.0f}%) → {NO_BODY_CSV}"
    )
    return d


def load_no_body() -> pd.DataFrame | None:
    """Implement `load_no_body`."""
    p = Path(NO_BODY_CSV)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"ad_id": str})


RANK_CSV = "data/models/photo_damage_rank.csv"


def build_damage_rank(log=print) -> pd.DataFrame:
    """Implement `build_damage_rank`."""
    from sklearn.linear_model import LogisticRegression

    from kz.report.photo_labels import LABELS_CSV, read_journal

    _, rows = read_journal()
    lab = pd.DataFrame(rows)
    if lab.empty:
        raise RuntimeError(f"Label journal is empty: {LABELS_CSV}")
    lab = lab.drop_duplicates(["ad_id", "position"], keep="last")
    lab["position"] = lab.position.astype(int)
    if "dataset_split" in lab:
        split = lab.dataset_split.fillna("").replace("", "train")
        lab = lab[split != "audit"]

    idx, emb = load_embeddings()
    idx = idx.reset_index(drop=True)
    idx["row"] = idx.index
    idx["position"] = idx.position.astype(int)
    d = lab.merge(idx[["ad_id", "position", "row"]], on=["ad_id", "position"])
    if "review_status" in d:
        from kz.report.photo_labels import NEEDS_REVIEW

        d = d[d.review_status.fillna("") != NEEDS_REVIEW]
    d = d[d.label.isin(["damaged", "wreck", "intact"])]
    y = d.label.isin(["damaged", "wreck"]).astype(int).to_numpy()
    if y.sum() < 5:
        raise RuntimeError(f"Only {y.sum()} positive examples; ranking is premature")

    model = LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced")
    model.fit(emb[d.row.to_numpy()], y)
    out = idx[["ad_id", "position"]].copy()
    out["damage_rank"] = model.predict_proba(emb)[:, 1]
    Path(RANK_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RANK_CSV, index=False)
    log(
        f"Ranking model: trained on {len(d)} labels ({y.sum()} positives), "
        f"scored {len(out)} frames → {RANK_CSV}"
    )
    return out


def load_damage_rank() -> pd.DataFrame | None:
    p = Path(RANK_CSV)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"ad_id": str})


if __name__ == "__main__":
    main()
