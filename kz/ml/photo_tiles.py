# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.photo_tiles` module."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


GRID = 3


OVERLAP = 0.25

BATCH = 64
MAX_PHOTOS = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0


def tile_boxes(
    w: int, h: int, grid: int = GRID, overlap: float = OVERLAP
) -> list[tuple[int, int, int, int]]:
    """Implement `tile_boxes`."""
    tw, th = w / grid, h / grid
    dx, dy = tw * overlap, th * overlap
    boxes = []
    for r in range(grid):
        for c in range(grid):
            x1 = max(0, int(c * tw - dx))
            y1 = max(0, int(r * th - dy))
            x2 = min(w, int((c + 1) * tw + dx))
            y2 = min(h, int((r + 1) * th + dy))
            boxes.append((x1, y1, x2, y2))
    return boxes


def score_tiles(paths: list[str], log=print) -> pd.DataFrame:
    """Implement `score_tiles`."""
    import torch
    from PIL import Image, ImageOps

    from kz.ml.photo_clip import PROMPT_PAIRS, _load_model, _text_vectors

    model, preprocess, tokenizer, device = _load_model()
    axes = {
        name: (
            _text_vectors(model, tokenizer, device, pos),
            _text_vectors(model, tokenizer, device, neg),
        )
        for name, (pos, neg) in PROMPT_PAIRS.items()
    }
    log(f"  device: {device}, frames: {len(paths)}, tiles per frame: {GRID * GRID}")

    rows = []
    with torch.no_grad():
        for n, path in enumerate(paths):
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            crops = [img.crop(b) for b in tile_boxes(*img.size)]
            per_tile = {name: [] for name in axes}
            for i in range(0, len(crops), BATCH):
                batch = torch.stack([preprocess(c) for c in crops[i : i + BATCH]]).to(device)
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                for name, (pos, neg) in axes.items():
                    per_tile[name].append((feats @ pos - feats @ neg).cpu().numpy())
            row = {}
            for name in axes:
                v = np.concatenate(per_tile[name])
                row[f"{name}_tilemax"] = float(v.max())
                row[f"{name}_tilemean"] = float(v.mean())
            rows.append(row)
            if n % 200 == 0:
                log(f"  {n}/{len(paths)}")
    return pd.DataFrame(rows)


def main():
    from sklearn.metrics import roc_auc_score

    from kz.core.db import get_engine
    from kz.ml.photo_clip import load as load_scores
    from kz.ml.photo_clip import load_embeddings

    idx, _ = load_embeddings()
    whole = load_scores()

    cd_all = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    cd_all["pos"] = (
        cd_all.damage_keywords.fillna("").str.len() > 0
    ) | cd_all.page_status_badge.fillna("-").str.contains("вар|ход|залож", case=False)
    pos_ads = set(cd_all.loc[cd_all.pos, "ad_id"])

    is_pos = idx.ad_id.isin(pos_ads)
    n_ctrl = MAX_PHOTOS or 600
    ctrl = idx[~is_pos].sample(n=min(n_ctrl, int((~is_pos).sum())), random_state=42)
    keep = pd.concat([idx[is_pos], ctrl]).sort_index()
    whole = whole.loc[keep.index]
    idx = keep

    print(f"Frames: {len(idx)}   from damage-positive listings: {int(is_pos.sum())}")
    print(f"Grid {GRID}×{GRID} with {int(OVERLAP * 100)}% overlap\n")
    tiles = score_tiles(idx["path"].tolist())
    d = pd.concat(
        [
            idx.reset_index(drop=True),
            whole.reset_index(drop=True).drop(columns=["ad_id", "position"]),
            tiles,
        ],
        axis=1,
    )

    value_cols = [c for c in d.columns if c.startswith("clip_")]
    per_ad = d.groupby("ad_id")[value_cols].max().reset_index()

    cd = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge FROM clean_data",
        get_engine(),
        dtype={"ad_id": str},
    )
    m = per_ad.merge(cd, on="ad_id", how="left")
    m["has_damage"] = m.damage_keywords.fillna("").str.len() > 0
    m["bad_badge"] = m.page_status_badge.fillna("-").str.contains("вар|ход|залож", case=False)

    for flag, name in [("has_damage", "damage terms"), ("bad_badge", "status badge")]:
        y = m[flag].to_numpy()
        if y.sum() < 5 or (~y).sum() < 5:
            print(f"\n{name}: {int(y.sum())} examples; too few")
            continue
        print(f"\n{name}: {int(y.sum())} versus {int((~y).sum())}   (AUC)")
        print(f"   {'axis':14} {'whole frame':>11} {'tile max':>12} {'tile mean':>12}")
        bases = sorted({c.split("_tile")[0] for c in value_cols if "_tile" in c})
        for base in bases:
            cells = []
            for col in (base, f"{base}_tilemax", f"{base}_tilemean"):
                cells.append(f"{roc_auc_score(y, m[col]):11.3f}" if col in m else f"{'—':>11}")
            print(f"   {base:14} " + " ".join(cells))

    print("\nIf tile max is clearly above whole-frame performance, damage is local")
    print("and bounding-box annotation is justified. Otherwise scale is not the")
    print("problem, and a crop-labeling interface would not add value.")


if __name__ == "__main__":
    main()
