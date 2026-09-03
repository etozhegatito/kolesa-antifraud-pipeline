# -*- coding: utf-8 -*-
"""Implementation for the `kz.ml.photo_features` module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMB_PATH = Path("data/models/photo_embeddings.npz")
BATCH = 32
N_COMPONENTS = 48
IMAGE_SIZE = 224


def photo_index(all_positions: bool = False) -> pd.DataFrame:
    """Implement `photo_index`."""
    from kz.collect.photo_fetch import MANIFEST

    cols = ["ad_id", "position", "path"] if all_positions else ["ad_id", "path"]
    if not MANIFEST.exists():
        return pd.DataFrame(columns=cols)
    man = pd.read_csv(MANIFEST, dtype={"ad_id": str})
    ok = man[(man["http_status"] == 200) & man["path"].notna()].copy()
    ok = ok[ok["path"].map(lambda p: Path(str(p)).exists())]
    ok = ok.sort_values(["ad_id", "position"])
    if not all_positions:
        ok = ok.drop_duplicates("ad_id")
    return ok[cols]


def quality_metrics(path: str) -> dict:
    """Implement `quality_metrics`."""
    from PIL import Image, ImageFilter, ImageOps, ImageStat

    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    lap = img.filter(ImageFilter.FIND_EDGES)
    st_lap, st_img = ImageStat.Stat(lap), ImageStat.Stat(img)
    return {
        "img_sharpness": float(st_lap.stddev[0] ** 2),
        "img_brightness": float(st_img.mean[0]),
        "img_contrast": float(st_img.stddev[0]),
        "img_pixels": int(img.width * img.height),
    }


def _model_and_transform():
    """Implement `_model_and_transform`."""
    import torch
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    net.to(device)
    tf = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return net, tf, device


def embed_paths(paths: list[str], log=print) -> np.ndarray:
    """Implement `embed_paths`."""
    import torch
    from PIL import Image, ImageOps

    net, tf, device = _model_and_transform()
    log(f"  device: {device}, images: {len(paths)}")
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i : i + BATCH]
            batch = torch.stack(
                [tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in chunk]
            )
            out.append(net(batch.to(device)).cpu().numpy())
            if (i // BATCH) % 10 == 0:
                log(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    return np.vstack(out)


def build(save: bool = True, log=print) -> tuple[pd.DataFrame, np.ndarray]:
    """Implement `build`."""
    idx = photo_index()
    if idx.empty:
        raise SystemExit("No downloaded photos. Run: python -m kz.collect.photo_fetch")

    old_q, old_emb = None, None
    if EMB_PATH.exists():
        try:
            old_q, old_emb = load()
            known = set(old_q["ad_id"])
            idx = idx[~idx["ad_id"].isin(known)]
            log(f"Already computed: {len(known)}; new: {len(idx)}")
        except Exception as e:  # noqa: BLE001 -- intentional exception
            log(f"Feature cache could not be read ({e}); recomputing.")
            old_q, old_emb = None, None
    if idx.empty:
        log("No new photos; features are current.")
        return old_q, old_emb

    log(f"Computing features for {len(idx)} photos")
    log("Quality metrics...")
    q = pd.DataFrame([quality_metrics(p) for p in idx["path"]])
    q.insert(0, "ad_id", idx["ad_id"].to_numpy())

    log("ResNet50 embeddings...")
    emb = embed_paths(idx["path"].tolist(), log=log)

    if old_q is not None:
        q = pd.concat([old_q, q], ignore_index=True)
        emb = np.vstack([old_emb, emb])
        idx = pd.DataFrame({"ad_id": q["ad_id"]})

    if save:
        EMB_PATH.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            EMB_PATH,
            ad_id=idx["ad_id"].to_numpy().astype("U32"),
            emb=emb.astype(np.float32),
            quality=q.drop(columns="ad_id").to_numpy().astype(np.float64),
            quality_cols=np.array([c for c in q.columns if c != "ad_id"], dtype="U32"),
        )
        log(f"Saved → {EMB_PATH} ({EMB_PATH.stat().st_size / 1e6:.1f} MB)")
    return q, emb


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """Implement `load`."""
    if not EMB_PATH.exists():
        raise FileNotFoundError("Photo features are missing. Run: python -m kz.ml.photo_features")
    z = np.load(EMB_PATH, allow_pickle=False)
    q = pd.DataFrame(z["quality"], columns=[str(c) for c in z["quality_cols"]])
    q.insert(0, "ad_id", [str(a) for a in z["ad_id"]])
    return q, z["emb"]


def load_quality() -> pd.DataFrame:
    """Implement `load_quality`."""
    q, _ = load()
    cols = [c for c in q.columns if c != "ad_id"]
    return q.groupby("ad_id", as_index=False)[cols].mean()


def reduce_embeddings(
    emb: np.ndarray, n_components: int = N_COMPONENTS, seed: int = 42
) -> np.ndarray:
    """Implement `reduce_embeddings`."""
    from sklearn.decomposition import PCA

    n = min(n_components, emb.shape[0], emb.shape[1])
    return PCA(n_components=n, random_state=seed).fit_transform(emb)


def main():
    if "--stats" in sys.argv:
        idx = photo_index()
        print(f"Downloaded cover photos: {len(idx)}")
        if EMB_PATH.exists():
            q, emb = load()
            print(f"Features computed for {len(q)} images; embedding size {emb.shape[1]}")
            print(q.describe().round(1).to_string())
        else:
            print("Photo features have not been computed yet.")
        return
    build()


if __name__ == "__main__":
    main()
