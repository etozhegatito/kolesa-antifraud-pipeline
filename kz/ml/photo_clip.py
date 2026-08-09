# -*- coding: utf-8 -*-
"""Оценка состояния машины по фотографии без разметки (CLIP, zero-shot).

ЗАЧЕМ ЭТО ПОСЛЕ НЕУДАЧИ С RESNET. Эмбеддинг ResNet50 цене не помог, и
диагностика показала почему: сеть уверенно различает кузов (60,7% против
35,6% у случайного угадывания), но кузов у нас и так есть в признаках. Её
учили отличать объекты друг от друга, а не оценивать состояние.

CLIP обучен иначе: он связывает изображения с текстовыми описаниями. Поэтому
у него можно спросить напрямую, без единого размеченного примера:

    насколько это фото похоже на «фотография битой машины»
                        против «фотография машины в хорошем состоянии»

Приём называется zero-shot: модель применяется к задаче, которой её
специально не учили, а «обучающей выборкой» служат сами текстовые подписи.

ПОЧЕМУ ПОДПИСИ ПО-АНГЛИЙСКИ. CLIP обучался преимущественно на англоязычных
парах «картинка — подпись». Русские подписи он понимает заметно хуже, и это
исказило бы результат не в пользу метода.

Запуск: python -m kz.ml.photo_clip            посчитать и сохранить
        python -m kz.ml.photo_clip --validate проверить на известных случаях
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_PATH = Path("data/models/photo_clip.npz")
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH = 32

# Пары противоположных описаний. Оценка считается как разница близостей
# внутри пары, поэтому общий стиль формулировки взаимно сокращается и
# остаётся именно то, чем описания различаются.
PROMPT_PAIRS = {
    "clip_damaged": (
        ["a photo of a damaged car", "a photo of a wrecked car",
         "a car after an accident", "a crashed car with body damage"],
        ["a photo of a car in good condition", "a clean undamaged car",
         "a well maintained car", "a car in excellent condition"],
    ),
    "clip_rusty": (
        ["a rusty old car", "a car with rust and corrosion"],
        ["a car with clean paint", "a car with shiny bodywork"],
    ),
    "clip_dirty": (
        ["a dirty muddy car", "an unwashed car"],
        ["a freshly washed clean car", "a polished car"],
    ),
    "clip_studio": (
        ["a professional dealership photo of a car in a showroom",
         "a studio photograph of a car"],
        ["an amateur phone photo of a car on the street",
         "a car parked in a yard"],
    ),
}


def _load_model():
    import open_clip
    import torch

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


def _text_vectors(model, tokenizer, device, prompts: list[str]):
    """Средний нормированный вектор набора описаний.

    Усреднение по нескольким формулировкам устойчивее одной: конкретное
    слово может случайно попасть в неудачную область, а среднее по четырём
    описаниям одного смысла — уже нет.
    """
    import torch

    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        v = t.mean(dim=0)
        return v / v.norm()


def score_photos(paths: list[str], log=print) -> pd.DataFrame:
    """Для каждой картинки — оценка по каждой паре описаний."""
    import torch
    from PIL import Image

    model, preprocess, tokenizer, device = _load_model()
    log(f"  устройство: {device}, картинок: {len(paths)}")

    axes = {name: (_text_vectors(model, tokenizer, device, pos),
                   _text_vectors(model, tokenizer, device, neg))
            for name, (pos, neg) in PROMPT_PAIRS.items()}

    rows = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                 for p in chunk]).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out = {}
            for name, (pos, neg) in axes.items():
                # Разница косинусных близостей: >0 — ближе к «положительному»
                # описанию пары, <0 — к противоположному.
                out[name] = (feats @ pos - feats @ neg).cpu().numpy()
            for j in range(len(chunk)):
                rows.append({name: float(out[name][j]) for name in axes})
            if (i // BATCH) % 10 == 0:
                log(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    return pd.DataFrame(rows)


def build(log=print) -> pd.DataFrame:
    from kz.ml.photo_features import photo_index

    idx = photo_index()
    if idx.empty:
        raise SystemExit("Нет фотографий: python -m kz.collect.photo_fetch")
    log(f"Фотографий: {len(idx)}")
    scores = score_photos(idx["path"].tolist(), log=log)
    scores.insert(0, "ad_id", idx["ad_id"].to_numpy())
    CLIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CLIP_PATH,
        ad_id=scores["ad_id"].to_numpy().astype("U32"),
        scores=scores.drop(columns="ad_id").to_numpy().astype(np.float32),
        cols=np.array([c for c in scores.columns if c != "ad_id"], dtype="U32"))
    log(f"Сохранено → {CLIP_PATH}")
    return scores


def load() -> pd.DataFrame:
    if not CLIP_PATH.exists():
        raise FileNotFoundError("Сначала: python -m kz.ml.photo_clip")
    z = np.load(CLIP_PATH, allow_pickle=False)
    df = pd.DataFrame(z["scores"], columns=[str(c) for c in z["cols"]])
    df.insert(0, "ad_id", [str(a) for a in z["ad_id"]])
    return df


def validate(log=print) -> None:
    """Проверка на объявлениях, где состояние известно из другого источника.

    Разметки мало, поэтому сравниваем не точность, а распределения: если
    оценка осмысленна, у машин с damage-словами в тексте она должна быть
    систематически выше.
    """
    from kz.core.db import get_engine

    scores = load()
    cd = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge, price_tenge, age "
        "FROM clean_data", get_engine(), dtype={"ad_id": str})
    d = scores.merge(cd, on="ad_id", how="left")
    d["has_damage"] = d.damage_keywords.fillna("").str.len() > 0
    d["bad_badge"] = d.page_status_badge.fillna("-").str.contains(
        "вар|ход|залож", case=False)

    log(f"Машин с оценкой: {len(d)}\n")
    for flag, name in [("has_damage", "damage-слова в тексте"),
                       ("bad_badge", "бейдж «Аварийная»")]:
        a, b = d[d[flag]], d[~d[flag]]
        if len(a) < 3:
            log(f"{name}: примеров {len(a)} — слишком мало")
            continue
        log(f"{name}: {len(a)} против {len(b)}")
        for col in [c for c in scores.columns if c != "ad_id"]:
            log(f"   {col:14} с признаком {a[col].mean():+.4f}   "
                f"без {b[col].mean():+.4f}   разница {a[col].mean()-b[col].mean():+.4f}")
        log("")


def main():
    if "--validate" in sys.argv:
        validate()
        return
    build()
    validate()


if __name__ == "__main__":
    main()
