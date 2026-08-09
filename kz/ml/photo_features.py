# -*- coding: utf-8 -*-
"""Признаки из фотографий: эмбеддинги и простые метрики качества снимка.

ЗАЧЕМ. Ошибка модели цены сосредоточена там, где цену определяет состояние
машины: на дешёвых (31% MAPE) и старых (32%). В табличных признаках состояния
нет и взяться ему неоткуда — год и пробег не отличают ухоженную машину от
убитой. Единственное место, где состояние видно, — фотография.

ДВА ВИДА ПРИЗНАКОВ:

  эмбеддинг    выход предпоследнего слоя ResNet50, обученной на ImageNet.
               Это вектор из 2048 чисел, описывающий содержимое картинки.
               Размечать ничего не нужно: если в снимке есть информация о
               состоянии, она в векторе окажется, и модель её найдёт сама.

  качество     резкость, яркость, контраст, разрешение. Считаются напрямую,
               без нейросети. Отсюда рекомендации продавцу: «фото смазано»,
               «снимок слишком тёмный».

ПРО РАЗМЕРНОСТЬ. 2048 признаков на ~2700 машин — верный способ переобучиться:
признаков почти столько же, сколько наблюдений. Поэтому эмбеддинг сжимается
методом главных компонент (PCA) до N_COMPONENTS. PCA не смотрит на цену,
только на сами картинки, поэтому утечки цели здесь нет.

Запуск: python -m kz.ml.photo_features        посчитать и сохранить
        python -m kz.ml.photo_features --stats  что уже посчитано
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMB_PATH = Path("data/models/photo_embeddings.npz")
BATCH = 32
N_COMPONENTS = 48          # во сколько чисел сжимаем эмбеддинг
IMAGE_SIZE = 224           # вход ResNet


def photo_index(all_positions: bool = False) -> pd.DataFrame:
    """Какие фотографии лежат на диске: ad_id → путь.

    По умолчанию одна обложка на объявление — так задумано для признаков
    модели цены: пять кадров одной машины дали бы пять строк на один
    объект, и любая агрегация должна быть осознанной, а не побочным
    эффектом чтения индекса.

    all_positions=True возвращает ВСЕ кадры вместе с позицией. Это нужно
    для оценки состояния: обложка — парадный кадр, продавец ставит туда
    лучший ракурс, а повреждения, если их вообще показывают, попадают на
    снимки 2-5.
    """
    from kz.collect.photo_fetch import MANIFEST

    cols = ["ad_id", "position", "path"] if all_positions else ["ad_id", "path"]
    if not MANIFEST.exists():
        return pd.DataFrame(columns=cols)
    man = pd.read_csv(MANIFEST, dtype={"ad_id": str})
    ok = man[(man["http_status"] == 200) & man["path"].notna()].copy()
    ok = ok[ok["path"].map(lambda p: Path(str(p)).exists())]
    ok = ok.sort_values(["ad_id", "position"])
    if not all_positions:
        ok = ok.drop_duplicates("ad_id")     # минимальная позиция = обложка
    return ok[cols]


def quality_metrics(path: str) -> dict:
    """Метрики качества снимка без нейросети.

    Резкость — дисперсия лапласиана: у смазанного фото мало резких перепадов
    яркости, поэтому дисперсия мала. Классический приём, работает без
    обучения и объясним человеку.
    """
    from PIL import Image, ImageFilter, ImageStat

    img = Image.open(path).convert("L")
    lap = img.filter(ImageFilter.FIND_EDGES)
    st_lap, st_img = ImageStat.Stat(lap), ImageStat.Stat(img)
    return {
        "img_sharpness": float(st_lap.stddev[0] ** 2),
        "img_brightness": float(st_img.mean[0]),
        "img_contrast": float(st_img.stddev[0]),
        "img_pixels": int(img.width * img.height),
    }


def _model_and_transform():
    """ResNet50 без последнего слоя: на выходе вектор, а не класс ImageNet."""
    import torch
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()          # убираем классификатор
    net.eval()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return net, tf, device


def embed_paths(paths: list[str], log=print) -> np.ndarray:
    """Векторы для списка картинок, батчами."""
    import torch
    from PIL import Image

    net, tf, device = _model_and_transform()
    log(f"  устройство: {device}, картинок: {len(paths)}")
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk])
            out.append(net(batch.to(device)).cpu().numpy())
            if (i // BATCH) % 10 == 0:
                log(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    return np.vstack(out)


def build(save: bool = True, log=print) -> tuple[pd.DataFrame, np.ndarray]:
    """Посчитать признаки для скачанных обложек.

    Инкрементально: уже посчитанные объявления пропускаются. Прогон сети по
    трём тысячам картинок занимает минуты, и повторять его на каждом запуске
    конвейера ради десятка новых фото — бессмысленная трата.
    """
    idx = photo_index()
    if idx.empty:
        raise SystemExit("Нет скачанных фотографий: python -m kz.collect.photo_fetch")

    old_q, old_emb = None, None
    if EMB_PATH.exists():
        try:
            old_q, old_emb = load()
            known = set(old_q["ad_id"])
            idx = idx[~idx["ad_id"].isin(known)]
            log(f"Уже посчитано: {len(known)}; новых: {len(idx)}")
        except Exception as e:                    # noqa: BLE001 — битый кэш не фатален
            log(f"Кэш признаков не прочитался ({e}), считаю заново.")
            old_q, old_emb = None, None
    if idx.empty:
        log("Новых фотографий нет — признаки актуальны.")
        return old_q, old_emb

    log(f"Считаю для {len(idx)} фотографий")
    log("Метрики качества…")
    q = pd.DataFrame([quality_metrics(p) for p in idx["path"]])
    q.insert(0, "ad_id", idx["ad_id"].to_numpy())

    log("Эмбеддинги ResNet50…")
    emb = embed_paths(idx["path"].tolist(), log=log)

    if old_q is not None:
        q = pd.concat([old_q, q], ignore_index=True)
        emb = np.vstack([old_emb, emb])
        idx = pd.DataFrame({"ad_id": q["ad_id"]})

    if save:
        EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Строки сохраняем с явным юникод-типом, а не как объекты: объектный
        # массив читается только через pickle, а включать pickle ради списка
        # идентификаторов — значит разрешить выполнение кода при чтении файла.
        np.savez_compressed(
            EMB_PATH,
            ad_id=idx["ad_id"].to_numpy().astype("U32"),
            emb=emb.astype(np.float32),
            quality=q.drop(columns="ad_id").to_numpy().astype(np.float64),
            quality_cols=np.array([c for c in q.columns if c != "ad_id"],
                                  dtype="U32"))
        log(f"Сохранено → {EMB_PATH} ({EMB_PATH.stat().st_size/1e6:.1f} МБ)")
    return q, emb


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """Сохранённые признаки: таблица качества и матрица эмбеддингов."""
    if not EMB_PATH.exists():
        raise FileNotFoundError(
            "Нет признаков из фото. Сначала: python -m kz.ml.photo_features")
    z = np.load(EMB_PATH, allow_pickle=False)
    q = pd.DataFrame(z["quality"], columns=[str(c) for c in z["quality_cols"]])
    q.insert(0, "ad_id", [str(a) for a in z["ad_id"]])
    return q, z["emb"]


def reduce_embeddings(emb: np.ndarray, n_components: int = N_COMPONENTS,
                      seed: int = 42) -> np.ndarray:
    """Сжать эмбеддинг до n_components главных компонент.

    Без сжатия признаков было бы почти столько же, сколько машин, и модель
    выучила бы шум. PCA работает только с картинками и не видит цену,
    поэтому утечки цели не создаёт.
    """
    from sklearn.decomposition import PCA

    n = min(n_components, emb.shape[0], emb.shape[1])
    return PCA(n_components=n, random_state=seed).fit_transform(emb)


def main():
    if "--stats" in sys.argv:
        idx = photo_index()
        print(f"Скачано обложек: {len(idx)}")
        if EMB_PATH.exists():
            q, emb = load()
            print(f"Признаки посчитаны для {len(q)}, размер эмбеддинга {emb.shape[1]}")
            print(q.describe().round(1).to_string())
        else:
            print("Признаки ещё не считались.")
        return
    build()


if __name__ == "__main__":
    main()
