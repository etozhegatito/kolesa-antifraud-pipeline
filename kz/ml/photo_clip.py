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


def score_photos(paths: list[str], log=print,
                 keep_embeddings: bool = False):
    """Оценки по парам описаний и, по запросу, сами векторы CLIP.

    Векторы нужны для дообучения на СВОИХ метках. Zero-shot упёрся ровно
    там, где интереснее всего: `clip_damaged` неотличим от монетки, потому
    что описания «битая машина» слишком общие для наших снимков. Логистическая
    регрессия поверх этих же векторов, обученная на паре сотен размеченных
    вручную кадров, — способ обойти это, не дообучая сеть целиком.

    Раньше векторы выбрасывались: задача была «проверить гипотезу», и
    хранить 512 чисел на кадр казалось лишним. Оказалось, что именно они и
    нужны, а пересчёт стоит десять минут.
    """
    import torch
    from PIL import Image

    model, preprocess, tokenizer, device = _load_model()
    log(f"  устройство: {device}, картинок: {len(paths)}")

    axes = {name: (_text_vectors(model, tokenizer, device, pos),
                   _text_vectors(model, tokenizer, device, neg))
            for name, (pos, neg) in PROMPT_PAIRS.items()}

    rows = []
    embeddings = [] if keep_embeddings else None
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
        raise SystemExit("Нет фотографий: python -m kz.collect.photo_fetch")
    pos = idx["position"].to_numpy() if all_positions else np.ones(len(idx), int)
    log(f"Фотографий: {len(idx)} у {idx['ad_id'].nunique()} объявлений")
    scores, emb = score_photos(idx["path"].tolist(), log=log,
                               keep_embeddings=True)
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
        # Пути нужны, чтобы связать вектор с файлом на диске: без них
        # разметка по кадрам не сошлась бы с эмбеддингами.
        path=idx["path"].to_numpy().astype("U160"),
        emb=emb)
    log(f"Сохранено → {CLIP_PATH}  "
        f"(оценки {scores[value_cols].shape}, векторы {emb.shape})")
    return scores


def load_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    """Векторы CLIP плюс их привязка к кадру: ad_id, позиция, путь к файлу.

    Отдельно от load(), потому что весят они на два порядка больше оценок, а
    нужны только для обучения на своих метках.
    """
    if not CLIP_PATH.exists():
        raise FileNotFoundError("Сначала: python -m kz.ml.photo_clip")
    z = np.load(CLIP_PATH, allow_pickle=False)
    if "emb" not in z.files:
        raise KeyError("В артефакте нет векторов — пересчитайте "
                       "python -m kz.ml.photo_clip")
    idx = pd.DataFrame({"ad_id": [str(a) for a in z["ad_id"]],
                        "position": z["position"],
                        "path": [str(p) for p in z["path"]]})
    return idx, z["emb"]


def load() -> pd.DataFrame:
    if not CLIP_PATH.exists():
        raise FileNotFoundError("Сначала: python -m kz.ml.photo_clip")
    z = np.load(CLIP_PATH, allow_pickle=False)
    df = pd.DataFrame(z["scores"], columns=[str(c) for c in z["cols"]])
    df.insert(0, "ad_id", [str(a) for a in z["ad_id"]])
    # старые файлы позицию не хранили — там всё было обложками
    df.insert(1, "position",
              z["position"] if "position" in z.files else 1)
    return df


def aggregate(per_photo: pd.DataFrame) -> pd.DataFrame:
    """Оценки кадров → одна строка на объявление.

    Две сводки, потому что они отвечают на разные вопросы:

      max  — «есть ли ХОТЬ ОДИН кадр, похожий на битую машину». Именно так
             и выглядит честное объявление о повреждённой машине: четыре
             обычных снимка и один с помятым крылом. Среднее такой кадр
             размажет и потеряет.

      mean — «машина в целом выглядит плохо». Устойчивее к случайному
             неудачному ракурсу, но слепа к одиночной улике.

      cover — оценка обложки, чтобы было с чем сравнивать: весь смысл
             затеи в том, помогают ли кадры 2-5 сверх парадного.
    """
    cols = [c for c in per_photo.columns if c not in ("ad_id", "position")]
    g = per_photo.groupby("ad_id")
    out = g[cols].max().add_suffix("_max").join(
          g[cols].mean().add_suffix("_mean"))
    cover = (per_photo.sort_values("position")
                      .drop_duplicates("ad_id")
                      .set_index("ad_id")[cols].add_suffix("_cover"))
    out = out.join(cover)
    out["n_photos"] = g.size()
    return out.reset_index()


def validate(log=print) -> None:
    """Отличает ли оценка машины, про которые мы знаем правду со стороны.

    Разметки состояния у нас нет, зато есть два внешних свидетеля: бейдж
    сайта «Аварийная/Не на ходу» и damage-слова в тексте объявления.

    Меряем AUC, а не разницу средних. AUC (площадь под ROC-кривой) читается
    буквально: возьмём наугад одну битую машину и одну целую — с какой
    вероятностью оценка у битой окажется выше. 0.5 — монетка, оценка не
    знает ничего; 1.0 — идеальное разделение. Разница средних этого не
    показывает: она может быть заметной из-за нескольких выбросов, тогда
    как ранжирование останется случайным.

    Главный вопрос замера — не «работает ли CLIP вообще», а **добавляют ли
    кадры 2-5 что-то сверх обложки**. Поэтому три колонки рядом: _cover,
    _max и _mean.
    """
    from sklearn.metrics import roc_auc_score

    from kz.core.db import get_engine

    per_photo = load()
    d = aggregate(per_photo)
    cd = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge, price_tenge, age "
        "FROM clean_data", get_engine(), dtype={"ad_id": str})
    d = d.merge(cd, on="ad_id", how="left")
    d["has_damage"] = d.damage_keywords.fillna("").str.len() > 0
    d["bad_badge"] = d.page_status_badge.fillna("-").str.contains(
        "вар|ход|залож", case=False)

    log(f"Объявлений с оценкой: {len(d)}   кадров: {len(per_photo)}   "
        f"в среднем {len(per_photo)/max(len(d),1):.1f} на объявление\n")

    bases = sorted({c for c in per_photo.columns
                    if c not in ("ad_id", "position")})
    for flag, name in [("has_damage", "damage-слова в тексте"),
                       ("bad_badge", "бейдж «Аварийная/Не на ходу»")]:
        y = d[flag].to_numpy()
        if y.sum() < 5 or (~y).sum() < 5:
            log(f"{name}: примеров {int(y.sum())} — слишком мало для AUC\n")
            continue
        log(f"{name}: {int(y.sum())} против {int((~y).sum())}   (AUC, 0.5 = монетка)")
        log(f"   {'ось':14} {'обложка':>9} {'максимум':>9} {'среднее':>9}")
        for base in bases:
            cells = []
            for suffix in ("_cover", "_max", "_mean"):
                col = base + suffix
                cells.append(f"{roc_auc_score(y, d[col]):9.3f}"
                             if col in d else f"{'—':>9}")
            log(f"   {base:14} " + " ".join(cells))
        log("")

    log("Кадры 2-5 полезны ровно настолько, насколько «максимум» и «среднее»")
    log("обгоняют «обложку»: обложка — парадный ракурс, и если сигнал есть")
    log("только там, значит повреждения показывают уже на первом снимке.\n")
    _redundancy_check(d, log=log)


def _redundancy_check(d: pd.DataFrame, log=print) -> None:
    """Не переоткрывает ли CLIP возраст и цену.

    Главная ловушка всего замера. «Ржавая» и «грязная» машина — почти
    синонимы «старой и дешёвой», а возраст с ценой у модели уже есть.
    Высокий AUC сам по себе не доказывает пользу: он докажет её только
    если оценка добавляет что-то СВЕРХ того, что мы и так знаем.

    Поэтому сравниваем две логистические регрессии: на одних возрасте с
    ценой и на них же плюс оценка CLIP. Прирост AUC и есть вся польза.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    d = d.copy()
    d["log_price"] = np.log(pd.to_numeric(d["price_tenge"], errors="coerce"))
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    probes = [c for c in ("clip_damaged_max", "clip_rusty_mean",
                          "clip_dirty_mean") if c in d]

    log("Добавляет ли CLIP что-то СВЕРХ возраста и цены:")
    for flag, name in [("has_damage", "damage-слова"),
                       ("bad_badge", "бейдж «Аварийная»")]:
        work = d[["age", "log_price", flag] + probes].dropna()
        y = work[flag].to_numpy().astype(int)
        if y.sum() < 5:
            log(f"  {name}: {int(y.sum())} примеров — слишком мало")
            continue
        base = StandardScaler().fit_transform(work[["age", "log_price"]])
        auc0 = roc_auc_score(y, LogisticRegression(max_iter=1000)
                             .fit(base, y).predict_proba(base)[:, 1])
        log(f"  {name} ({int(y.sum())} положительных): "
            f"возраст+цена дают AUC {auc0:.3f}")
        for c in probes:
            X = StandardScaler().fit_transform(work[["age", "log_price", c]])
            auc = roc_auc_score(y, LogisticRegression(max_iter=1000)
                                .fit(X, y).predict_proba(X)[:, 1])
            log(f"     + {c:20} {auc:.3f}  ({auc - auc0:+.3f})")


def main():
    if "--validate" in sys.argv:
        validate()
        return
    build()
    validate()


if __name__ == "__main__":
    main()
