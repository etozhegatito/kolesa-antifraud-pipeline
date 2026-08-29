# -*- coding: utf-8 -*-
"""Честная supervised-проверка детектора повреждений на ручных метках.

`photo_clip --rank` специально является только поисковиком для очереди и
обучается на всех метках. По его цифре нельзя судить о качестве детектора.
Этот модуль отвечает на другой вопрос: видит ли фотография повреждение на
НОВОМ объявлении лучше, чем простая подсказка «машина старая и дешёвая».

Защита от трёх распространённых самообманов:

1. Все кадры одного ad_id лежат в одном фолде (`StratifiedGroupKFold`).
2. Метрика считается по ОБЪЯВЛЕНИЯМ: максимум по кадрам, а не будто пять
   фотографий одной машины — пять независимых машин.
3. Рядом всегда печатается baseline из возраста и log(цены). Высокий AUC
   фото ничего не доказывает, если те же классы разделяет таблица.

Запуск: python -m kz.ml.photo_damage
Сети не касается; читает сохранённые CLIP-векторы и ручной CSV-журнал.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kz.core.db import get_engine
from kz.ml.photo_clip import load as load_clip_scores
from kz.ml.photo_clip import load_embeddings
from kz.report.photo_labels import read_journal

CHEAP_EDGE = 5_000_000
N_SPLITS = 5
SEED = 42
BOOTSTRAPS = 1000
POSITIVE = {"damaged", "wreck"}
USED = POSITIVE | {"intact"}


def load_labelled() -> tuple[pd.DataFrame, np.ndarray]:
    """Метки + индекс сохранённых эмбеддингов + цена/возраст."""
    _, rows = read_journal()
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise RuntimeError("Журнал photo_labels пуст")
    labels = labels.drop_duplicates(["ad_id", "position"], keep="last")
    labels = labels[labels.label.isin(USED)].copy()
    labels["ad_id"] = labels.ad_id.astype(str)
    labels["position"] = pd.to_numeric(labels.position, errors="coerce")
    labels = labels.dropna(subset=["position"])
    labels["position"] = labels.position.astype(int)

    idx, emb = load_embeddings()
    idx = idx.reset_index(drop=True)
    idx["embedding_row"] = idx.index
    idx["ad_id"] = idx.ad_id.astype(str)
    idx["position"] = idx.position.astype(int)
    d = labels.merge(idx[["ad_id", "position", "embedding_row"]],
                     on=["ad_id", "position"], how="inner")

    cars = pd.read_sql(
        "SELECT ad_id, age, price_tenge FROM clean_data", get_engine(),
        dtype={"ad_id": str},
    )
    d = d.merge(cars, on="ad_id", how="left")
    d["age"] = pd.to_numeric(d.age, errors="coerce")
    d["price_tenge"] = pd.to_numeric(d.price_tenge, errors="coerce")
    d = d.dropna(subset=["age", "price_tenge"])
    d["target"] = d.label.isin(POSITIVE).astype(int)
    return d.reset_index(drop=True), emb


def _splits(d: pd.DataFrame):
    positive_ads = d.loc[d.target == 1, "ad_id"].nunique()
    negative_ads = d.loc[d.target == 0, "ad_id"].nunique()
    n = min(N_SPLITS, positive_ads, negative_ads)
    if n < 2:
        raise RuntimeError("Нужно хотя бы по два независимых объявления класса")
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    dummy = np.zeros((len(d), 1))
    return list(cv.split(dummy, d.target.to_numpy(), d.ad_id.to_numpy()))


def oof_scores(d: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """OOF-счёты таблицы, фото и их объединения для каждого кадра."""
    y = d.target.to_numpy()
    photo = emb[d.embedding_row.to_numpy()]
    table = np.column_stack([
        d.age.to_numpy(dtype=float),
        np.log(d.price_tenge.to_numpy(dtype=float)),
    ])
    out = d[["ad_id", "target"]].copy()
    out["table"] = np.nan
    out["photo"] = np.nan
    out["combined"] = np.nan

    for tr, te in _splits(d):
        table_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        photo_model = LogisticRegression(
            C=0.003, max_iter=3000, class_weight="balanced"
        )
        combined_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"),
        )
        table_model.fit(table[tr], y[tr])
        photo_model.fit(photo[tr], y[tr])
        joined = np.column_stack([table, photo])
        combined_model.fit(joined[tr], y[tr])
        out.loc[te, "table"] = table_model.predict_proba(table[te])[:, 1]
        out.loc[te, "photo"] = photo_model.predict_proba(photo[te])[:, 1]
        out.loc[te, "combined"] = combined_model.predict_proba(joined[te])[:, 1]
    return out


def per_ad(frame_scores: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на машину: повреждена, если повреждён любой кадр."""
    return frame_scores.groupby("ad_id", as_index=False).agg(
        target=("target", "max"),
        table=("table", "max"),
        photo=("photo", "max"),
        combined=("combined", "max"),
    )


def auc_ci(y, score, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    """AUC и percentile bootstrap CI; ресэмплинг уже идёт по объявлениям."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return auc, float(lo), float(hi)


def auc_delta_ci(y, candidate, baseline,
                 n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    """Парный bootstrap для AUC(candidate) − AUC(baseline).

    Отдельные интервалы двух AUC не отвечают на вопрос, значима ли разница.
    Ресэмплируем одни и те же объявления для обеих моделей, сохраняя их
    корреляцию — это и есть продуктовый gate, описанный в выводе.
    """
    y = np.asarray(y, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    delta = float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline))
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(
                roc_auc_score(y[idx], candidate[idx])
                - roc_auc_score(y[idx], baseline[idx])
            )
    lo, hi = np.percentile(values, [2.5, 97.5])
    return delta, float(lo), float(hi)


def zero_shot_per_ad(d: pd.DataFrame) -> pd.DataFrame:
    scores = load_clip_scores()[["ad_id", "position", "clip_damaged"]].copy()
    scores["ad_id"] = scores.ad_id.astype(str)
    scores["position"] = scores.position.astype(int)
    keys = d[["ad_id", "position", "target"]].merge(
        scores, on=["ad_id", "position"], how="inner"
    )
    return keys.groupby("ad_id", as_index=False).agg(
        target=("target", "max"), zero_shot=("clip_damaged", "max")
    )


def main() -> None:
    d, emb = load_labelled()
    d = d[d.price_tenge < CHEAP_EDGE].reset_index(drop=True)
    n_pos_ads = d.loc[d.target == 1, "ad_id"].nunique()
    n_neg_ads = d.loc[d.target == 0, "ad_id"].nunique()
    print(f"Дешёвый сегмент: {len(d)} кадров, {n_pos_ads} повреждённых и "
          f"{n_neg_ads} целых объявлений")

    ads = per_ad(oof_scores(d, emb))
    zero = zero_shot_per_ad(d)
    ads = ads.merge(zero[["ad_id", "zero_shot"]], on="ad_id", how="left")
    print("\nOOF AUC по независимым объявлениям (95% bootstrap CI):")
    for col, name in [
        ("table", "возраст + цена"),
        ("zero_shot", "CLIP zero-shot"),
        ("photo", "CLIP-вектор, наши метки"),
        ("combined", "таблица + CLIP-вектор"),
    ]:
        work = ads[["target", col]].dropna()
        auc, lo, hi = auc_ci(work.target, work[col])
        print(f"  {name:27} {auc:.3f}  [{lo:.3f}; {hi:.3f}]")
    print("\nПарная разница против возраста+цены:")
    for col, name in [("photo", "CLIP-вектор"),
                      ("combined", "таблица + CLIP")]:
        work = ads[["target", "table", col]].dropna()
        delta, lo, hi = auc_delta_ci(work.target, work[col], work.table)
        print(f"  {name:27} ΔAUC {delta:+.3f}  [{lo:+.3f}; {hi:+.3f}]")
    print("\nКритерий перехода в продукт: фото или combined должны обгонять "
          "возраст+цену, а доверительный интервал разницы не должен накрывать ноль.")


if __name__ == "__main__":
    main()
