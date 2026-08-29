# -*- coding: utf-8 -*-
"""Помогают ли фотографии предсказывать цену — честный замер.

Это проверка гипотезы, ради которой скачивались снимки: на дешёвых и старых
машинах цену определяет состояние, в табличных признаках его нет, а на фото
оно видно. Если гипотеза верна, добавление признаков из изображения снизит
ошибку. Если нет — узнаем это дёшево и не потащим бесполезную сложность в
production.

Сравниваются три набора на ОДНИХ И ТЕХ ЖЕ разбиениях (paired comparison —
иначе разница утонет в разбросе между фолдами, который у нас около 1 п.п.):

  1. только табличные признаки          — то, что есть сейчас;
  2. + метрики качества снимка          — резкость, яркость, контраст;
  3. + сжатый эмбеддинг ResNet50        — содержимое картинки.

ВАЖНО ПРО ЧЕСТНОСТЬ ЗАМЕРА. Сравнение идёт только по машинам, у которых
фотография скачана. Иначе набор 1 считался бы на всех 4334 строках, а
наборы 2-3 на подмножестве, и разница отражала бы состав выборки, а не
пользу от фото.

Запуск: python -m kz.ml.photo_ablation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from kz.ml import photo_features
from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    duplicate_groups,
    load,
    new_model,
    prepare_training_data,
    regression_metrics,
)

N_SPLITS = 5


def cv_mape(df: pd.DataFrame, features: list[str], cats: list[str]) -> dict:
    """Grouped CV на заданном наборе признаков."""
    groups = duplicate_groups(df)
    n = min(N_SPLITS, groups.nunique())
    X, y = df[features], df["log_price"]
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        m = new_model()
        m.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=cats))
        oof[te] = m.predict(X.iloc[te])
    r = regression_metrics(y, oof)
    ape = np.abs(np.exp(oof) - np.exp(y)) / np.exp(y)
    return {"mape": r["mape_pct"], "median": float(np.median(ape) * 100),
            "r2": r["r2_log"]}


def cv_mape_with_embeddings(df: pd.DataFrame, emb: np.ndarray,
                            base_features: list[str],
                            cats: list[str]) -> dict:
    """Grouped CV с PCA, обученной только внутри train-фолда.

    PCA не использует цену, но это всё равно обучаемое преобразование:
    если сделать ``fit_transform`` до CV, направления компонент увидят
    распределение test-фотографий. Это feature leakage. Поэтому каждый
    фолд получает собственную PCA, fit только на ``tr``.
    """
    groups = duplicate_groups(df)
    n_splits = min(N_SPLITS, groups.nunique())
    y = df["log_price"].to_numpy()
    oof = np.full(len(df), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(df, y, groups):
        n_components = min(photo_features.N_COMPONENTS, len(tr), emb.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        train_pc = pca.fit_transform(emb[tr])
        test_pc = pca.transform(emb[te])
        ecols = [f"img_pc{i}" for i in range(n_components)]

        train_x = df.iloc[tr][base_features].reset_index(drop=True).copy()
        test_x = df.iloc[te][base_features].reset_index(drop=True).copy()
        train_x[ecols] = train_pc
        test_x[ecols] = test_pc

        model = new_model()
        model.fit(Pool(train_x, y[tr], cat_features=cats))
        oof[te] = model.predict(test_x)

    r = regression_metrics(y, oof)
    ape = np.abs(np.exp(oof) - np.exp(y)) / np.exp(y)
    return {"mape": r["mape_pct"], "median": float(np.median(ape) * 100),
            "r2": r["r2_log"]}


def main():
    df = prepare_training_data(load())
    quality, emb = photo_features.load()

    # Только машины с фотографией: иначе сравнивали бы разные выборки.
    df = df.merge(quality, on="ad_id", how="inner")
    order = {a: i for i, a in enumerate(quality["ad_id"])}
    emb = emb[[order[a] for a in df["ad_id"]]]
    df = df.reset_index(drop=True)
    print(f"Машин с фотографией и ценой: {len(df)}\n")
    if len(df) < 300:
        print("⚠ Выборка мала — разница утонет в шуме. Докачай фотографии:")
        print("  python -m kz.collect.photo_fetch --limit 1200")

    qcols = [c for c in quality.columns if c != "ad_id"]
    n_components = min(photo_features.N_COMPONENTS, len(df), emb.shape[1])
    print(f"Эмбеддинг сжимается до {n_components} компонент ВНУТРИ каждого "
          "train-фолда\n")

    runs = [
        ("только табличные признаки", FEATURES, CAT_FEATURES),
        ("+ качество снимка", FEATURES + qcols, CAT_FEATURES),
        ("+ эмбеддинг картинки", None, CAT_FEATURES),
    ]
    base = None
    for label, feats, cats in runs:
        if feats is None:
            r = cv_mape_with_embeddings(df, emb, FEATURES + qcols, cats)
        else:
            r = cv_mape(df, feats, cats)
        delta = "" if base is None else f"   {r['mape'] - base:+.2f} п.п."
        base = r["mape"] if base is None else base
        print(f"  {label:28} MAPE={r['mape']:5.2f}%  медиана={r['median']:5.2f}%"
              f"  R²={r['r2']:.3f}{delta}")

    print("\nРазница меньше 0.5 п.п. — шум: разброс между фолдами около 1 п.п.,\n"
          "и только парность сравнения (одни и те же разбиения) делает\n"
          "меньшие различия вообще различимыми.")


if __name__ == "__main__":
    main()
