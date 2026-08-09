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
    reduced = photo_features.reduce_embeddings(emb)
    ecols = [f"img_pc{i}" for i in range(reduced.shape[1])]
    df[ecols] = reduced
    print(f"Эмбеддинг сжат до {reduced.shape[1]} компонент\n")

    runs = [
        ("только табличные признаки", FEATURES, CAT_FEATURES),
        ("+ качество снимка", FEATURES + qcols, CAT_FEATURES),
        ("+ эмбеддинг картинки", FEATURES + qcols + ecols, CAT_FEATURES),
    ]
    base = None
    for label, feats, cats in runs:
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
