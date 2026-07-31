# -*- coding: utf-8 -*-
"""Кривая обучения: упрётся ли качество в объём данных или в признаки.

Вопрос «надо ли парсить больше» нельзя решить рассуждением — только замером.
Модель учится на подвыборках растущего размера, и смотрим на MAPE:

  ошибка ещё падает к правому краю  → данных мало, сбор поможет;
  кривая вышла на плато            → упёрлись в ПРИЗНАКИ, а не в объём,
                                      и новые строки того же вида ничего
                                      не добавят.

Подвыборка берётся по ГРУППАМ дублей, а не по строкам: иначе перезалив
одной машины попал бы и в train, и в test, и кривая соврала бы в свою
пользу. Внутри каждой доли — тот же grouped CV, что и в основном обучении,
поэтому числа сравнимы с метриками в metadata.

Запуск: python -m kz.ml.learning_curve
Выход:  консоль + data/eda/learning_curve.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    duplicate_groups,
    load,
    new_model,
    prepare_training_data,
    regression_metrics,
)

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
N_SPLITS = 5
OUT_PNG = Path("data/eda/learning_curve.png")
RANDOM_SEED = 42


def subsample_by_groups(df: pd.DataFrame, groups: pd.Series, frac: float,
                        seed: int = RANDOM_SEED):
    """Доля данных, отобранная целыми группами дублей."""
    if frac >= 1.0:
        return df, groups
    uniq = pd.Series(groups.unique())
    keep = uniq.sample(frac=frac, random_state=seed)
    mask = groups.isin(set(keep))
    return df[mask], groups[mask]


def cv_mape(df: pd.DataFrame, groups: pd.Series, n_splits: int = N_SPLITS) -> dict:
    """Grouped CV на данном срезе: MAPE и R² в log-пространстве."""
    from catboost import Pool
    from sklearn.model_selection import GroupKFold

    n = min(n_splits, groups.nunique())
    if n < 2:
        raise ValueError("мало независимых групп")
    X, y = df[FEATURES], df["log_price"]
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        model = new_model()
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        oof[te] = model.predict(X.iloc[te])
    return regression_metrics(y, oof)


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["rows"] for r in rows]
    mape = [r["mape_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.plot(n, mape, "o-", color="#2563c9", lw=2)
    for x, y in zip(n, mape):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("строк в обучении")
    ax.set_ylabel("MAPE, % (grouped CV)")
    ax.set_title("Кривая обучения: падает ли ошибка с ростом данных")
    ax.grid(alpha=.3)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    print(f"\nГрафик → {OUT_PNG}")


def main():
    df = prepare_training_data(load())
    groups_all = duplicate_groups(df)
    print(f"Всего чистых строк: {len(df)}, независимых групп: "
          f"{groups_all.nunique()}\n")

    rows = []
    for frac in FRACTIONS:
        part, groups = subsample_by_groups(df, groups_all, frac)
        m = cv_mape(part, groups)
        rows.append({"frac": frac, "rows": len(part), **m})
        print(f"  {frac*100:5.0f}%  строк={len(part):5d}  "
              f"MAPE={m['mape_pct']:5.2f}%  R²(log)={m['r2_log']:.3f}  "
              f"MAE={m['mae_tenge']/1e6:.2f}М")

    # Насколько ошибка снизилась на последней прибавке данных: это и есть
    # ответ «даст ли ещё сбор». Плато = упёрлись в признаки.
    last_gain = rows[-2]["mape_pct"] - rows[-1]["mape_pct"]
    first_gain = rows[0]["mape_pct"] - rows[1]["mape_pct"]
    print(f"\nВыигрыш от первой прибавки данных: {first_gain:+.2f} п.п. MAPE")
    print(f"Выигрыш от последней прибавки:     {last_gain:+.2f} п.п. MAPE")
    if abs(last_gain) < 0.5:
        print("→ Кривая на плато: новые строки ТОГО ЖЕ вида почти ничего не дают.")
        print("  Улучшать надо признаки (комплектация, состояние, фото), а не объём.")
    else:
        print("→ Ошибка ещё падает: сбор данных пока окупается.")
    plot(rows)


if __name__ == "__main__":
    main()
