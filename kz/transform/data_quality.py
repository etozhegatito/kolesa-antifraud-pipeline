# -*- coding: utf-8 -*-
"""
data_quality.py — санитария данных ПЕРЕД моделью цены.

Два уровня, намеренно разные по «жёсткости»:

  1. Точечный авто-скраб (детерминированный): плейсхолдеры пробега вида
     777777 / 999999 — продавец «забил» поле повтором цифры, это НЕ реальный
     пробег. Такое зануляем (→ NaN), иначе модель учит 777k км как настоящие.
     Правило узкое (репдигит И >300k км), чтобы не задеть реальные 99999/150000.

  2. Isolation Forest — ШИРОКАЯ сеть для РЕВЬЮ, НЕ для авто-удаления и НЕ для
     фрода. Проверено на данных (см. учебник): из его топа почти всё — либо
     редкое-но-честное (G-класс, старьё), либо мусор в полях. Для приманок он
     не годится (глобальный выброс ≠ «дёшево для этой машины»). Поэтому это
     флаг «глянь глазами перед обучением», а не фильтр.
"""

import numpy as np
import pandas as pd

# Признаки для iForest-ревью (числовые; NaN заполняются медианой внутри).
DQ_FEATURES = ["age", "mileage_km", "engine_volume", "photos_count"]


def is_junk_mileage(m) -> bool:
    """Плейсхолдер пробега: одна цифра повторена (≥5 знаков) И значение
    неправдоподобно высокое (>300k км) → это «забитое» поле, не пробег.
    Чистая функция — тестируется без данных. 99999/111111/150000 — НЕ junk."""
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return False
    try:
        s = str(int(m))
    except (ValueError, TypeError):
        return False
    return len(s) >= 5 and len(set(s)) == 1 and int(m) > 300_000


def scrub_junk_mileage(df: pd.DataFrame, col: str = "mileage_km"):
    """Плейсхолдер-пробеги → NaN (модель посчитает их пропуском, а не
    реальными 777k км). Возвращает (df, сколько занулено)."""
    df = df.copy()
    junk = df[col].map(is_junk_mileage)
    df.loc[junk, col] = np.nan
    return df, int(junk.sum())


def iforest_anomaly(df: pd.DataFrame, features=None, contamination: float = 0.02) -> pd.Series:
    """Флаг многомерной аномалии (Isolation Forest) для РЕВЬЮ качества данных.
    НЕ авто-удаление и НЕ фрод — ловит глобальные выбросы (редкие/старые
    машины, мусорные значения). Возвращает bool-Series по индексу df."""
    from sklearn.ensemble import IsolationForest
    feats = features or DQ_FEATURES
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    iso = IsolationForest(contamination=contamination, random_state=42)
    return pd.Series(iso.fit_predict(X) == -1, index=df.index)
