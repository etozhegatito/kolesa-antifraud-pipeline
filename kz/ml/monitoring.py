# -*- coding: utf-8 -*-
"""Мониторинг: не разъехались ли данные с теми, на которых училась модель.

ЗАЧЕМ. Модель обучена один раз на срезе данных. Дальше рынок живёт своей
жизнью: приходят другие марки, меняется средний возраст, дорожает сегмент.
Модель об этом не узнает — она будет уверенно предсказывать по устаревшим
закономерностям, и метрики поедут молча.

Проблема называется **дрейф** (drift), и у неё два вида:

  дрейф признаков (data drift) — изменилось распределение входов. Например,
    в базе стало вдвое больше свежих машин, чем было при обучении.

  дрейф качества (concept drift) — связь признаков с ценой изменилась.
    Например, подорожал бензин, и дизельные машины стали цениться иначе.
    Этот вид опаснее: входы выглядят прежними, а модель уже неправа.

КАК МЕРЯЕМ ДРЕЙФ ПРИЗНАКОВ. Индекс стабильности популяции (PSI, population
stability index) сравнивает два распределения по корзинам:

    PSI = сумма по корзинам (доля_сейчас − доля_тогда) × ln(доля_сейчас /
                                                            доля_тогда)

Читается так: для каждой корзины берём насколько изменилась её доля и во
сколько раз она изменилась, перемножаем и складываем. Общепринятые пороги:

    PSI < 0,1     распределение стабильно
    0,1 … 0,25    заметный сдвиг, стоит присмотреться
    PSI > 0,25    сильный сдвиг, модель пора переобучать

Пороги эмпирические, из банковского скоринга, где метод и появился.

Запуск: python -m kz.ml.monitoring
Выход:  консоль + запись в data/eda/monitoring_history.csv
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path("data/eda/monitoring_history.csv")
N_BINS = 10
PSI_WATCH, PSI_ALERT = 0.10, 0.25

HISTORY_COLS = ["checked_at", "training_rows", "current_rows", "max_psi",
                "max_psi_feature", "n_watch", "n_alert", "model_created",
                "model_mape_pct"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = N_BINS) -> float:
    """Индекс стабильности между двумя выборками одного признака.

    Границы корзин берутся по «эталонной» выборке (той, на которой учились):
    вопрос ведь в том, как новые данные распределились по СТАРЫМ корзинам.
    Пустые корзины заменяются малым числом, иначе логарифм ушёл бы в
    бесконечность из-за одной пустой ячейки.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) < bins:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """То же для категорий: корзины — сами значения."""
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    eps = 1e-6
    ev = np.clip(e.reindex(idx).fillna(0).to_numpy(), eps, None)
    av = np.clip(a.reindex(idx).fillna(0).to_numpy(), eps, None)
    return float(np.sum((av - ev) * np.log(av / ev)))


def level(value: float) -> str:
    if not np.isfinite(value):
        return "нет данных"
    if value > PSI_ALERT:
        return "СИЛЬНЫЙ сдвиг"
    if value > PSI_WATCH:
        return "заметный сдвиг"
    return "стабильно"


def compare(training: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI по каждому признаку модели."""
    from kz.ml.train_price_model import CAT_FEATURES, NUM_FEATURES

    rows = []
    for f in NUM_FEATURES:
        if f in training.columns and f in current.columns:
            rows.append({"feature": f, "type": "число",
                         "psi": psi(training[f].to_numpy(), current[f].to_numpy())})
    for f in CAT_FEATURES:
        if f in training.columns and f in current.columns:
            rows.append({"feature": f, "type": "категория",
                         "psi": categorical_psi(training[f], current[f])})
    out = pd.DataFrame(rows)
    out["уровень"] = out["psi"].map(level)
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def training_snapshot() -> pd.DataFrame:
    """Данные, на которых училась текущая модель.

    Точного снимка обучающей выборки мы не храним — он весил бы столько же,
    сколько сама таблица. Зато в метаданных записано, сколько было строк и
    когда, а clean_data пересобирается из неизменяемого сырья. Поэтому берём
    самые ранние строки в том же количестве: это честное приближение того,
    что видела модель.
    """
    from kz.ml.train_price_model import load, load_artifact, prepare_training_data

    _, meta = load_artifact()
    n = int(meta.get("training_rows", 0))
    df = prepare_training_data(load())
    if "scraped_at" in df.columns:
        df = df.sort_values("scraped_at")
    return df.head(n) if 0 < n < len(df) else df


def append_history(row: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    fresh = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        if fresh:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in HISTORY_COLS})


def main():
    from kz.ml.train_price_model import load, load_artifact, prepare_training_data

    _, meta = load_artifact()
    current = prepare_training_data(load())
    train = training_snapshot()
    trained_on = int(meta.get("training_rows", 0))
    fresh_rows = len(current) - trained_on

    print(f"Модель обучена {meta.get('created_at_utc','?')[:16]} "
          f"на {trained_on} строках")
    print(f"Сейчас в чистом слое: {len(current)} строк (+{fresh_rows})\n")

    # Вырожденный случай: модель обучена ровно на тех данных, что лежат
    # сейчас. Сравнивать нечего — снимок обучающей выборки И ЕСТЬ текущая
    # выборка, PSI выйдет нулевым по построению, а не потому что данные
    # стабильны. Печатать в такой ситуации «данные стабильны» — значит
    # успокаивать отчётом, который ничего не проверил.
    if fresh_rows <= 0:
        print("Сравнивать не с чем: модель обучена ровно на текущих данных.")
        print("Дрейф измеряется между обучением и НОВЫМИ данными, поэтому")
        print("проверку имеет смысл делать ДО переобучения, а не после.")
        print("\nЗамер не записан в историю — нулю здесь верить нельзя.")
        return

    table = compare(train, current)
    print("Сдвиг распределения признаков (PSI):")
    for _, r in table.iterrows():
        mark = "  ←" if r["psi"] > PSI_WATCH else ""
        print(f"  {r['feature']:20} {r['psi']:6.3f}   {r['уровень']}{mark}")

    n_watch = int((table["psi"] > PSI_WATCH).sum())
    n_alert = int((table["psi"] > PSI_ALERT).sum())
    top = table.iloc[0]

    val = meta.get("validation", {}).get("grouped_cv", {}).get("model", {})
    append_history({
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": meta.get("training_rows"),
        "current_rows": len(current),
        "max_psi": round(float(top["psi"]), 4),
        "max_psi_feature": top["feature"],
        "n_watch": n_watch, "n_alert": n_alert,
        "model_created": meta.get("created_at_utc", "")[:16],
        "model_mape_pct": round(float(val.get("mape_pct", float("nan"))), 2),
    })

    print(f"\nИтог: заметный сдвиг у {n_watch} признаков, сильный у {n_alert}.")
    if n_alert:
        print("→ Пора переобучать: python -m kz.ops.run_all --ml")
    elif n_watch:
        print("→ Пока терпимо, но стоит следить.")
    else:
        print("→ Данные стабильны относительно обучающей выборки.")
    print(f"\nИстория проверок → {HISTORY}")
    print("Она нужна, чтобы видеть ТРЕНД: одиночный замер не отличает "
          "случайное колебание от медленного расхождения.")


if __name__ == "__main__":
    main()
