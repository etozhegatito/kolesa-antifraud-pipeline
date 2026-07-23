# -*- coding: utf-8 -*-
"""
evaluate_detector.py — доказательство, что детектор работает (или нет).

Считает precision/recall/F1 флага is_suspicious ПРОТИВ ручных вердиктов
(ground truth из data/manual_labels.csv) + матрицу ошибок + precision
КАЖДОГО правила по отдельности (какой флаг ловит фрод, а какой красит
честных). Пока вердиктов нет — печатает, что и сколько разметить, и
выходит без ошибки: харнесс готов заранее, метрика появится в тот же
миг, когда появится разметка.

Ground truth: verdict='fraud' → положительный класс, 'legit' →
отрицательный, 'unknown' и без вердикта → исключаются из счёта.
Предсказание модели: is_suspicious из clean_data.

Запуск: python evaluate_detector.py         (читает Postgres + CSV)
Полностью офлайн, ни одного запроса к сайту.
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
import pathlib as _p
_expected = "evaluate_detector.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import math
from pathlib import Path

import pandas as pd

from db import get_engine

LABELS_CSV = "data/manual_labels.csv"
LINE = "─" * 64
MIN_FOR_METRICS = 20   # ниже — числа слишком шумные, чтобы им верить


def load_labeled() -> pd.DataFrame:
    """clean_data (предсказания) ⋈ manual_labels (истина), только
    строки с однозначным вердиктом fraud/legit."""
    clean = pd.read_sql(
        "SELECT ad_id, is_suspicious, suspicion_reasons FROM clean_data",
        get_engine(), dtype={"ad_id": str})
    if not Path(LABELS_CSV).exists():
        return pd.DataFrame()
    lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
    lab["verdict"] = lab["verdict"].astype("string").str.strip().str.lower()
    # manual_labels — append-only журнал. При уточнении решения один ad_id
    # встречается повторно; последняя запись является актуальной.
    lab = lab.drop_duplicates("ad_id", keep="last")
    lab = lab[lab["verdict"].isin(["fraud", "legit"])]
    optional = [
        c for c in ["sampling_stratum", "stratum_population", "stratum_sample_size"]
        if c in lab.columns
    ]
    return clean.merge(lab[["ad_id", "verdict", *optional]], on="ad_id", how="inner")


def confusion(df: pd.DataFrame) -> dict:
    is_fraud = df["verdict"] == "fraud"
    flagged = df["is_suspicious"] == 1
    return {
        "TP": int((is_fraud & flagged).sum()),    # фрод, поймали
        "FP": int((~is_fraud & flagged).sum()),   # честный, зря пометили
        "FN": int((is_fraud & ~flagged).sum()),   # фрод, пропустили
        "TN": int((~is_fraud & ~flagged).sum()),  # честный, верно пропустили
    }


def weighted_confusion(df: pd.DataFrame) -> dict | None:
    """Оценка по населению через inverse-probability weights очереди.

    Возвращает None для старых ручных строк без sampling metadata: придумывать
    им веса было бы хуже, чем честно не показывать population recall.
    """
    required = {"stratum_population", "stratum_sample_size"}
    if not required.issubset(df.columns):
        return None
    population = pd.to_numeric(df["stratum_population"], errors="coerce")
    sample = pd.to_numeric(df["stratum_sample_size"], errors="coerce")
    if population.isna().any() or sample.isna().any() or (sample <= 0).any():
        return None
    weight = population / sample
    is_fraud = df["verdict"] == "fraud"
    flagged = df["is_suspicious"] == 1
    return {
        "TP": float(weight[is_fraud & flagged].sum()),
        "FP": float(weight[~is_fraud & flagged].sum()),
        "FN": float(weight[is_fraud & ~flagged].sum()),
        "TN": float(weight[~is_fraud & ~flagged].sum()),
    }


def _prf(c: dict) -> tuple[float, float, float]:
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if math.isfinite(precision) and math.isfinite(recall):
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
    else:
        f1 = float("nan")
    return precision, recall, f1


def main():
    df = load_labeled()

    print(LINE)
    print("КАЧЕСТВО ДЕТЕКТОРА (is_suspicious vs ручные вердикты)")
    print(LINE)

    if df.empty:
        print("\nРазмеченных вердиктов пока нет — метрику посчитать не на чем.")
        print("Что делать:")
        print("  1. открой data/eda/labeling_queue.csv (очередь, худшие сверху)")
        print("  2. пройди по url, проставь verdict = fraud / legit / unknown")
        print("  3. скопируй размеченные строки в data/manual_labels.csv")
        print("  4. запусти clean.py (чтобы вердикты вошли), потом снова этот скрипт")
        print(f"\nДля устойчивой метрики нужно ≥ {MIN_FOR_METRICS} вердиктов "
              "fraud/legit (сейчас 0).")
        print(LINE)
        return

    n = len(df)
    n_fraud = int((df["verdict"] == "fraud").sum())
    c = confusion(df)
    precision, recall, f1 = _prf(c)

    print(f"\nРазмечено: {n} (фрод: {n_fraud}, честных: {n - n_fraud})")
    if n < MIN_FOR_METRICS:
        print(f"⚠ мало для устойчивой оценки (нужно ≥ {MIN_FOR_METRICS}) — "
              "числа ниже ориентировочные")
    print("  recall относится к размеченной выборке; для оценки пропусков "
          "обязательно размечай слой random_control из labeling_queue.csv")

    print("\n► Матрица ошибок")
    print(f"                 помечен   не помечен")
    print(f"  реально фрод      {c['TP']:>4}       {c['FN']:>4}   ← FN = пропущенный фрод (опасно)")
    print(f"  реально честный   {c['FP']:>4}       {c['TN']:>4}   ← FP = ложная тревога (шум)")

    print("\n► Метрики")
    print(f"  precision = {precision:.1%}   (из помеченных — сколько реально фрод)")
    print(f"  recall    = {recall:.1%}   (из всего фрода — сколько поймали)")
    print(f"  F1        = {f1:.1%}   (баланс двух)")

    weighted = weighted_confusion(df)
    if weighted is not None:
        wp, wr, wf = _prf(weighted)
        print("\n► Population estimate с весами sampling strata")
        print(f"  precision = {wp:.1%}   recall = {wr:.1%}   F1 = {wf:.1%}")
        print("  Это оценка на весь срез через inverse-probability weights, "
              "а не сырая метрика обогащённой очереди.")
    else:
        print("\n  Population estimate пока недоступен: старые verdict не содержат "
              "sampling metadata из новой трёхслойной очереди.")

    # precision каждого правила — какой флаг работает, какой шумит
    print("\n► Precision по правилам (только среди размеченных)")
    flagged = df[df["is_suspicious"] == 1].copy()
    if flagged.empty:
        print("  (ни одно размеченное объявление не флагнуто)")
    else:
        rows = []
        reasons = (flagged.assign(r=flagged["suspicion_reasons"].str.split("|"))
                   .explode("r"))
        for reason, g in reasons.groupby("r"):
            if not reason:
                continue
            fr = int((g["verdict"] == "fraud").sum())
            rows.append((reason, fr, len(g), fr / len(g)))
        for reason, fr, tot, p in sorted(rows, key=lambda x: -x[3]):
            print(f"  {reason:<24} {fr}/{tot}  precision={p:.0%}")
    print(LINE)


if __name__ == "__main__":
    main()
