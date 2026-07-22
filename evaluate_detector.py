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
    lab = lab[lab["verdict"].isin(["fraud", "legit"])]
    return clean.merge(lab[["ad_id", "verdict"]], on="ad_id", how="inner")


def confusion(df: pd.DataFrame) -> dict:
    is_fraud = df["verdict"] == "fraud"
    flagged = df["is_suspicious"] == 1
    return {
        "TP": int((is_fraud & flagged).sum()),    # фрод, поймали
        "FP": int((~is_fraud & flagged).sum()),   # честный, зря пометили
        "FN": int((is_fraud & ~flagged).sum()),   # фрод, пропустили
        "TN": int((~is_fraud & ~flagged).sum()),  # честный, верно пропустили
    }


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
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else float("nan"))

    print(f"\nРазмечено: {n} (фрод: {n_fraud}, честных: {n - n_fraud})")
    if n < MIN_FOR_METRICS:
        print(f"⚠ мало для устойчивой оценки (нужно ≥ {MIN_FOR_METRICS}) — "
              "числа ниже ориентировочные")

    print("\n► Матрица ошибок")
    print(f"                 помечен   не помечен")
    print(f"  реально фрод      {c['TP']:>4}       {c['FN']:>4}   ← FN = пропущенный фрод (опасно)")
    print(f"  реально честный   {c['FP']:>4}       {c['TN']:>4}   ← FP = ложная тревога (шум)")

    print("\n► Метрики")
    print(f"  precision = {precision:.1%}   (из помеченных — сколько реально фрод)")
    print(f"  recall    = {recall:.1%}   (из всего фрода — сколько поймали)")
    print(f"  F1        = {f1:.1%}   (баланс двух)")

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
