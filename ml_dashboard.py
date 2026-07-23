# -*- coding: utf-8 -*-
"""
ml_dashboard.py — визуальный отчёт по МОДЕЛИ цены (аналог explore.py, но про ML).

Формирует data/eda/ml_dashboard.png из 4 панелей + печатает сводку:
  1. Предсказание vs факт (log-log) — насколько модель попадает;
  2. Важность признаков — что реально двигает цену;
  3. Ошибка (MAPE) по возрастным корзинам — ГДЕ модель слаба;
  4. Распределение остатков — есть ли смещение/хвосты.

Метрики честные — out-of-fold (каждая точка предсказана, когда была в
тест-фолде, а не в обучении). Обучаем только на чистых (is_suspicious=0).

Запуск: python ml_dashboard.py   (офлайн, только Postgres)
"""

import matplotlib
matplotlib.use("Agg")                       # рендер в файл без окна
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data_quality import scrub_junk_mileage
from train_price_model import FEATURES, grouped_oof_predictions, load, load_artifact

OUT_PNG = "data/eda/ml_dashboard.png"
AGE_ORDER = ["0-3", "4-7", "8-12", "13-20", "21+"]

plt.rcParams.update({
    "figure.facecolor": "#12141a", "axes.facecolor": "#191c24",
    "axes.edgecolor": "#3a3f4d", "axes.labelcolor": "#e6e6e6",
    "text.color": "#e6e6e6", "xtick.color": "#aab", "ytick.color": "#aab",
    "grid.color": "#2a2e3a", "axes.grid": True, "grid.linewidth": 0.5,
    "font.size": 10,
})
C_OK, C_BAD, C_ACC = "#4fa3ff", "#ff5d5d", "#ffd166"


def main():
    df = load()
    df = df[df["price_tenge"].notna() & (df["price_tenge"] > 0)]
    df["log_price"] = np.log(df["price_tenge"])
    df, _ = scrub_junk_mileage(df)
    clean = df[df["is_suspicious"] == 0].reset_index(drop=True)
    X, y = clean[FEATURES], clean["log_price"]

    # Grouped OOF: точку не видит модель, и её точный перезалив также
    # не может оказаться в train-фолде.
    oof, baseline_oof = grouped_oof_predictions(clean)
    clean["oof_log"] = oof
    clean["ape"] = np.abs(np.exp(oof) - np.exp(y)) / np.exp(y) * 100

    r2 = r2_score(y, oof)
    mae = mean_absolute_error(np.exp(y), np.exp(oof))
    mape = float(clean["ape"].mean())
    final, _ = load_artifact()                # тот же артефакт, что в inference
    baseline_mape = float(
        (np.abs(np.exp(baseline_oof) - np.exp(y)) / np.exp(y) * 100).mean()
    )

    # ── сводка в консоль ────────────────────────────────────────────────────
    print(f"Модель цены (grouped out-of-fold, {len(X)} машин, чистые):")
    print(f"  R²(log) = {r2:.3f}   MAPE = {mape:.1f}%   MAE = {mae/1e6:.2f}М ₸")
    print(f"  baseline MAPE = {baseline_mape:.1f}%")

    # ── дашборд ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Модель оценки цены · R²={r2:.3f} · MAPE={mape:.1f}%  "
                 f"(grouped OOF, {len(X)} машин)", fontsize=14, fontweight="bold")

    # 1) предсказание vs факт (log-log)
    a = ax[0, 0]
    a.scatter(np.exp(y), np.exp(oof), s=8, alpha=.3, color=C_OK)
    lim = [np.exp(y).min(), np.exp(y).max()]
    a.plot(lim, lim, color=C_ACC, ls="--", lw=1.5, label="идеал (пред=факт)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title("Предсказание vs факт"); a.set_xlabel("факт, ₸"); a.set_ylabel("предикт, ₸")
    a.legend(fontsize=8)

    # 2) важность признаков (топ-12)
    a = ax[0, 1]
    imp = sorted(zip(FEATURES, final.get_feature_importance()), key=lambda t: t[1])[-12:]
    a.barh([f for f, _ in imp], [v for _, v in imp], color=C_OK, alpha=.85)
    a.set_title("Важность признаков (что двигает цену)")

    # 3) MAPE по возрасту — где модель слаба
    a = ax[1, 0]
    g = clean.groupby("age_bucket")["ape"].mean().reindex(AGE_ORDER)
    a.bar(g.index.astype(str), g.values, color=C_BAD, alpha=.85)
    a.axhline(mape, color=C_ACC, ls="--", lw=1.2, label=f"средний {mape:.0f}%")
    a.set_title("Ошибка (MAPE) по возрасту — старьё предсказуемо хуже")
    a.set_xlabel("возраст, лет"); a.set_ylabel("MAPE, %"); a.legend(fontsize=8)

    # 4) распределение остатков (лог) — смещение/хвосты
    a = ax[1, 1]
    resid = y.values - oof
    a.hist(resid, bins=60, color=C_OK, alpha=.85)
    a.axvline(0, color=C_ACC, ls="--", lw=1.2)
    a.set_title("Остатки log(факт)−log(предикт): 0=точно, <0=дороже модели")
    a.set_xlabel("остаток (лог)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nДашборд → {OUT_PNG}")


if __name__ == "__main__":
    main()
