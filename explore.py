# -*- coding: utf-8 -*-
"""
explore.py — разведочный анализ (EDA) поверх clean_data.csv.

EDA (exploratory data analysis) — этап, когда ты СМОТРИШЬ на данные
глазами до всякого моделирования. Правило: не строй модель на данных,
которые не видел на графике — половина проблем ловится взглядом.

Что делает скрипт:
  1. Консольный отчёт: сводка, топ подозрительных, разбор причин.
  2. Метрика странности №2 — IQR-заборы (в дополнение к z-score из Job 2):
     два независимых детектора; их пересечение = самые надёжные флаги.
  3. Дашборд из 6 графиков → dashboard.png
  4. suspicious_sorted.csv — все флаги, отсортированные по силе аномалии.

Запуск: python explore.py
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "explore.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # рендер в файл без окна
import matplotlib.pyplot as plt

from db import get_engine

OUT_PNG  = "data/eda/dashboard.png"
OUT_SUSP = "data/eda/suspicious_sorted.csv"
CONTROL_SAMPLE_SIZE = 50

# ─── Внешний вид ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#12141a",
    "axes.facecolor":    "#191c24",
    "axes.edgecolor":    "#3a3f4d",
    "axes.labelcolor":   "#e6e6e6",
    "text.color":        "#e6e6e6",
    "xtick.color":       "#aab",
    "ytick.color":       "#aab",
    "grid.color":        "#2a2e3a",
    "axes.grid":         True,
    "grid.linewidth":    0.5,
    "font.size":         10,
})
C_OK, C_BAD, C_ACC, C_INFO = "#4fa3ff", "#ff5d5d", "#ffd166", "#9b7bff"


# ─── IQR: вторая метрика странности ──────────────────────────────────────────
def add_iqr_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-метод (метод Тьюки) по log-цене внутри возрастной корзины.

    Формула по символам:
        Q1  — первый квартиль: значение, ниже которого 25% наблюдений
        Q3  — третий квартиль: ниже него 75% наблюдений
        IQR = Q3 − Q1   («межквартильный размах» — ширина «середины»
                          распределения, где живут центральные 50% данных)
        нижний забор = Q1 − 1.5·IQR
        верхний забор = Q3 + 1.5·IQR
        всё за заборами — выброс.

    Откуда 1.5: эвристика Тьюки. Для нормального распределения за заборы
    выпадает ~0.7% точек — редкое, но не невозможное. Хочешь строже —
    бери 3.0 («экстремальные выбросы»).

    Чем IQR отличается от нашего z-score из Job 2:
      z-score меряет, НАСКОЛЬКО далеко точка (непрерывная шкала),
      IQR даёт бинарный вердикт «за забором / в заборе».
      Оба робастны (строятся на квантилях, устойчивы к мусору).
    Зачем два детектора: пересечение флагов = высокая уверенность.
    Аналогия — два врача с разными методами поставили один диагноз.
    """
    df = df.copy()
    def fences(s: pd.Series) -> pd.DataFrame:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return pd.DataFrame({
            "iqr_low":  q1 - 1.5 * iqr,
            "iqr_high": q3 + 1.5 * iqr,
        }, index=s.index)

    g = df.groupby("age_bucket", observed=True)["log_price"]
    df[["iqr_low", "iqr_high"]] = g.apply(fences).reset_index(level=0, drop=True)
    df["iqr_outlier"] = np.select(
        [df["log_price"] < df["iqr_low"], df["log_price"] > df["iqr_high"]],
        ["low", "high"], default="",
    )
    # Согласие детекторов: и z-score, и IQR кричат «дёшево»
    df["both_detectors_low"] = (
        df["suspicion_reasons"].str.contains("price_anomaly_low", na=False)
        & (df["iqr_outlier"] == "low")
    ).astype(int)
    return df


# ─── Консольный отчёт ─────────────────────────────────────────────────────────
def _write_csv(path: str, df: pd.DataFrame) -> None:
    """CSV без pandas round-trip целых чисел в строки вида '50.0'."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(df.columns))
        writer.writeheader()
        for row in df.to_dict("records"):
            writer.writerow({k: ("" if pd.isna(v) else v) for k, v in row.items()})


def select_labeling_rows(
    df: pd.DataFrame,
    residual_mask: pd.Series | None = None,
    control_n: int = CONTROL_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Три слоя разметки: precision правил, новые ML-кандидаты и FN-контроль.

    Очередь только из rule-positive строк не может обнаружить false negative,
    поэтому recall по ней принципиально неизмерим. Случайный control — это
    поиск пропущенного fraud среди строк, которые не поднял ни один детектор.
    """
    work = df.copy()
    residual_mask = (
        residual_mask.reindex(work.index, fill_value=False).astype(bool)
        if residual_mask is not None
        else pd.Series(False, index=work.index)
    )
    rule = work["is_suspicious"].eq(1)
    residual = (~rule) & residual_mask
    control = (~rule) & (~residual)

    work["sampling_stratum"] = np.select(
        [rule, residual],
        ["rule_positive", "residual_candidate"],
        default="random_control",
    )
    population = work["sampling_stratum"].value_counts().to_dict()

    # Не предлагаем повторно уже принятые или явно отложенные решения.
    if "verdict" in work.columns:
        verdict = work["verdict"].astype("string").str.strip().str.lower()
        unresolved = ~verdict.isin(["fraud", "legit", "unknown"])
    else:
        unresolved = pd.Series(True, index=work.index)

    pos = work[unresolved & rule].sort_values(
        ["both_detectors_low", "price_z"], ascending=[False, True]
    )
    res = work[unresolved & residual].sort_values("residual_gap", ascending=False)
    controls = work[unresolved & control].copy()
    # Детерминированная псевдослучайная выборка: повторный explore не меняет
    # контроль без изменения набора ad_id.
    controls["_sample_key"] = pd.util.hash_pandas_object(
        controls["ad_id"].astype(str), index=False
    )
    controls = controls.nsmallest(min(control_n, len(controls)), "_sample_key")
    controls = controls.drop(columns="_sample_key")

    q = pd.concat([pos, res, controls], ignore_index=True)
    q["stratum_population"] = q["sampling_stratum"].map(population).astype(int)
    sample_counts = q["sampling_stratum"].value_counts().to_dict()
    q["stratum_sample_size"] = q["sampling_stratum"].map(sample_counts).astype(int)
    return q


def export_labeling_queue(df: pd.DataFrame):
    """Аудируемая очередь для precision, новых кандидатов и поиска FN."""
    work = df.copy()
    work["residual_gap"] = np.nan
    residual_mask = pd.Series(False, index=work.index)
    try:
        from residual_detector import (
            AGE_MAX,
            MIN_SUPPORT,
            load_floor_artifact,
            score_floor,
        )
        from train_price_model import FEATURES

        model, metadata = load_floor_artifact()
        floor = score_floor(model, metadata, work[FEATURES])
        work["residual_gap"] = floor - np.log(work["price_tenge"])
        clean = work[work["is_suspicious"] == 0]
        support = clean.groupby(["brand", "model"]).size()
        sup = pd.Series(
            [int(support.get((b, m), 0)) for b, m in zip(work["brand"], work["model"])],
            index=work.index,
        )
        residual_mask = (
            work["residual_gap"].gt(0)
            & sup.ge(MIN_SUPPORT)
            & work["age"].le(AGE_MAX)
        )
    except FileNotFoundError:
        # Первый run_all до обучения модели: rule + random control всё равно
        # образуют полезную очередь. После residual_detector.py слой добавится.
        pass

    q = select_labeling_rows(work, residual_mask)
    cols = [
        "sampling_stratum", "stratum_population", "stratum_sample_size",
        "ad_id", "url", "title", "year", "price_tenge", "mileage_km",
        "price_z", "residual_gap", "suspicion_reasons",
    ]
    for extra in ["customs_cleared", "steering", "damage_keywords",
                  "seller_comment"]:
        if extra in q.columns:
            cols.append(extra)
    q = q[cols]
    if "seller_comment" in q.columns:
        q["seller_comment"] = q["seller_comment"].fillna("").str[:150]
    q["verdict"] = ""     # ← заполняй: legit / fraud / unknown
    q["comment"] = ""
    out = "data/eda/labeling_queue.csv"
    _write_csv(out, q)
    counts = q["sampling_stratum"].value_counts()
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"Очередь на разметку ({len(q)}: {summary}) → {out}")


def console_report(df: pd.DataFrame):
    line = "─" * 72
    print(line)
    print(f"Объявлений: {len(df)}   Подозрительных: {df.is_suspicious.sum()} "
          f"({df.is_suspicious.mean():.1%})   "
          f"Согласие двух детекторов: {df.both_detectors_low.sum()}")
    print(line)

    print("\n► Причины подозрений:")
    reasons = (df.loc[df.is_suspicious == 1, "suspicion_reasons"]
                 .str.split("|").explode().value_counts())
    print(reasons.to_string())

    if "info_flags" in df.columns:
        info = (df["info_flags"].fillna("").replace("", np.nan).dropna()
                  .str.split("|").explode().value_counts())
        if len(info):
            print("\n► Информационные пометки (оправдания и пр.):")
            print(info.to_string())

    if "customs_cleared" in df.columns:
        enr_n = df["customs_cleared"].notna().sum()
        print(f"\n► Обогащение: {enr_n}/{len(df)} "
              f"({enr_n/len(df):.0%}); растаможка «Нет»: "
              f"{df['customs_cleared'].eq('Нет').sum()}, "
              f"правый руль: {df.get('steering', pd.Series()).eq('Справа').sum()}, "
              f"damage-слова: {(df.get('damage_keywords', pd.Series()).fillna('') != '').sum()}")

    print("\n► ТОП-15 самых аномально дешёвых (сортировка по price_z):")
    cols = ["ad_id", "title", "year", "price_tenge", "mileage_km",
            "price_z", "z_group_level", "iqr_outlier", "views_count", "url"]
    susp = (df[df.is_suspicious == 1]
              .sort_values("price_z")
              .head(15))
    print(susp[cols].to_string(index=False))

    print("\n► Медианная цена по возрастным корзинам:")
    med = df.groupby("age_bucket", observed=True)["price_tenge"] \
            .agg(["count", "median"]).astype(int)
    print(med.to_string())
    print(line)


# ─── Дашборд ──────────────────────────────────────────────────────────────────
def build_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Kolesa.kz · Алматы — качество данных и аномалии",
                 fontsize=15, fontweight="bold", y=0.99)

    ok  = df[df.is_suspicious == 0]
    bad = df[df.is_suspicious == 1]

    # 1) Распределение цен (log-ось): видно логнормальность
    ax = axes[0, 0]
    ax.hist(df.price_tenge, bins=np.logspace(
        np.log10(df.price_tenge.min()), np.log10(df.price_tenge.max()), 40),
        color=C_OK, alpha=.85)
    ax.set_xscale("log")
    ax.axvline(df.price_tenge.median(), color=C_ACC, ls="--", lw=1.5,
               label=f"медиана {df.price_tenge.median()/1e6:.1f} млн")
    ax.axvline(df.price_tenge.mean(), color=C_BAD, ls="--", lw=1.5,
               label=f"среднее {df.price_tenge.mean()/1e6:.1f} млн")
    ax.set_title("Цены: log-шкала (среднее > медианы → правый хвост)")
    ax.set_xlabel("цена, ₸"); ax.legend(fontsize=8)

    # 2) Boxplot по возрасту — это IQR, нарисованный руками:
    #    ящик = Q1..Q3, черта в ящике = медиана, усы = заборы Тьюки,
    #    точки за усами = те самые IQR-выбросы.
    ax = axes[0, 1]
    order = ["0-3", "4-7", "8-12", "13-20", "21+"]
    data = [df.loc[df.age_bucket == b, "log_price"].dropna() for b in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True,
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=C_BAD, alpha=.6))
    for box in bp["boxes"]:
        box.set(facecolor=C_OK, alpha=.55)
    for med_l in bp["medians"]:
        med_l.set(color=C_ACC, lw=2)
    ax.set_title("log(цена) по возрасту: ящик = IQR, точки = выбросы")
    ax.set_xlabel("возраст, лет"); ax.set_ylabel("ln(цена)")

    # 3) Год × цена: подозрительные — красным. Главный «детективный» график
    ax = axes[0, 2]
    ax.scatter(ok.year, ok.price_tenge, s=12, alpha=.45, color=C_OK,
               label="чистые")
    ax.scatter(bad.year, bad.price_tenge, s=34, alpha=.95, color=C_BAD,
               marker="x", label="подозрительные")
    agree = df[df.both_detectors_low == 1]
    ax.scatter(agree.year, agree.price_tenge, s=130, facecolors="none",
               edgecolors=C_ACC, lw=1.6, label="оба детектора")
    ax.set_yscale("log")
    ax.set_title("Год выпуска × цена")
    ax.set_xlabel("год"); ax.set_ylabel("цена, ₸ (log)"); ax.legend(fontsize=8)

    # 4) Причины флагов
    ax = axes[1, 0]
    reasons = (df.loc[df.is_suspicious == 1, "suspicion_reasons"]
                 .str.split("|").explode().value_counts())
    ax.barh(reasons.index[::-1], reasons.values[::-1], color=C_BAD, alpha=.85)
    ax.set_title("Причины подозрений")

    # 5) Пропуски по колонкам
    ax = axes[1, 1]
    na = (df[["mileage_km", "description", "body_type", "condition",
              "labels", "engine_volume"]].isna().mean() * 100).sort_values()
    ax.barh(na.index, na.values, color=C_INFO, alpha=.85)
    ax.set_title("Пропуски, % (mileage — MNAR!)")
    ax.set_xlabel("%")

    # 6) Пробег × цена: аномалии часто «дешёвые при малом пробеге»
    ax = axes[1, 2]
    okm, badm = ok.dropna(subset=["mileage_km"]), bad.dropna(subset=["mileage_km"])
    ax.scatter(okm.mileage_km, okm.price_tenge, s=12, alpha=.4, color=C_OK)
    ax.scatter(badm.mileage_km, badm.price_tenge, s=34, alpha=.95,
               color=C_BAD, marker="x")
    ax.set_xscale("symlog"); ax.set_yscale("log")
    ax.set_title("Пробег × цена (подозрительные — ✕)")
    ax.set_xlabel("пробег, км"); ax.set_ylabel("цена, ₸")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nДашборд → {OUT_PNG}")


def main():
    df = pd.read_sql("SELECT * FROM clean_data", get_engine())
    # Postgres не хранит pandas Categorical — round-trip даёт обычный
    # object/string, и группировки ниже сортируются алфавитно вместо
    # возрастного порядка (0-3, 4-7, ... вместо 0-3, 13-20, 21+, 4-7, ...).
    # Восстанавливаем порядок, который раньше давал CSV с dtype="category".
    age_order = ["0-3", "4-7", "8-12", "13-20", "21+"]
    df["age_bucket"] = pd.Categorical(df["age_bucket"], categories=age_order, ordered=True)
    df = add_iqr_flags(df)
    console_report(df)

    # Полный список флагов, отсортированный «худшие сверху»:
    # сначала согласие двух детекторов, внутри — по price_z (самые дешёвые)
    susp = (df[df.is_suspicious == 1]
              .sort_values(["both_detectors_low", "price_z"],
                           ascending=[False, True]))
    _write_csv(OUT_SUSP, susp)
    print(f"Флаги (отсортированы) → {OUT_SUSP}")

    export_labeling_queue(df)

    build_dashboard(df)


if __name__ == "__main__":
    main()
