# -*- coding: utf-8 -*-
"""Сколько объявление проживёт на рынке — анализ выживаемости.

ЗАЧЕМ ОТДЕЛЬНЫЙ МЕТОД, А НЕ ОБЫЧНАЯ РЕГРЕССИЯ. Наивный подход — взять
проданные машины и предсказывать «дней до продажи» — даёт систематически
заниженный ответ. Причина в том, что мы наблюдаем рынок ограниченное время:
объявление, которое продаётся сорок дней, в выборку «проданных» ещё не
попало, а то, что ушло за три дня, попало сразу.

Это называется **правым цензурированием**: для части объектов событие ещё не
случилось, и мы знаем только, что оно случится не раньше, чем через
наблюдённое время. Выбрасывать такие объекты нельзя (потеряем медленные
продажи), приравнивать их к «продалось сегодня» — тоже.

Анализ выживаемости работает именно с такими данными.

  Оценка Каплана-Мейера — доля объявлений, ещё живых через t дней. Каждый
    цензурированный объект участвует ровно до момента, пока наблюдался, и
    дальше корректно исключается из знаменателя.

  Модель Кокса — какие факторы ускоряют уход с рынка. Коэффициент читается
    как отношение рисков: 1,5 означает, что при прочих равных объявление
    уходит в полтора раза быстрее.

ЧЕСТНАЯ ГРАНИЦА. Окно наблюдения около трёх недель, событий 183. Этого
достаточно, чтобы увидеть форму кривой на коротких сроках, и НЕ достаточно,
чтобы говорить о сроках больше трёх недель: там просто нет данных. Все
выводы за пределами окна — экстраполяция, и в отчёте это сказано прямо.

Запуск: python -m kz.ml.survival
Выход:  консоль + data/eda/survival.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kz.core.db import get_engine
from kz.ml.time_to_sell import parse_posted

OUT_PNG = Path("data/eda/survival.png")
HORIZON = 21          # дней: дальше данных почти нет, экстраполировать нечестно
MIN_EVENTS_PER_FEATURE = 10   # общепринятое правило для модели Кокса


def load_survival() -> pd.DataFrame:
    """Таблица «сколько прожило и случилось ли событие».

    Начало отсчёта — дата публикации из карточки, а не дата, когда мы впервые
    увидели объявление: сбор начался позже, чем часть объявлений появилась,
    и от нашего расписания срок жизни зависеть не должен.
    """
    eng = get_engine()
    # Берём все колонки: дальше по этой же таблице считается справедливая цена
    # моделью, а ей нужен полный набор признаков.
    cd = pd.read_sql("SELECT * FROM clean_data", eng, dtype={"ad_id": str})
    # status в clean_data уже есть — из ad_status берём только дату проверки,
    # иначе слияние раздвоило бы колонку в status_x/status_y.
    st = pd.read_sql("SELECT ad_id, checked_at FROM ad_status", eng,
                     dtype={"ad_id": str})
    sg = pd.read_sql("SELECT ad_id, MAX(seen_date) AS last_seen FROM sightings "
                     "GROUP BY ad_id", eng, dtype={"ad_id": str})

    d = cd.merge(st, on="ad_id", how="left").merge(sg, on="ad_id", how="left")
    d["start"] = pd.to_datetime(d["posted_date"].map(parse_posted))
    d["event"] = d["status"].isin(["archived", "deleted"]).astype(int)
    # конец наблюдения: дата проверки для ушедших, последняя встреча для живых
    d["end"] = pd.to_datetime(
        np.where(d["event"] == 1, d["checked_at"], d["last_seen"]))
    d["days"] = (d["end"] - d["start"]).dt.days
    d = d[d["days"].notna() & (d["days"] >= 0) & d["price_tenge"].notna()]
    return d.reset_index(drop=True)


def add_price_position(d: pd.DataFrame) -> pd.DataFrame:
    """Насколько цена отличается от справедливой по модели.

    Это и есть переменная, ради которой всё затевалось: продавец хочет знать,
    ускорит ли скидка продажу.
    """
    from kz.ml.train_price_model import coerce_features, FEATURES, load_artifact

    model, _ = load_artifact()
    X = coerce_features(d.copy())[FEATURES]
    fair = np.exp(model.predict(X))
    d = d.copy()
    d["fair_price"] = fair
    d["price_ratio"] = d["price_tenge"] / fair
    # три группы вместо непрерывной величины: событий мало, и дробить сильнее
    # значит получить доверительные интервалы шире самого эффекта
    d["price_group"] = pd.cut(
        d["price_ratio"], [0, 0.9, 1.1, np.inf],
        labels=["дешевле рынка", "по рынку", "дороже рынка"])
    return d


def kaplan_meier(d: pd.DataFrame, log=print):
    """Кривая выживания: доля объявлений, ещё висящих через t дней."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter()
    km.fit(d["days"], d["event"], label="все объявления")
    log("\nДоля объявлений, ещё не ушедших с рынка:")
    for t in (3, 7, 14, HORIZON):
        s = float(km.survival_function_at_times(t).iloc[0])
        log(f"  через {t:2d} дн. — {s*100:5.1f}%   (то есть ушло {100-s*100:.1f}%)")
    med = km.median_survival_time_
    log(f"\nМедианный срок жизни: "
        + (f"{med:.0f} дн." if np.isfinite(med) else
           "не достигнут — за окно наблюдения ушла меньше половины объявлений"))
    return km


def by_price_group(d: pd.DataFrame, log=print):
    """Продаются ли дешёвые быстрее — главный продуктовый вопрос."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    log("\nУход с рынка по группам цены (доля ушедших к 14-му дню):")
    curves = {}
    for name, g in d.groupby("price_group", observed=True):
        if g["event"].sum() < 5:
            log(f"  {name:16} событий {int(g['event'].sum())} — слишком мало")
            continue
        km = KaplanMeierFitter().fit(g["days"], g["event"], label=str(name))
        s14 = float(km.survival_function_at_times(14).iloc[0])
        log(f"  {str(name):16} n={len(g):5d}  событий={int(g['event'].sum()):4d}  "
            f"ушло к 14 дню {100-s14*100:5.1f}%")
        curves[str(name)] = km

    res = multivariate_logrank_test(d["days"], d["price_group"], d["event"])
    log(f"\nЛогранговый тест различия кривых: p = {res.p_value:.4f}"
        + ("  — различие значимо" if res.p_value < 0.05
           else "  — различие НЕ значимо"))
    return curves


def cox_model(d: pd.DataFrame, log=print):
    """Какие факторы ускоряют уход с рынка."""
    from lifelines import CoxPHFitter

    feats = ["price_ratio", "age", "photos_count", "is_vip"]
    n_events = int(d["event"].sum())
    allowed = n_events // MIN_EVENTS_PER_FEATURE
    if allowed < len(feats):
        feats = feats[:max(1, allowed)]
        log(f"\n(признаков ограничено до {len(feats)}: событий {n_events}, "
            f"правило — не меньше {MIN_EVENTS_PER_FEATURE} событий на признак)")

    work = d[feats + ["days", "event"]].dropna().copy()
    cph = CoxPHFitter().fit(work, duration_col="days", event_col="event")
    log("\nМодель Кокса — отношение рисков (больше 1 = уходит быстрее):")
    s = cph.summary
    for name in s.index:
        hr, p = s.loc[name, "exp(coef)"], s.loc[name, "p"]
        mark = "  ← значимо" if p < 0.05 else ""
        log(f"  {name:16} HR={hr:5.2f}   p={p:.4f}{mark}")
    return cph


def plot(km, curves, path: Path = OUT_PNG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    km.plot_survival_function(ax=ax[0], color="#2563c9")
    ax[0].set_title("Доля объявлений на рынке")
    ax[0].set_xlabel("дней с публикации"); ax[0].set_ylabel("ещё не ушли")
    ax[0].set_xlim(0, HORIZON); ax[0].grid(alpha=.3)
    for label, c in curves.items():
        c.plot_survival_function(ax=ax[1])
    ax[1].set_title("По группам цены")
    ax[1].set_xlabel("дней с публикации"); ax[1].set_xlim(0, HORIZON)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\nГрафик → {path}")


def main():
    d = load_survival()
    print(f"Объявлений: {len(d)}   событий: {int(d.event.sum())}   "
          f"цензурировано: {int((1-d.event).sum())}")
    print(f"Доля событий {d.event.mean()*100:.1f}% — большинство ещё висит, "
          "и именно поэтому нужен анализ выживаемости, а не регрессия.")

    km = kaplan_meier(d)
    d = add_price_position(d)
    curves = by_price_group(d)
    try:
        cox_model(d)
    except Exception as e:                       # noqa: BLE001 — данных может не хватить
        print(f"\nМодель Кокса не сошлась: {e}")
    plot(km, curves)
    print(f"\nГраница честности: окно наблюдения около {int(d.days.max())} дней. "
          f"Выводы о сроках дольше {HORIZON} дней данными не обеспечены.")


if __name__ == "__main__":
    main()
