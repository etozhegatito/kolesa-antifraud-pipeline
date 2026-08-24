# -*- coding: utf-8 -*-
"""Интервал справедливой цены с ИЗМЕРЕННЫМ покрытием.

ЗАЧЕМ ЭТО ВМЕСТО ПОГОНИ ЗА MAPE. Сервис до сих пор отдавал диапазон как
жёсткий коридор ±12/15% вокруг прогноза — одинаковый для новой Camry и для
Delica 1993 года. Для первой он честный, для второй фикция: там модель не
знает цену и с точностью 12%, и с точностью 40%.

Хуже другое. MAPE как главную метрику **выгодно обманывать**: замер показал,
что достаточно умножить все прогнозы на 0,95, и средняя ошибка падает на
полпроцента — потому что exp() несимметричен и занижение ограничено сотней
процентов, а завышение ничем. Продукт при этом портится: шести продавцам из
десяти сервис называет цену ниже запрошенной (см. docs/FINDINGS.md, п. 9).

Покрытие интервала таким фокусом не выиграть. Занизишь прогноз — граница
уедет, и доля попаданий сломается сразу. Метрика, которую нельзя выиграть
сдвигом константы, годится на роль главной, а MAPE — нет.

КАК СЧИТАЕТСЯ. Квантильная регрессия сама по себе не даёт обещанного
покрытия: обучив модель на 10-й процентиль, мы НЕ получаем ровно 10%
наблюдений ниже неё на новых данных. Поэтому берётся приём под названием
**конформизованная квантильная регрессия** (conformalized quantile
regression, CQR):

  1. обучаются две квантильные модели — нижняя и верхняя граница;
  2. на данных, которых модели не видели, считается «мера невязки»

         E = max(нижняя_граница − факт,  факт − верхняя_граница)

     Она положительна ровно тогда, когда факт вылез за интервал, и равна
     тому, насколько вылез;
  3. берётся эмпирический квантиль этих невязок, и обе границы
     раздвигаются на него.

Смысл шага 3: мы не верим номинальным процентилям, а измеряем, насколько
они врут, и компенсируем ровно на измеренную величину. Взамен получается
**гарантия конечной выборки**: доля попаданий на новых данных будет около
целевой, каким бы кривым ни оказался исходный квантильный прогноз.

ЧЕСТНАЯ ОГОВОРКА. Невязки считаются по out-of-fold предсказаниям моделей,
обученных на четырёх пятых данных, а финальные границы обучены на всех.
Финальные модели чуть точнее тех, по которым калибровались, поэтому реальное
покрытие выходит слегка ВЫШЕ целевого — интервал получается консервативным.
Ошибаться в эту сторону здесь правильно: слишком широкий интервал честен,
слишком узкий обманывает.

ШИРИНУ ТОЖЕ НАДО СМОТРЕТЬ. Покрытие 100% даёт и интервал «от нуля до
бесконечности», поэтому рядом с покрытием всегда печатается медианная
относительная ширина. Полезен только узкий интервал с честным покрытием.

Запуск: python -m kz.ml.price_interval
Выход:  data/models/price_interval.cbm(.upper) + метаданные
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from kz.ml import train_price_model as _tpm
from kz.ml.train_price_model import (
    CAT_FEATURES,
    FEATURES,
    code_fingerprint,
    coerce_features,
    duplicate_groups,
    load,
    new_model,
    prepare_training_data,
)

TARGET_COVERAGE = 0.80        # обещаем «8 из 10 машин попадают в интервал»
N_SPLITS = 5

# Границы групп для калибровки, в тенге ПРЕДСКАЗАННОЙ цены.
#
# Почему предсказанной, а не фактической: фактическая — это то, что мы
# предсказываем. Условиться на ней при обучении можно, а при выдаче прогноза
# новой машине уже нет: цены ещё нет. Калибровка, которую нельзя применить,
# бесполезна, поэтому группа определяется по собственному прогнозу модели.
GROUP_EDGES = [0, 5e6, 10e6, 20e6, float("inf")]
GROUP_NAMES = ["<5M", "5-10M", "10-20M", "20M+"]

# Ниже этого числа наблюдений группа калибруется общей поправкой. Своя
# поправка на полусотне строк — это шум, выданный за настройку.
MIN_GROUP = 200

MODEL_DIR = Path("data/models")
LOWER_PATH = MODEL_DIR / "price_interval_lower.cbm"
UPPER_PATH = MODEL_DIR / "price_interval_upper.cbm"
META_PATH = MODEL_DIR / "price_interval.metadata.json"
SCHEMA_VERSION = 2


def quantile_levels(target: float = TARGET_COVERAGE) -> tuple[float, float]:
    """Целевое покрытие → уровни квантилей, симметрично по хвостам.

    Для 0,80 это 0,10 и 0,90: по десять процентов остаётся с каждой стороны.
    """
    tail = (1.0 - target) / 2.0
    return tail, 1.0 - tail


def conformity(y_log, lo_log, hi_log) -> np.ndarray:
    """Мера невязки CQR: насколько факт вылез за интервал.

    Отрицательная, если факт внутри (и тем меньше, чем дальше от границ),
    положительная, если вылез. Один максимум на две стороны — поэтому
    поправка получается общей, а не по хвостам отдельно.
    """
    y = np.asarray(y_log, dtype=float)
    return np.maximum(np.asarray(lo_log, dtype=float) - y,
                      y - np.asarray(hi_log, dtype=float))


def conformal_offset(scores: np.ndarray, target: float = TARGET_COVERAGE) -> float:
    """Поправка к границам: эмпирический квантиль невязок.

    Множитель (n+1)/n — не косметика: он и даёт гарантию конечной выборки.
    Без него обещанное покрытие достигается лишь в пределе большого n, а у
    нас данных немного.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, target * (n + 1) / n)
    return float(np.quantile(scores, level, method="higher"))


def tail_offsets(y_log, lo_log, hi_log,
                 target: float = TARGET_COVERAGE) -> tuple[float, float]:
    """Поправки к нижней и верхней границе ПО ОТДЕЛЬНОСТИ.

    Обычный CQR берёт один максимум на две стороны и раздвигает границы
    одинаково. Это гарантирует общее покрытие, но ничего не говорит о том,
    как промахи распределены по хвостам, — а у нас они распределены криво:
    у дешёвых машин 15,2% вываливались ВНИЗ против 5,9% вверх, у дорогих
    зеркально. То есть интервал систематически стоял высоко на дешёвых и
    низко на дорогих, при формально верных 80%.

    Здесь каждый хвост калибруется своим квантилем: столько же процентов
    слева, сколько справа. Симметричность покрытия — не косметика: продавцу
    дешёвой машины важно, что верхняя граница не завышена, а не то, что
    интервал в среднем правильной ширины.
    """
    y = np.asarray(y_log, dtype=float)
    tail = (1.0 - target) / 2.0            # по 10% на сторону при target=0.8

    def edge(scores: np.ndarray) -> float:
        scores = scores[np.isfinite(scores)]
        n = len(scores)
        if n == 0:
            return 0.0
        level = min(1.0, (1.0 - tail) * (n + 1) / n)
        return float(np.quantile(scores, level, method="higher"))

    return edge(np.asarray(lo_log, dtype=float) - y), edge(y - np.asarray(hi_log, dtype=float))


def group_of(price: np.ndarray) -> np.ndarray:
    """Индекс группы по цене (предсказанной при выдаче, любой при замере)."""
    return np.clip(np.searchsorted(GROUP_EDGES, np.asarray(price, dtype=float),
                                   side="right") - 1,
                   0, len(GROUP_NAMES) - 1)


def group_offsets(y_log, lo_log, hi_log, pred_price,
                  target: float = TARGET_COVERAGE) -> dict:
    """Свои поправки для каждой ценовой группы, с общими как запасными.

    Приём называется мондриановской конформной калибровкой: выборка режется
    на заранее объявленные группы, и гарантия покрытия выполняется ВНУТРИ
    каждой, а не только в среднем по всем.

    Цена этого — меньше данных на каждую оценку. Поэтому группа меньше
    MIN_GROUP считается по общей поправке: лучше честное среднее, чем
    подгонка под полсотни строк.
    """
    g = group_of(pred_price)
    fallback = tail_offsets(y_log, lo_log, hi_log, target)
    out = {"global": list(fallback), "groups": {}}
    y = np.asarray(y_log, dtype=float)
    for i, name in enumerate(GROUP_NAMES):
        m = g == i
        if m.sum() < MIN_GROUP:
            out["groups"][name] = {"offsets": list(fallback), "n": int(m.sum()),
                                   "source": "общая (группа мала)"}
            continue
        out["groups"][name] = {
            "offsets": list(tail_offsets(y[m], np.asarray(lo_log)[m],
                                         np.asarray(hi_log)[m], target)),
            "n": int(m.sum()), "source": "своя"}
    return out


def apply_offsets(lo_log, hi_log, offsets: dict):
    """Раздвинуть границы поправками той группы, в которую попал прогноз.

    Группа берётся по СЫРОЙ середине интервала — она известна до применения
    поправок, поэтому не возникает замкнутого круга «поправка зависит от
    группы, а группа от поправки».
    """
    lo_log = np.asarray(lo_log, dtype=float)
    hi_log = np.asarray(hi_log, dtype=float)
    g = group_of(np.exp((lo_log + hi_log) / 2))
    d_lo = np.empty(len(lo_log))
    d_hi = np.empty(len(hi_log))
    for i, name in enumerate(GROUP_NAMES):
        pair = offsets["groups"].get(name, {}).get("offsets", offsets["global"])
        m = g == i
        d_lo[m], d_hi[m] = pair[0], pair[1]
    return lo_log - d_lo, hi_log + d_hi


def oof_bounds(clean: pd.DataFrame, target: float = TARGET_COVERAGE):
    """Границы, посчитанные по фолдам, которых модель не видела.

    Группы дублей не разрываются между фолдами — иначе перезалив той же
    машины попал бы и в обучение, и в калибровку, невязки вышли бы
    оптимистичными, а интервал — слишком узким.
    """
    lo_a, hi_a = quantile_levels(target)
    groups = duplicate_groups(clean)
    n = min(N_SPLITS, groups.nunique())
    if n < 2:
        raise ValueError("Недостаточно независимых групп для калибровки")

    lo = np.full(len(clean), np.nan)
    hi = np.full(len(clean), np.nan)
    X, y = clean[FEATURES], clean["log_price"]
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES)
        m_lo = new_model(loss_function=f"Quantile:alpha={lo_a}")
        m_lo.fit(pool)
        m_hi = new_model(loss_function=f"Quantile:alpha={hi_a}")
        m_hi.fit(pool)
        lo[te] = m_lo.predict(X.iloc[te])
        hi[te] = m_hi.predict(X.iloc[te])
    return lo, hi


def coverage_report(y_log, lo_log, hi_log, price) -> dict:
    """Попадание и ширина — вместе, потому что по отдельности каждое лжёт."""
    y = np.asarray(y_log, dtype=float)
    inside = (y >= lo_log) & (y <= hi_log)
    low, high = np.exp(lo_log), np.exp(hi_log)
    width = (high - low) / np.asarray(price, dtype=float)
    return {
        "coverage": float(inside.mean()),
        "median_width_pct": float(np.median(width) * 100),
        "mean_width_pct": float(np.mean(width) * 100),
        "below": float((y < lo_log).mean()),
        "above": float((y > hi_log).mean()),
    }


def fit(clean: pd.DataFrame, target: float = TARGET_COVERAGE, log=print):
    """Финальные границы + поправки по группам, измеренные out-of-fold."""
    lo_oof, hi_oof = oof_bounds(clean, target)
    y, price = clean["log_price"], clean["price_tenge"]

    raw = coverage_report(y, lo_oof, hi_oof, price)
    log(f"Без калибровки:      покрытие {raw['coverage']*100:.1f}% "
        f"(цель {target*100:.0f}%), ширина {raw['median_width_pct']:.0f}%")

    # Общая симметричная поправка — то, с чего начинали. Оставлена как база
    # сравнения: без неё непонятно, что дало разделение по группам.
    sym = conformal_offset(conformity(y, lo_oof, hi_oof), target)
    sym_rep = coverage_report(y, lo_oof - sym, hi_oof + sym, price)
    log(f"Общая поправка:      покрытие {sym_rep['coverage']*100:.1f}%, "
        f"ширина {sym_rep['median_width_pct']:.0f}%")

    mid_price = np.exp((lo_oof + hi_oof) / 2)
    offsets = group_offsets(y, lo_oof, hi_oof, mid_price, target)
    lo_cal, hi_cal = apply_offsets(lo_oof, hi_oof, offsets)
    fixed = coverage_report(y, lo_cal, hi_cal, price)
    log(f"По группам:          покрытие {fixed['coverage']*100:.1f}%, "
        f"ширина {fixed['median_width_pct']:.0f}%\n")

    log("Поправки по группам (в логарифме, вниз / вверх):")
    for name, info in offsets["groups"].items():
        d_lo, d_hi = info["offsets"]
        log(f"  {name:<7} n={info['n']:<5} вниз {d_lo:+.3f}  вверх {d_hi:+.3f}"
            f"   {info['source']}")

    lo_a, hi_a = quantile_levels(target)
    pool = Pool(clean[FEATURES], clean["log_price"], cat_features=CAT_FEATURES)
    final_lo = new_model(loss_function=f"Quantile:alpha={lo_a}")
    final_lo.fit(pool)
    final_hi = new_model(loss_function=f"Quantile:alpha={hi_a}")
    final_hi.fit(pool)
    return final_lo, final_hi, offsets, (lo_cal, hi_cal), fixed, sym_rep


def _save_model(model: CatBoostRegressor, path: Path) -> None:
    """Атомарно: потребитель не должен увидеть полфайла."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem, suffix=".cbm",
                                    dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        model.save_model(str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_artifact(lower, upper, metadata: dict) -> None:
    _save_model(lower, LOWER_PATH)
    _save_model(upper, UPPER_PATH)
    tmp = META_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    os.replace(tmp, META_PATH)


def load_artifact():
    if not (LOWER_PATH.exists() and UPPER_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError(
            "Нет артефакта интервала. Сначала: python -m kz.ml.price_interval")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Несовместимая версия артефакта интервала")
    if meta.get("features") != FEATURES:
        raise ValueError("Схема признаков интервала не совпадает с кодом")
    lo, hi = CatBoostRegressor(), CatBoostRegressor()
    lo.load_model(str(LOWER_PATH))
    hi.load_model(str(UPPER_PATH))
    return lo, hi, meta


def predict_interval(X: pd.DataFrame, models=None) -> tuple[np.ndarray, np.ndarray]:
    """Интервал в ТЕНГЕ для готовых строк признаков.

    Нижняя и верхняя границы — ДВЕ НЕЗАВИСИМЫЕ модели, и ничто не мешает им
    поменяться местами на нетипичном сочетании признаков: 10-й процентиль
    предскажет выше 90-го. Это известная болезнь квантильной регрессии,
    называется пересечением квантилей.

    На наших данных это редкость (4 строки из 6718), но выдать пользователю
    диапазон «от 16 млн до 14 млн» нельзя вовсе. Лечится перестановкой:
    меньшее значение объявляется нижней границей. Приём стандартный и
    покрытие не портит — интервал остаётся тем же отрезком, просто с
    правильно названными концами.
    """
    lo, hi, meta = models or load_artifact()
    prepared = coerce_features(X)[FEATURES]
    lo_log, hi_log = apply_offsets(lo.predict(prepared), hi.predict(prepared),
                                   meta["offsets"])
    low, high = np.exp(lo_log), np.exp(hi_log)
    return np.minimum(low, high), np.maximum(low, high)


def by_segment(clean: pd.DataFrame, lo_log, hi_log, log=print) -> dict:
    """Покрытие по сегментам — в ДВУХ разрезах, и разница между ними важна.

    По ПРЕДСКАЗАННОЙ цене — то, на чём калибровка обусловлена, и то, что
    единственно доступно при выдаче прогноза новой машине. Здесь покрытие и
    хвосты обязаны быть ровными: если нет, калибровка сломана.

    По ФАКТИЧЕСКОЙ цене — разрез, в котором остаётся перекос: у дешёвых
    машин факт чаще вываливается ВНИЗ, у дорогих ВВЕРХ. Долго принимал это
    за дефект калибровки. Это не дефект и устранить его нельзя.

    Причина в том, что группировка по факту — это обусловливание на том, что
    мы предсказываем. Машины, чья настоящая цена низка, — по построению те,
    которые модель переоценила (прогноз тянется к среднему). Никакая
    калибровка по признакам этого не снимет: в момент прогноза неизвестно, с
    какой стороны от своей ошибки окажется машина.

    Показываем оба разреза именно поэтому. Первый доказывает, что калибровка
    работает. Второй честно сообщает продавцу дешёвой машины: если она
    действительно дёшева, наш интервал скорее стоит выше неё.
    """
    price = clean["price_tenge"].to_numpy(dtype=float)
    y = clean["log_price"].to_numpy()
    predicted = np.exp((np.asarray(lo_log) + np.asarray(hi_log)) / 2)

    out = {}
    for label, key, tag in [("по предсказанной цене", predicted, "predicted"),
                            ("по фактической цене", price, "actual")]:
        log(f"\nПокрытие {label} (цель {TARGET_COVERAGE*100:.0f}%):")
        g = group_of(key)
        out[tag] = {}
        for i, name in enumerate(GROUP_NAMES):
            m = g == i
            if m.sum() < 20:
                continue
            r = coverage_report(y[m], np.asarray(lo_log)[m],
                                np.asarray(hi_log)[m], price[m])
            out[tag][name] = {"n": int(m.sum()), **r}
            skew = (r["below"] - r["above"]) * 100
            log(f"  {name:<7} n={m.sum():<5} покрытие {r['coverage']*100:5.1f}%   "
                f"ширина {r['median_width_pct']:5.0f}%   "
                f"ниже {r['below']*100:4.1f}%  выше {r['above']*100:4.1f}%"
                f"   перекос {skew:+5.1f}")
    log("\n  Ровные хвосты в первом разрезе — признак рабочей калибровки.")
    log("  Перекос во втором неустраним: группировка по факту обусловливает")
    log("  на том, что мы предсказываем (см. докстринг by_segment).")
    return out


def main():
    clean = prepare_training_data(load()).reset_index(drop=True)
    print(f"Строк для калибровки: {len(clean)}   "
          f"цель покрытия: {TARGET_COVERAGE*100:.0f}%\n")

    lower, upper, offsets, (lo_cal, hi_cal), overall, sym = fit(clean)
    segments = by_segment(clean, lo_cal, hi_cal)

    # Пересечение квантилей — редкость, но молчать о нём нельзя: если доля
    # вырастет, значит модели границ разъезжаются и интервалу нельзя верить.
    crossed = int((lo_cal > hi_cal).sum())
    print(f"\nПересечений границ (нижняя выше верхней): {crossed} из "
          f"{len(clean)} ({crossed/len(clean)*100:.2f}%). При выдаче концы "
          f"переставляются местами.")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_sha256": code_fingerprint(__file__, _tpm.__file__),
        "features": FEATURES,
        "target_coverage": TARGET_COVERAGE,
        "quantile_levels": list(quantile_levels()),
        "calibration": "mondrian_cqr_asymmetric_tails_grouped_oof",
        "calibration_rows": int(len(clean)),
        "group_edges": GROUP_EDGES[1:-1],
        "min_group": MIN_GROUP,
        "offsets": offsets,
        "crossed_bounds": crossed,
        "oof": overall,
        "oof_symmetric_global": sym,   # база сравнения: что было до групп
        "segments": segments,
    }
    save_artifact(lower, upper, metadata)

    print(f"\nИтог: интервал накрывает {overall['coverage']*100:.1f}% машин "
          f"при обещанных {TARGET_COVERAGE*100:.0f}%, "
          f"медианная ширина {overall['median_width_pct']:.0f}% от цены.")
    print("Покрытие нельзя выиграть сдвигом прогноза — в этом и смысл замены "
          "MAPE на него как на главную метрику.")
    print(f"\nАртефакты → {LOWER_PATH.name}, {UPPER_PATH.name}, {META_PATH.name}")


if __name__ == "__main__":
    main()
