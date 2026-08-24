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

MODEL_DIR = Path("data/models")
LOWER_PATH = MODEL_DIR / "price_interval_lower.cbm"
UPPER_PATH = MODEL_DIR / "price_interval_upper.cbm"
META_PATH = MODEL_DIR / "price_interval.metadata.json"
SCHEMA_VERSION = 1


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
    """Финальные границы + поправка, измеренная на out-of-fold данных."""
    lo_oof, hi_oof = oof_bounds(clean, target)
    raw = coverage_report(clean["log_price"], lo_oof, hi_oof, clean["price_tenge"])
    log(f"До калибровки:  покрытие {raw['coverage']*100:.1f}% "
        f"(цель {target*100:.0f}%), медианная ширина {raw['median_width_pct']:.0f}%")

    offset = conformal_offset(
        conformity(clean["log_price"], lo_oof, hi_oof), target)
    fixed = coverage_report(clean["log_price"], lo_oof - offset,
                            hi_oof + offset, clean["price_tenge"])
    log(f"После:          покрытие {fixed['coverage']*100:.1f}%, "
        f"медианная ширина {fixed['median_width_pct']:.0f}%")
    log(f"Поправка к границам: ±{offset:.4f} в логарифме "
        f"(множитель ×{np.exp(offset):.3f})")

    lo_a, hi_a = quantile_levels(target)
    pool = Pool(clean[FEATURES], clean["log_price"], cat_features=CAT_FEATURES)
    final_lo = new_model(loss_function=f"Quantile:alpha={lo_a}")
    final_lo.fit(pool)
    final_hi = new_model(loss_function=f"Quantile:alpha={hi_a}")
    final_hi.fit(pool)
    return final_lo, final_hi, offset, (lo_oof - offset, hi_oof + offset), fixed


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
    off = float(meta["conformal_offset_log"])
    low = np.exp(lo.predict(prepared) - off)
    high = np.exp(hi.predict(prepared) + off)
    return np.minimum(low, high), np.maximum(low, high)


def by_segment(clean: pd.DataFrame, lo_log, hi_log, log=print) -> dict:
    """Покрытие по ценовым сегментам.

    Среднее покрытие может держаться за счёт того, что интервал слишком
    широк на дорогих машинах и слишком узок на дешёвых. Именно дешёвый
    сегмент даёт основную ошибку модели, поэтому смотреть надо раздельно.
    """
    price = clean["price_tenge"].to_numpy(dtype=float)
    buckets = pd.cut(price, [0, 5e6, 10e6, 20e6, np.inf],
                     labels=["<5M", "5-10M", "10-20M", "20M+"])
    out = {}
    log("\nПокрытие по сегментам (цель "
        f"{TARGET_COVERAGE*100:.0f}%, ширина — медианная):")
    for name in buckets.categories:
        m = np.asarray(buckets == name)
        if m.sum() < 20:
            continue
        r = coverage_report(clean["log_price"].to_numpy()[m], lo_log[m],
                            hi_log[m], price[m])
        out[str(name)] = {"n": int(m.sum()), **r}
        log(f"  {name:<7} n={m.sum():<5} покрытие {r['coverage']*100:5.1f}%   "
            f"ширина {r['median_width_pct']:5.0f}%   "
            f"ниже {r['below']*100:4.1f}%  выше {r['above']*100:4.1f}%")
    return out


def main():
    clean = prepare_training_data(load()).reset_index(drop=True)
    print(f"Строк для калибровки: {len(clean)}   "
          f"цель покрытия: {TARGET_COVERAGE*100:.0f}%\n")

    lower, upper, offset, (lo_cal, hi_cal), overall = fit(clean)
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
        "calibration": "conformalized_quantile_regression_grouped_oof",
        "calibration_rows": int(len(clean)),
        "conformal_offset_log": offset,
        "crossed_bounds": crossed,
        "oof": overall,
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
