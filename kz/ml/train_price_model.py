# -*- coding: utf-8 -*-
"""
Обучение и строгая проверка модели справедливой цены.

Здесь разделены три разных результата:
  1. Grouped CV — общая оценка без попадания точных перезаливов в разные фолды.
  2. Out-of-time holdout — проверка на самых новых объявлениях.
  3. Финальный CatBoost на всех чистых данных — версионируемый артефакт для
     predict_price.py и других потребителей.

Метрики CatBoost всегда сравниваются с простым и сильным baseline:
медианой log(price) по brand/model/year с последовательным fallback до
brand/model, brand и общей медианы. Без baseline высокая R² сама по себе
не доказывает пользу ML.

Запуск: python -m kz.ml.train_price_model   (офлайн, только Postgres)
Выход: data/models/price_model.cbm + price_model.metadata.json
       data/eda/price_model_oof.csv — обезличенные OOF-прогнозы для аудита
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from kz.transform import data_quality
from kz.transform.data_quality import iforest_anomaly, scrub_junk_mileage
from kz.core.db import get_engine

NUM_FEATURES = [
    "age", "mileage_km", "engine_volume", "photos_count",
    "is_mileage_missing", "is_vip", "has_monthly_price",
]
CAT_FEATURES = [
    "brand", "model", "engine_type", "transmission",
    "body_type", "condition",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "price_model.cbm"
SPECIALIST_MODEL_PATH = MODEL_DIR / "price_cheap_specialist.cbm"
METADATA_PATH = MODEL_DIR / "price_model.metadata.json"
OOF_DIAGNOSTICS_PATH = Path("data/eda/price_model_oof.csv")
ARTIFACT_SCHEMA_VERSION = 1
RANDOM_SEED = 42

# Специалист отвечает только там, где ОСНОВНАЯ модель предсказала <5 млн,
# но учится на более широкой полосе фактических цен <8 млн. Если учить его
# строго на <5 млн, пограничные машины, ошибочно направленные маршрутизатором,
# оказываются вне train-распределения и портят дорогой сегмент. Честный OOF
# замер каждого среза сохраняется в metadata вместе с bootstrap-интервалом.
CHEAP_ROUTE_MAX = 5_000_000
CHEAP_TRAIN_MAX = 8_000_000
BOOTSTRAP_REPEATS = 1000


def new_model(loss_function: str = "RMSE") -> CatBoostRegressor:
    """Единая фабрика: train, CV и inference не расходятся по параметрам.

    Параметры подобраны замером на grouped CV (2026-08-01), а не взяты из
    статьи. Проверенные варианты и их MAPE на 4334 строках:
        600 итераций, lr 0.05, depth 8   23.77%   (было)
        2000 итераций, lr 0.03, depth 8  22.96%   ← выбрано
        2000 итераций, lr 0.03, depth 6  23.37%
        то же + l2_leaf_reg=10           23.78%
    Больше итераций с меньшим шагом выигрывают около 0.8 п.п. Это меньше
    разброса между фолдами (±1.05 п.п.), поэтому по одному прогону такой
    выигрыш не увидеть — сравнение честное только потому, что все варианты
    считались на ОДНИХ И ТЕХ ЖЕ разбиениях.
    Цена: обучение стало примерно втрое дольше (секунды, не минуты).
    """
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function=loss_function,
        random_seed=RANDOM_SEED,
        verbose=False,
    )


class RoutedPriceModel:
    """Основная модель плюс специалист дешёвого сегмента.

    Маршрут выбирается только по предсказанию основной модели — фактической
    цены новой машины в бою нет. Обёртка сохраняет привычный `.predict()`,
    поэтому отчёты, survival и веб используют одну и ту же логику.
    """

    def __init__(self, main, specialist=None,
                 route_max: float = CHEAP_ROUTE_MAX):
        self.main = main
        self.specialist = specialist
        self.route_max = float(route_max)

    def route_mask(self, X) -> np.ndarray:
        base_log = np.asarray(self.main.predict(X), dtype=float)
        return np.exp(base_log) < self.route_max

    def predict(self, X) -> np.ndarray:
        base = np.asarray(self.main.predict(X), dtype=float)
        if self.specialist is None or len(base) == 0:
            return base
        mask = np.exp(base) < self.route_max
        if mask.any():
            rows = X.loc[mask] if isinstance(X, pd.DataFrame) else X[mask]
            base[mask] = self.specialist.predict(rows)
        return base

    def model_for(self, X):
        """Модель, реально отвечающая за одну строку (нужно для SHAP)."""
        if self.specialist is None:
            return self.main
        mask = self.route_mask(X)
        return self.specialist if len(mask) == 1 and bool(mask[0]) else self.main

    def get_feature_importance(self, *args, **kwargs):
        # Без конкретной строки показываем глобальную важность основной модели.
        # Для локального SHAP веб вызывает model_for(row).
        return self.main.get_feature_importance(*args, **kwargs)


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Единый preprocessing train/inference для схемы CatBoost."""
    out = df.copy()
    for c in NUM_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in CAT_FEATURES:
        out[c] = out[c].astype("string").fillna("NA").astype(str)
    return out


def load() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM clean_data", get_engine())
    df = coerce_features(df)
    df["price_tenge"] = pd.to_numeric(df["price_tenge"], errors="coerce")
    return df


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Одинаковая фильтрация и data-quality перед любым обучением."""
    out = df[df["price_tenge"].notna() & (df["price_tenge"] > 0)].copy()
    out, _ = scrub_junk_mileage(out)
    out["log_price"] = np.log(out["price_tenge"])
    return out[out["is_suspicious"] == 0].copy()


def duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Группа точного перезалива для защиты CV от leakage.

    Цена намеренно не входит в ключ: изменение цены у той же машины не должно
    превращать её в независимый объект. Если содержательного текста нет,
    используем ad_id — иначе массовые машины с круглым пробегом ошибочно
    склеились бы в одну группу.
    """
    text_col = "text_full" if "text_full" in df.columns else "description"
    text = df.get(text_col, pd.Series("", index=df.index)).fillna("").astype(str)
    text = text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    cols = []
    for c in ["brand", "model", "year", "mileage_km", "engine_volume", "body_type", "color"]:
        if c in df.columns:
            cols.append(df[c].fillna("").astype(str).str.lower().str.strip())
    base = cols[0] if cols else pd.Series("", index=df.index)
    for col in cols[1:]:
        base = base + "\x1f" + col

    meaningful = text.str.len() >= 15
    key = base + "\x1f" + text
    ad_id = df.get("ad_id", pd.Series(df.index.astype(str), index=df.index)).astype(str)
    key = key.where(meaningful, "ad:" + ad_id)
    return pd.util.hash_pandas_object(key, index=False).astype(str)


def regression_metrics(y_log, pred_log) -> dict[str, float]:
    """Метрики в log-пространстве и в исходных тенге."""
    actual = np.exp(np.asarray(y_log, dtype=float))
    pred = np.exp(np.asarray(pred_log, dtype=float))
    ape = np.abs(pred - actual) / actual
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "mae_tenge": float(mean_absolute_error(actual, pred)),
        "mape_pct": float(np.mean(ape) * 100),
        "median_ape_pct": float(np.median(ape) * 100),
    }


def grouped_bootstrap_mape_delta(
    df: pd.DataFrame,
    candidate_log: np.ndarray,
    base_log: np.ndarray,
    n_boot: int = BOOTSTRAP_REPEATS,
) -> dict[str, float | int | list[float]]:
    """Парный bootstrap ΔMAPE кандидата против base по группам дублей.

    Точечные −0.1 п.п. могут быть шумом. Ресэмплируем целые группы, чтобы
    перезаливы одной машины не получили независимых лотерейных билетов, и на
    каждом повторе сравниваем обе модели на одних и тех же строках.
    Отрицательная разница означает, что routed-модель лучше.
    """
    actual = df["price_tenge"].to_numpy(dtype=float)
    cand_ape = np.abs(np.exp(candidate_log) - actual) / actual * 100
    base_ape = np.abs(np.exp(base_log) - actual) / actual * 100
    group_codes, _ = pd.factorize(duplicate_groups(df), sort=False)
    n_groups = int(group_codes.max()) + 1
    sizes = np.bincount(group_codes, minlength=n_groups).astype(float)
    delta_sums = np.bincount(
        group_codes, weights=cand_ape - base_ape, minlength=n_groups
    )
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.integers(0, n_groups, n_groups)
        values[i] = delta_sums[sampled].sum() / sizes[sampled].sum()
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "mape_delta_pct_points": float((cand_ape - base_ape).mean()),
        "bootstrap_95_ci": [float(lo), float(hi)],
        "bootstrap_probability_better": float((values < 0).mean()),
        "bootstrap_repeats": int(n_boot),
        "independent_groups": n_groups,
    }


def _baseline_predict(
    train: pd.DataFrame, y_train: pd.Series, test: pd.DataFrame
) -> np.ndarray:
    """Иерархическая медиана, рассчитанная ТОЛЬКО на train."""
    work = train.copy()
    work["_target"] = np.asarray(y_train)
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    tiers = [
        ["brand", "model", "year"],
        ["brand", "model", "age_bucket"],
        ["brand", "model"],
        ["brand"],
    ]
    for keys in tiers:
        if not all(k in work.columns and k in test.columns for k in keys):
            continue
        med = work.groupby(keys, dropna=False)["_target"].median()
        lookup = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else test[keys[0]]
        values = med.reindex(lookup).to_numpy()
        pred = pred.fillna(pd.Series(values, index=test.index))
    return pred.fillna(float(np.median(y_train))).to_numpy()


def grouped_oof_predictions(
    df: pd.DataFrame, n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OOF routed-модель, baseline и основная модель на одних фолдах."""
    groups = duplicate_groups(df)
    n = min(n_splits, groups.nunique())
    if n < 2:
        raise ValueError("Для grouped CV нужно минимум две независимые группы")
    splitter = GroupKFold(n_splits=n)
    base_oof = np.full(len(df), np.nan)
    routed_oof = np.full(len(df), np.nan)
    baseline_oof = np.full(len(df), np.nan)
    X, y = df[FEATURES], df["log_price"]
    for tr, te in splitter.split(X, y, groups):
        model = new_model()
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        base_pred = np.asarray(model.predict(X.iloc[te]), dtype=float)
        base_oof[te] = base_pred

        train_rows = df.iloc[tr]
        cheap = train_rows["price_tenge"].to_numpy(dtype=float) < CHEAP_TRAIN_MAX
        specialist = new_model()
        specialist.fit(Pool(X.iloc[tr][cheap], y.iloc[tr][cheap],
                            cat_features=CAT_FEATURES))
        routed = base_pred.copy()
        route = np.exp(base_pred) < CHEAP_ROUTE_MAX
        if route.any():
            routed[route] = specialist.predict(X.iloc[te][route])
        routed_oof[te] = routed
        baseline_oof[te] = _baseline_predict(df.iloc[tr], y.iloc[tr], df.iloc[te])
    return routed_oof, baseline_oof, base_oof


def temporal_holdout(df: pd.DataFrame, test_fraction: float = 0.2):
    """Индексы train/test для проверки на будущем без пересечения дублей."""
    if "scraped_at" not in df.columns:
        return None
    ts = pd.to_datetime(df["scraped_at"], errors="coerce")
    valid = ts.notna()
    if valid.sum() < 100 or ts[valid].nunique() < 2:
        return None
    ordered = ts[valid].sort_values().index
    cut = max(1, int(len(ordered) * (1 - test_fraction)))
    train_idx, test_idx = ordered[:cut], ordered[cut:]
    test_groups = set(duplicate_groups(df.loc[test_idx]))
    train_groups = duplicate_groups(df.loc[train_idx])
    train_idx = train_idx[~train_groups.isin(test_groups).to_numpy()]
    if len(train_idx) < 50 or len(test_idx) < 20:
        return None
    return train_idx, test_idx


def evaluate_temporal(df: pd.DataFrame) -> dict | None:
    split = temporal_holdout(df)
    if split is None:
        return None
    tr, te = split
    model = new_model()
    model.fit(Pool(df.loc[tr, FEATURES], df.loc[tr, "log_price"],
                   cat_features=CAT_FEATURES))
    base_pred = np.asarray(model.predict(df.loc[te, FEATURES]), dtype=float)
    cheap = df.loc[tr, "price_tenge"].to_numpy(dtype=float) < CHEAP_TRAIN_MAX
    specialist = new_model()
    specialist.fit(Pool(df.loc[tr, FEATURES].iloc[cheap],
                        df.loc[tr, "log_price"].iloc[cheap],
                        cat_features=CAT_FEATURES))
    routed_pred = base_pred.copy()
    route = np.exp(base_pred) < CHEAP_ROUTE_MAX
    if route.any():
        routed_pred[route] = specialist.predict(df.loc[te, FEATURES].iloc[route])
    model_m = regression_metrics(df.loc[te, "log_price"], routed_pred)
    base_model_m = regression_metrics(df.loc[te, "log_price"], base_pred)
    comparison = grouped_bootstrap_mape_delta(
        df.loc[te], routed_pred, base_pred
    )
    baseline = _baseline_predict(df.loc[tr], df.loc[tr, "log_price"], df.loc[te])
    baseline_m = regression_metrics(df.loc[te, "log_price"], baseline)
    return {
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "train_until": str(pd.to_datetime(df.loc[tr, "scraped_at"]).max()),
        "test_from": str(pd.to_datetime(df.loc[te, "scraped_at"]).min()),
        "model": model_m,
        "base_model": base_model_m,
        "routed_vs_base": comparison,
        "route_fraction": float(route.mean()),
        "baseline": baseline_m,
    }


def segment_metrics(df: pd.DataFrame, pred_log: np.ndarray) -> dict[str, dict]:
    """MAPE и размер по ценовым сегментам — средняя не прячет слабые зоны."""
    actual = df["price_tenge"].to_numpy(dtype=float)
    ape = np.abs(np.exp(pred_log) - actual) / actual * 100
    buckets = pd.cut(
        actual,
        bins=[0, 5e6, 10e6, 20e6, np.inf],
        labels=["<5M", "5-10M", "10-20M", "20M+"],
    )
    result = {}
    for name in buckets.categories:
        mask = np.asarray(buckets == name)
        result[str(name)] = {"n": int(mask.sum()), "mape_pct": float(ape[mask].mean())}
    return result


def save_oof_diagnostics(
    df: pd.DataFrame,
    routed_log: np.ndarray,
    base_log: np.ndarray,
    baseline_log: np.ndarray,
) -> None:
    """Сохраняет честные построчные OOF-прогнозы для сегментного аудита.

    Повторно обучать десять CatBoost-моделей только ради вопроса «где именно
    ошибаемся?» дорого и создаёт риск сравнить разные разбиения. Поэтому
    сохраняем прогнозы того же grouped CV, из которого попала MAPE в metadata.
    Тексты, URL и фотографии сюда не входят; артефакт остаётся локальным в
    gitignored ``data/``.
    """
    actual = df["price_tenge"].to_numpy(dtype=float)
    report = pd.DataFrame({
        "duplicate_group": duplicate_groups(df).astype(str).to_numpy(),
        "age": pd.to_numeric(df["age"], errors="coerce").to_numpy(),
        "actual_price_tenge": actual,
        "routed_pred_tenge": np.exp(np.asarray(routed_log, dtype=float)),
        "base_pred_tenge": np.exp(np.asarray(base_log, dtype=float)),
        "baseline_pred_tenge": np.exp(np.asarray(baseline_log, dtype=float)),
    })
    report["absolute_percentage_error_pct"] = (
        np.abs(report["routed_pred_tenge"] - actual) / actual * 100
    )

    OOF_DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OOF_DIAGNOSTICS_PATH.with_suffix(".csv.tmp")
    report.to_csv(tmp, index=False)
    os.replace(tmp, OOF_DIAGNOSTICS_PATH)


def _data_fingerprint(df: pd.DataFrame) -> str:
    cols = [c for c in ["ad_id", "scraped_at", "price_tenge"] if c in df.columns]
    stable = df[cols].astype(str).sort_values(cols).to_csv(index=False)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def code_fingerprint(*paths: str) -> str:
    """Хэш фактического кода, важный при обучении из dirty worktree.

    В хэш идёт только ИМЯ файла, без пути: путь зависит от раскладки проекта
    и машины, а отпечаток должен зависеть исключительно от самого кода.
    Раньше здесь были относительные пути-строки, и переезд файлов в пакет
    ронял обучение с FileNotFoundError — вызывающие теперь передают __file__.
    """
    digest = hashlib.sha256()
    for name in sorted(paths, key=lambda p: Path(p).name):
        digest.update(Path(name).name.encode("utf-8"))
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def _save_model_atomic(model: CatBoostRegressor, path: Path, prefix: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".cbm", dir=MODEL_DIR)
    os.close(fd)
    tmp_model = Path(tmp_name)
    try:
        model.save_model(str(tmp_model))
        os.replace(tmp_model, path)
    finally:
        tmp_model.unlink(missing_ok=True)


def save_artifact(model: CatBoostRegressor, specialist: CatBoostRegressor,
                  metadata: dict) -> None:
    """Атомарно публикует модель и метаданные: потребитель не увидит полфайла."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _save_model_atomic(model, MODEL_PATH, "price_model.")
    _save_model_atomic(specialist, SPECIALIST_MODEL_PATH, "price_cheap.")

    tmp_meta = METADATA_PATH.with_suffix(".json.tmp")
    tmp_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_meta, METADATA_PATH)


def load_artifact() -> tuple[RoutedPriceModel, dict]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Нет обученного артефакта {MODEL_PATH}. Сначала: python -m kz.ml.train_price_model"
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Несовместимая версия артефакта модели")
    if metadata.get("features") != FEATURES:
        raise ValueError("Схема признаков артефакта не совпадает с текущим кодом")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    specialist = None
    routing = metadata.get("routing")
    if routing:
        if not SPECIALIST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Метаданные требуют специалиста, но нет {SPECIALIST_MODEL_PATH}"
            )
        specialist = CatBoostRegressor()
        specialist.load_model(str(SPECIALIST_MODEL_PATH))
    return RoutedPriceModel(
        model,
        specialist,
        route_max=(routing or {}).get("route_below_tenge", CHEAP_ROUTE_MAX),
    ), metadata


def main():
    raw = load()
    valid = raw[raw["price_tenge"].notna() & (raw["price_tenge"] > 0)].copy()
    valid, n_junk = scrub_junk_mileage(valid)
    valid["log_price"] = np.log(valid["price_tenge"])
    if n_junk:
        print(f"Data-quality: занулено {n_junk} плейсхолдер-пробегов")
    dq = iforest_anomaly(valid)
    print(f"Data-quality: iForest пометил {int(dq.sum())} строк для ручного ревью")

    clean = valid[valid["is_suspicious"] == 0].copy()
    model_oof, baseline_oof, base_oof = grouped_oof_predictions(clean)
    grouped_model = regression_metrics(clean["log_price"], model_oof)
    grouped_base_model = regression_metrics(clean["log_price"], base_oof)
    grouped_base = regression_metrics(clean["log_price"], baseline_oof)
    comparison = grouped_bootstrap_mape_delta(clean, model_oof, base_oof)
    lift = grouped_base["mape_pct"] - grouped_model["mape_pct"]

    print(f"\nGrouped 5-fold CV без leakage дублей ({len(clean)} машин):")
    print(f"  Основная: R²(log)={grouped_base_model['r2_log']:.3f}  "
          f"MAE={grouped_base_model['mae_tenge']/1e6:.2f}М ₸  "
          f"MAPE={grouped_base_model['mape_pct']:.2f}%")
    print(f"  + специалист <5M: R²(log)={grouped_model['r2_log']:.3f}  "
          f"MAE={grouped_model['mae_tenge']/1e6:.2f}М ₸  "
          f"MAPE={grouped_model['mape_pct']:.2f}%")
    print(f"  Baseline: R²(log)={grouped_base['r2_log']:.3f}  "
          f"MAE={grouped_base['mae_tenge']/1e6:.2f}М ₸  "
          f"MAPE={grouped_base['mape_pct']:.1f}%")
    print(f"  Выигрыш CatBoost по MAPE: {lift:+.1f} п.п.")
    ci = comparison["bootstrap_95_ci"]
    print(f"  Routed − основная: {comparison['mape_delta_pct_points']:+.2f} п.п.  "
          f"95% bootstrap ДИ [{ci[0]:+.2f}; {ci[1]:+.2f}]")

    temporal = evaluate_temporal(clean)
    if temporal:
        tm, tm_base, tb = (temporal["model"], temporal["base_model"],
                           temporal["baseline"])
        print(f"\nOut-of-time: train={temporal['train_rows']}, test={temporal['test_rows']}")
        print(f"  Основная MAPE={tm_base['mape_pct']:.1f}%  "
              f"R²(log)={tm_base['r2_log']:.3f}")
        print(f"  + специалист MAPE={tm['mape_pct']:.1f}%  "
              f"R²(log)={tm['r2_log']:.3f}")
        tci = temporal["routed_vs_base"]["bootstrap_95_ci"]
        print("  Routed − основная: "
              f"{temporal['routed_vs_base']['mape_delta_pct_points']:+.2f} п.п.  "
              f"95% bootstrap ДИ [{tci[0]:+.2f}; {tci[1]:+.2f}]")
        print(f"  Baseline MAPE={tb['mape_pct']:.1f}%  R²(log)={tb['r2_log']:.3f}")
    else:
        print("\nOut-of-time: пока недостаточно временного диапазона")

    segments = segment_metrics(clean, model_oof)
    print("\nGrouped-CV MAPE по цене:")
    for name, metric in segments.items():
        print(f"  {name:<7} n={metric['n']:<4} MAPE={metric['mape_pct']:.1f}%")

    final = new_model()
    final.fit(Pool(clean[FEATURES], clean["log_price"], cat_features=CAT_FEATURES))
    cheap = clean["price_tenge"].to_numpy(dtype=float) < CHEAP_TRAIN_MAX
    specialist = new_model()
    specialist.fit(Pool(clean.loc[cheap, FEATURES], clean.loc[cheap, "log_price"],
                        cat_features=CAT_FEATURES))
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "training_code_sha256": code_fingerprint(
            __file__, data_quality.__file__
        ),
        "data_fingerprint_sha256": _data_fingerprint(clean),
        "training_rows": int(len(clean)),
        "routing": {
            "route_below_tenge": CHEAP_ROUTE_MAX,
            "specialist_train_below_tenge": CHEAP_TRAIN_MAX,
            "specialist_training_rows": int(cheap.sum()),
            "artifact": str(SPECIALIST_MODEL_PATH),
        },
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        # Словарь категорий сохраняется не для модели (CatBoost справляется
        # сам), а чтобы было с чем сверить формы интерфейса. Реальный случай:
        # в форме оценки отсутствовал «кроссовер» — второй по частоте кузов,
        # 1549 объявлений. Владелец кроссовера выбирал «внедорожник» и
        # получал оценку по другому классу машин.
        # "NA" сюда не попадает: это сентинел пропуска из coerce_features,
        # а не значение, которое человек мог бы выбрать в форме.
        "categorical_vocabulary": {
            c: sorted(v for v in clean[c].astype(str).value_counts()
                      .loc[lambda s: s >= max(3, len(clean) // 500)].index
                      if v != "NA")
            for c in CAT_FEATURES if c not in ("brand", "model")
        },
        "target": "log(first_seen_listing_price_tenge)",
        "target_policy": {
            "source": "raw_ads.price_tenge",
            "observation": "first_saved_listing_price",
            "later_prices_table": "sightings",
            "is_transaction_price": False,
        },
        "validation": {
            "grouped_cv": {"model": grouped_model,
                           "base_model": grouped_base_model,
                           "baseline": grouped_base,
                           "routed_vs_base": comparison},
            "temporal_holdout": temporal,
            "segments": segments,
        },
        "oof_diagnostics": {
            "path": str(OOF_DIAGNOSTICS_PATH),
            "rows": int(len(clean)),
        },
    }
    save_artifact(final, specialist, metadata)
    save_oof_diagnostics(clean, model_oof, base_oof, baseline_oof)
    print(f"\nАртефакт модели → {MODEL_PATH}")
    print(f"Специалист       → {SPECIALIST_MODEL_PATH}")
    print(f"Метаданные       → {METADATA_PATH}")
    print(f"OOF-диагностика  → {OOF_DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
