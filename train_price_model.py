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

Запуск: python train_price_model.py   (офлайн, только Postgres)
Выход: data/models/price_model.cbm + price_model.metadata.json
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
from sklearn.model_selection import GroupKFold, KFold

from data_quality import iforest_anomaly, scrub_junk_mileage
from db import get_engine

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
METADATA_PATH = MODEL_DIR / "price_model.metadata.json"
ARTIFACT_SCHEMA_VERSION = 1
RANDOM_SEED = 42


def new_model(loss_function: str = "RMSE") -> CatBoostRegressor:
    """Единая фабрика: train, CV и inference не расходятся по параметрам."""
    return CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=8,
        loss_function=loss_function,
        random_seed=RANDOM_SEED,
        verbose=False,
    )


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
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "mae_tenge": float(mean_absolute_error(actual, pred)),
        "mape_pct": float(np.mean(np.abs(pred - actual) / actual) * 100),
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
) -> tuple[np.ndarray, np.ndarray]:
    """OOF-предикты CatBoost и baseline без разделения дублей между фолдами."""
    groups = duplicate_groups(df)
    n = min(n_splits, groups.nunique())
    if n < 2:
        raise ValueError("Для grouped CV нужно минимум две независимые группы")
    splitter = GroupKFold(n_splits=n)
    model_oof = np.full(len(df), np.nan)
    baseline_oof = np.full(len(df), np.nan)
    X, y = df[FEATURES], df["log_price"]
    for tr, te in splitter.split(X, y, groups):
        model = new_model()
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        model_oof[te] = model.predict(X.iloc[te])
        baseline_oof[te] = _baseline_predict(df.iloc[tr], y.iloc[tr], df.iloc[te])
    return model_oof, baseline_oof


def cross_validate(X, y, n_splits=5, groups=None):
    """Совместимый helper: массив [R²(log), MAE(₸), MAPE(%)] по фолдам.

    Новый production-путь передаёт groups. Без groups оставлен обычный KFold
    для обратной совместимости небольших исследовательских вызовов.
    """
    splitter = (
        GroupKFold(n_splits=n_splits)
        if groups is not None
        else KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    )
    out = []
    split_args = (X, y, groups) if groups is not None else (X, y)
    for tr, te in splitter.split(*split_args):
        model = new_model()
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES))
        m = regression_metrics(y.iloc[te], model.predict(X.iloc[te]))
        out.append([m["r2_log"], m["mae_tenge"], m["mape_pct"]])
    return np.asarray(out)


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
    model_m = regression_metrics(
        df.loc[te, "log_price"], model.predict(df.loc[te, FEATURES])
    )
    baseline = _baseline_predict(df.loc[tr], df.loc[tr, "log_price"], df.loc[te])
    base_m = regression_metrics(df.loc[te, "log_price"], baseline)
    return {
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "train_until": str(pd.to_datetime(df.loc[tr, "scraped_at"]).max()),
        "test_from": str(pd.to_datetime(df.loc[te, "scraped_at"]).min()),
        "model": model_m,
        "baseline": base_m,
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
    """Хэш фактического кода, важный при обучении из dirty worktree."""
    digest = hashlib.sha256()
    for name in sorted(paths):
        path = Path(name)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_artifact(model: CatBoostRegressor, metadata: dict) -> None:
    """Атомарно публикует модель и метаданные: потребитель не увидит полфайла."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="price_model.", suffix=".cbm", dir=MODEL_DIR)
    os.close(fd)
    tmp_model = Path(tmp_name)
    try:
        model.save_model(str(tmp_model))
        os.replace(tmp_model, MODEL_PATH)
    finally:
        tmp_model.unlink(missing_ok=True)

    tmp_meta = METADATA_PATH.with_suffix(".json.tmp")
    tmp_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_meta, METADATA_PATH)


def load_artifact() -> tuple[CatBoostRegressor, dict]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Нет обученного артефакта {MODEL_PATH}. Сначала: python train_price_model.py"
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Несовместимая версия артефакта модели")
    if metadata.get("features") != FEATURES:
        raise ValueError("Схема признаков артефакта не совпадает с текущим кодом")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model, metadata


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
    model_oof, baseline_oof = grouped_oof_predictions(clean)
    grouped_model = regression_metrics(clean["log_price"], model_oof)
    grouped_base = regression_metrics(clean["log_price"], baseline_oof)
    lift = grouped_base["mape_pct"] - grouped_model["mape_pct"]

    print(f"\nGrouped 5-fold CV без leakage дублей ({len(clean)} машин):")
    print(f"  CatBoost: R²(log)={grouped_model['r2_log']:.3f}  "
          f"MAE={grouped_model['mae_tenge']/1e6:.2f}М ₸  "
          f"MAPE={grouped_model['mape_pct']:.1f}%")
    print(f"  Baseline: R²(log)={grouped_base['r2_log']:.3f}  "
          f"MAE={grouped_base['mae_tenge']/1e6:.2f}М ₸  "
          f"MAPE={grouped_base['mape_pct']:.1f}%")
    print(f"  Выигрыш CatBoost по MAPE: {lift:+.1f} п.п.")

    temporal = evaluate_temporal(clean)
    if temporal:
        tm, tb = temporal["model"], temporal["baseline"]
        print(f"\nOut-of-time: train={temporal['train_rows']}, test={temporal['test_rows']}")
        print(f"  CatBoost MAPE={tm['mape_pct']:.1f}%  R²(log)={tm['r2_log']:.3f}")
        print(f"  Baseline MAPE={tb['mape_pct']:.1f}%  R²(log)={tb['r2_log']:.3f}")
    else:
        print("\nOut-of-time: пока недостаточно временного диапазона")

    segments = segment_metrics(clean, model_oof)
    print("\nGrouped-CV MAPE по цене:")
    for name, metric in segments.items():
        print(f"  {name:<7} n={metric['n']:<4} MAPE={metric['mape_pct']:.1f}%")

    final = new_model()
    final.fit(Pool(clean[FEATURES], clean["log_price"], cat_features=CAT_FEATURES))
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "training_code_sha256": code_fingerprint(
            "train_price_model.py", "data_quality.py"
        ),
        "data_fingerprint_sha256": _data_fingerprint(clean),
        "training_rows": int(len(clean)),
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "target": "log(price_tenge)",
        "validation": {
            "grouped_cv": {"model": grouped_model, "baseline": grouped_base},
            "temporal_holdout": temporal,
            "segments": segments,
        },
    }
    save_artifact(final, metadata)
    print(f"\nАртефакт модели → {MODEL_PATH}")
    print(f"Метаданные       → {METADATA_PATH}")


if __name__ == "__main__":
    main()
