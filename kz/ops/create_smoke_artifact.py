# -*- coding: utf-8 -*-
"""Создать крошечный НЕПРОДУКТОВЫЙ артефакт для Docker smoke-test.

Настоящая модель и данные не публикуются. Но без файлов модели Docker не
может даже проверить, что контейнер стартует и отвечает на ``/api/health``.
Этот модуль обучает несколько деревьев на синтетических строках и пишет их
только в явно переданный каталог. Использовать результат для оценки машин
нельзя.

Запуск в CI::

    python -m kz.ops.create_smoke_artifact --output .ci-smoke/models
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from kz.ml.train_price_model import (
    ARTIFACT_SCHEMA_VERSION,
    CAT_FEATURES,
    CHEAP_ROUTE_MAX,
    FEATURES,
)


def create(output: Path) -> None:
    """Записать совместимые main/specialist/metadata в безопасный каталог."""
    resolved = output.resolve()
    production = (Path.cwd() / "data/models").resolve()
    if resolved == production:
        raise ValueError("smoke-артефакт не должен перезаписывать data/models")

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    prices = []
    brands = ("Toyota", "Hyundai", "Kia")
    bodies = ("седан", "кроссовер", "хэтчбек")
    for i in range(18):
        rows.append({
            "age": 2 + i,
            "mileage_km": 12_000 + i * 9_000,
            "engine_volume": 1.6 + (i % 4) * 0.4,
            "photos_count": 5 + i % 7,
            "is_mileage_missing": 0,
            "is_vip": i % 2,
            "has_monthly_price": (i + 1) % 2,
            "brand": brands[i % len(brands)],
            "model": f"Smoke-{i % 6}",
            "engine_type": "бензин" if i % 3 else "дизель",
            "transmission": "автомат" if i % 2 else "механика",
            "body_type": bodies[i % len(bodies)],
            "condition": "б/у",
        })
        prices.append(3_000_000 + i * 550_000)

    frame = pd.DataFrame(rows)[FEATURES]
    target = np.log(np.asarray(prices, dtype=float))
    params = dict(iterations=6, depth=3, learning_rate=0.1,
                  loss_function="RMSE", random_seed=42, verbose=False)
    main = CatBoostRegressor(**params)
    specialist = CatBoostRegressor(**params)
    pool = Pool(frame, target, cat_features=CAT_FEATURES)
    main.fit(pool)
    specialist.fit(pool)
    main.save_model(str(output / "price_model.cbm"))
    specialist.save_model(str(output / "price_cheap_specialist.cbm"))

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(frame),
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "target": "log(first_seen_listing_price_tenge)",
        "routing": {"route_below_tenge": CHEAP_ROUTE_MAX},
        "validation": {"grouped_cv": {"model": {"mape_pct": None}}},
        "artifact_purpose": "ci_smoke_test_only",
    }
    (output / "price_model.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create(args.output)


if __name__ == "__main__":
    main()
