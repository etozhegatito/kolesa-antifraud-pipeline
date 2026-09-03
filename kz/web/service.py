# -*- coding: utf-8 -*-
"""Price-estimation logic for the web interface, independent of HTTP.

The split is intentional: these pure functions are callable from tests and
the console, while FastAPI remains a thin adapter in ``app.py``.

One estimate contains:
  estimate        point estimate of a fair advertised price;
  range_low/high  a range with measured coverage: about eight out of ten
                  validation vehicles fall inside it;
  drivers         SHAP contributions explaining the individual prediction;
  position        the seller's price among comparable live listings;
  warnings        the same anomaly signals used to build the review queue,
                  reframed as useful pre-publication checks for honest sellers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kz.core.db import get_engine
from kz.ml.predict_price import make_row
from kz.ml.train_price_model import CAT_FEATURES, FEATURES, load_artifact

# Use this coarse, fixed range only when the calibrated interval artifact is
# missing, for example immediately after a feature-schema change. The normal
# interval comes from kz/ml/price_interval.py and has measured coverage.
FALLBACK_LOW, FALLBACK_HIGH = 0.88, 1.15

# Minimum sample size required for a comparable-market position.
MIN_SIMILAR = 8

_model = None
_meta = None
_interval = None
_interval_missing = False
_db_warned = False


def query(sql: str, params: dict) -> pd.DataFrame | None:
    """Run an optional database query.

    Core estimation lives entirely in the model artifact. The database only
    provides enhancements such as comparable listings and price position. The
    public container intentionally has no database, so connection failure must
    degrade gracefully instead of turning an estimate into an HTTP 500.

    Log the warning once per process to avoid repeating it on every request.
    """
    global _db_warned
    try:
        return pd.read_sql(sql, get_engine(), params=params)
    except Exception as e:  # noqa: BLE001 — any connection failure
        if not _db_warned:
            print(
                f"[web] database unavailable ({type(e).__name__}); "
                f"estimation works, comparable listings are disabled"
            )
            _db_warned = True
        return None


def get_model():
    """Load the multi-megabyte model artifact once per process."""
    global _model, _meta
    if _model is None:
        _model, _meta = load_artifact()
    return _model, _meta


def get_interval_models():
    """Load the interval artifact when available.

    Point estimation remains valid without it; only the range becomes coarser.
    Cache a missing artifact and log once instead of failing every request.
    """
    global _interval, _interval_missing
    if _interval is None and not _interval_missing:
        try:
            from kz.ml.price_interval import load_artifact as load_interval

            _interval = load_interval()
        except Exception as e:  # noqa: BLE001 — optional artifact
            _interval_missing = True
            print(
                f"[web] interval artifact unavailable ({type(e).__name__}); "
                f"using a coarse fallback range"
            )
    return _interval


def price_range(car: dict, fair: float) -> tuple[float, float, dict]:
    """Return interval bounds and the method used to produce them.

    Method metadata distinguishes measured conformal coverage from a fallback
    range based on average error.
    """
    models = get_interval_models()
    if models is None:
        return (fair * FALLBACK_LOW, fair * FALLBACK_HIGH, {"method": "fallback", "coverage": None})

    from kz.ml.price_interval import predict_interval

    low, high = predict_interval(make_row(**car), models=models)
    meta = models[2]
    return (
        float(low[0]),
        float(high[0]),
        {
            "method": "conformal",
            "coverage": meta.get("oof", {}).get("coverage"),
            "target_coverage": meta.get("target_coverage"),
        },
    )


def estimate_price(car: dict) -> float:
    """Estimate a fair advertised price in tenge from vehicle attributes."""
    model, _ = get_model()
    return float(np.exp(model.predict(make_row(**car))[0]))


def price_drivers(car: dict, top: int = 6) -> list[dict]:
    """Explain which features raised or lowered this vehicle's estimate.

    CatBoost returns SHAP contributions in log-price space. Convert them to
    percentage multipliers because “about 10% higher” is easier to interpret
    than “plus 0.1 log units.”
    """
    from catboost import Pool

    model, _ = get_model()
    row = make_row(**car)
    active = model.model_for(row) if hasattr(model, "model_for") else model
    shap = active.get_feature_importance(Pool(row, cat_features=CAT_FEATURES), type="ShapValues")[0]
    contribs = shap[:-1]  # final element is the expected value
    order = np.argsort(-np.abs(contribs))[:top]
    out = []
    for i in order:
        c = float(contribs[i])
        out.append(
            {
                "feature": FEATURES[i],
                "value": row.iloc[0][FEATURES[i]],
                "effect_pct": (np.exp(c) - 1) * 100,
            }
        )
    return out


def similar_cars(car: dict, limit: int = 5) -> pd.DataFrame:
    """Find live listings with the same make/model and similar age.

    Concrete market examples make a model estimate easier to validate.
    """
    brand, model_name = car.get("brand"), car.get("model")
    age = car.get("age")
    if not brand or not model_name or age is None:
        return pd.DataFrame()
    q = """SELECT ad_id, brand, model, year, price_tenge, mileage_km, age
           FROM clean_data
           WHERE brand = %(b)s AND model = %(m)s AND is_suspicious = 0
             AND price_tenge > 0 AND ABS(age - %(a)s) <= 2
           ORDER BY ABS(age - %(a)s), price_tenge"""
    df = query(q, {"b": brand, "m": model_name, "a": int(age)})
    if df is None:
        return pd.DataFrame()
    df["ad_id"] = df["ad_id"].astype(str)
    return df.head(limit)


def price_position(car: dict, asking_price: float | None) -> dict | None:
    """Place the seller's price among comparable advertised prices.

    This is deliberately not a time-to-sale forecast. The observation history
    is short and right-censored, so the result only says whether the asking
    price is low, central, or high relative to comparable listings.
    """
    if not asking_price or asking_price <= 0:
        return None
    brand, model_name, age = car.get("brand"), car.get("model"), car.get("age")
    if not brand or not model_name or age is None:
        return None
    q = """SELECT price_tenge FROM clean_data
           WHERE brand = %(b)s AND model = %(m)s AND is_suspicious = 0
             AND price_tenge > 0 AND ABS(age - %(a)s) <= 2"""
    prices = query(q, {"b": brand, "m": model_name, "a": int(age)})
    if prices is None or len(prices) < MIN_SIMILAR:
        return None
    p = prices.price_tenge.to_numpy()
    pct = float((p < asking_price).mean() * 100)
    if pct <= 25:
        label = "Below most comparable listings"
    elif pct >= 75:
        label = "Above most comparable listings"
    else:
        label = "Near the middle of the market"
    return {
        "percentile": pct,
        "label": label,
        "n_similar": int(len(p)),
        "p25": float(np.percentile(p, 25)),
        "p75": float(np.percentile(p, 75)),
    }


def listing_warnings(
    car: dict, asking_price: float | None, fair: float, text: str = ""
) -> list[str]:
    """Check a draft listing with the same signals used in anomaly review.

    The purpose is not to accuse a seller. It helps an honest seller notice
    when a listing resembles a bait ad and explain an unusual price upfront.
    """
    out = []
    if asking_price and fair > 0:
        ratio = asking_price / fair
        if ratio < 0.6:
            has_reason = bool(text and text.strip())
            out.append(
                "The asking price is about {:.0f}% below comparable vehicles. ".format(
                    (1 - ratio) * 100
                )
                + (
                    "Explain why: buyers may treat an unexplained low price as bait."
                    if not has_reason
                    else "The description is present; make sure it states the reason explicitly."
                )
            )
        elif ratio > 1.5:
            out.append(
                "The asking price is about {:.0f}% above comparable vehicles; "
                "the listing may take longer to sell.".format((ratio - 1) * 100)
            )

    def num(key, default=0.0):
        """Coerce a public input field to a number without trusting its type."""
        try:
            return float(car.get(key))
        except (TypeError, ValueError):
            return default

    if not num("mileage_km"):
        out.append(
            "Mileage is missing. Listings with mileage receive about "
            "16% more views in this dataset."
        )
    if num("photos_count") < 5:
        out.append(
            "Fewer than five photos. Listings with at least five photos "
            "receive about 77% more views in this dataset."
        )
    if len(text.strip()) < 50:
        out.append(
            "The description is shorter than 50 characters. Listings "
            "with at least 200 characters receive about 36% more views."
        )
    return out


def jsonable(obj):
    """Convert NumPy/Pandas scalars to plain Python values.

    ``json.dumps`` cannot serialize ``int64`` and ``float64`` directly. NaN
    also becomes ``None`` because JSON has no portable NaN representation.
    """
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is not None and pd.isna(obj) is True:
        return None
    return obj


def full_estimate(car: dict, asking_price: float | None = None, text: str = "") -> dict:
    """Return the service's complete estimate for one vehicle."""
    fair = estimate_price(car)
    _, meta = get_model()
    val = meta.get("validation", {}).get("grouped_cv", {}).get("model", {})
    low, high, how = price_range(car, fair)
    return jsonable(
        {
            "fair_price": fair,
            "range_low": low,
            "range_high": high,
            "range_method": how,
            "model_mape_pct": val.get("mape_pct"),
            "trained_rows": meta.get("training_rows"),
            "drivers": price_drivers(car),
            "position": price_position(car, asking_price),
            "warnings": listing_warnings(car, asking_price, fair, text),
            "similar": similar_cars(car).to_dict("records"),
        }
    )
