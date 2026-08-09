# -*- coding: utf-8 -*-
"""Логика оценки для веб-интерфейса, без единой строчки HTTP.

Разделение намеренное: здесь чистые функции, которые можно вызвать из тестов
и из консоли, а FastAPI в app.py остаётся тонкой обёрткой. Иначе проверить
логику можно было бы только подняв сервер.

Что отдаётся по одной машине:
  estimate        точечная оценка справедливой цены;
  range_low/high  диапазон, а не одно число: точность модели ±23%, и делать
                  вид, что мы знаем цену до тенге, — обман;
  drivers         разложение прогноза по вкладам признаков (SHAP), чтобы
                  человек видел, ПОЧЕМУ столько, а не верил чёрному ящику;
  position        позиция цены продавца среди похожих машин;
  warnings        предупреждения того же детектора, что и в антифроде —
                  честному продавцу полезно узнать, что его объявление
                  выглядит как приманка, ДО публикации.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kz.core.db import get_engine
from kz.ml.predict_price import make_row
from kz.ml.train_price_model import CAT_FEATURES, FEATURES, load_artifact

# Диапазон вокруг точечной оценки. Берём из фактической ошибки модели на
# кросс-валидации, а не из головы: половина машин предсказывается точнее
# медианного APE, поэтому такой коридор честно отражает типичный промах.
RANGE_LOW, RANGE_HIGH = 0.88, 1.15

# Минимум похожих машин, чтобы говорить о позиции среди конкурентов.
MIN_SIMILAR = 8

_model = None
_meta = None


def get_model():
    """Артефакт грузится один раз на процесс: он весит мегабайты, а запросов
    к сервису много."""
    global _model, _meta
    if _model is None:
        _model, _meta = load_artifact()
    return _model, _meta


def estimate_price(car: dict) -> float:
    """Справедливая цена в тенге по характеристикам машины."""
    model, _ = get_model()
    return float(np.exp(model.predict(make_row(**car))[0]))


def price_drivers(car: dict, top: int = 6) -> list[dict]:
    """Что подняло и что опустило цену именно этой машины.

    CatBoost раскладывает конкретный прогноз на вклады признаков (SHAP).
    Вклады считаются в логарифме цены, поэтому переводим их в множители:
    +0.1 в логарифме — это «дороже примерно на 10%», что человек понимает,
    в отличие от «плюс 0.1 логарифма».
    """
    from catboost import Pool

    model, _ = get_model()
    row = make_row(**car)
    shap = model.get_feature_importance(
        Pool(row, cat_features=CAT_FEATURES), type="ShapValues")[0]
    contribs = shap[:-1]                    # последний элемент — базовое значение
    order = np.argsort(-np.abs(contribs))[:top]
    out = []
    for i in order:
        c = float(contribs[i])
        out.append({
            "feature": FEATURES[i],
            "value": row.iloc[0][FEATURES[i]],
            "effect_pct": (np.exp(c) - 1) * 100,   # «дороже/дешевле на N%»
        })
    return out


def similar_cars(car: dict, limit: int = 5) -> pd.DataFrame:
    """Похожие живые объявления: та же марка и модель, близкий возраст.

    Нужны для доверия: одно дело «модель считает 6 млн», другое — увидеть
    пять реальных машин рядом.
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
    df = pd.read_sql(q, get_engine(),
                     params={"b": brand, "m": model_name, "a": int(age)},
                     dtype={"ad_id": str})
    return df.head(limit)


def price_position(car: dict, asking_price: float | None) -> dict | None:
    """Где цена продавца среди похожих машин.

    ВАЖНО про формулировки: это позиция среди ВЫСТАВЛЕННЫХ цен, а не прогноз
    срока продажи. Сказать «продашь за день» мы пока не можем — истории
    наблюдений слишком мало, и в ней видны только быстрые продажи
    (правое цензурирование). Поэтому здесь честное «дешевле/дороже
    большинства», без обещаний по срокам.
    """
    if not asking_price or asking_price <= 0:
        return None
    brand, model_name, age = car.get("brand"), car.get("model"), car.get("age")
    if not brand or not model_name or age is None:
        return None
    q = """SELECT price_tenge FROM clean_data
           WHERE brand = %(b)s AND model = %(m)s AND is_suspicious = 0
             AND price_tenge > 0 AND ABS(age - %(a)s) <= 2"""
    prices = pd.read_sql(q, get_engine(),
                         params={"b": brand, "m": model_name, "a": int(age)})
    if len(prices) < MIN_SIMILAR:
        return None
    p = prices.price_tenge.to_numpy()
    pct = float((p < asking_price).mean() * 100)
    if pct <= 25:
        label = "дешевле большинства похожих"
    elif pct >= 75:
        label = "дороже большинства похожих"
    else:
        label = "в середине рынка"
    return {"percentile": pct, "label": label, "n_similar": int(len(p)),
            "p25": float(np.percentile(p, 25)), "p75": float(np.percentile(p, 75))}


def listing_warnings(car: dict, asking_price: float | None,
                     fair: float, text: str = "") -> list[str]:
    """Проверка объявления теми же сигналами, что и антифрод-детектор.

    Смысл не в том, чтобы обвинить продавца, а наоборот: честному человеку
    полезно узнать, что его объявление выглядит подозрительно, и объяснить
    дешевизну заранее.
    """
    out = []
    if asking_price and fair > 0:
        ratio = asking_price / fair
        if ratio < 0.6:
            has_reason = bool(text and text.strip())
            out.append(
                "Цена примерно на {:.0f}% ниже похожих машин. ".format((1 - ratio) * 100)
                + ("Опишите причину — покупатели принимают необъяснимо дешёвые "
                   "объявления за приманку." if not has_reason else
                   "В описании есть текст — убедитесь, что причина названа прямо."))
        elif ratio > 1.5:
            out.append("Цена примерно на {:.0f}% выше похожих машин — "
                       "объявление может провисеть долго."
                       .format((ratio - 1) * 100))
    if car.get("mileage_km") in (None, "", 0):
        out.append("Не указан пробег. Объявления с пробегом смотрят примерно "
                   "на 16% чаще.")
    if (car.get("photos_count") or 0) < 5:
        out.append("Меньше пяти фотографий. Объявления с пятью и более "
                   "смотрят примерно на 77% чаще.")
    if len(text.strip()) < 50:
        out.append("Описание короче 50 символов. Объявления с описанием от "
                   "200 символов смотрят примерно на 36% чаще.")
    return out


def jsonable(obj):
    """numpy/pandas-типы → обычные python-типы.

    pandas отдаёт int64 и float64, а json.dumps их не умеет: без этой
    конверсии ответ падал с «Object of type int64 is not JSON serializable».
    NaN тоже превращаем в None — в JSON нет NaN, и браузер получил бы
    невалидный ответ.
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


def full_estimate(car: dict, asking_price: float | None = None,
                  text: str = "") -> dict:
    """Всё, что сервис знает про одну машину, одним вызовом."""
    fair = estimate_price(car)
    _, meta = get_model()
    val = meta.get("validation", {}).get("grouped_cv", {}).get("model", {})
    return jsonable({
        "fair_price": fair,
        "range_low": fair * RANGE_LOW,
        "range_high": fair * RANGE_HIGH,
        "model_mape_pct": val.get("mape_pct"),
        "trained_rows": meta.get("training_rows"),
        "drivers": price_drivers(car),
        "position": price_position(car, asking_price),
        "warnings": listing_warnings(car, asking_price, fair, text),
        "similar": similar_cars(car).to_dict("records"),
    })
