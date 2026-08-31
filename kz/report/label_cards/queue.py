# -*- coding: utf-8 -*-
"""Что показывать разметчику: очередь объявлений из базы.

Сюда собирается ровно то, что нужно для вердикта, — и ни одного запроса к
kolesa.kz: весь текст, цена, бейдж и damage-слова уже лежат в базе, а фото
на стороннем CDN. Ручной браузинг бил бы по тому же IP, что и джобы, мимо
бюджета catch_up — именно эта смесь и положила адрес 2026-07-23.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kz.core.db import get_engine
from kz.report.label_cards.journal import LABELS_CSV

QUEUE_CSV = "data/eda/labeling_queue.csv"

def load_rows(include_queue: bool = True) -> pd.DataFrame:
    """Полная очередь вердиктов; ``False`` оставлен для аудита правил.

    Одних правиловых подозрительных недостаточно: без random_control нельзя
    оценить пропущенный fraud, а без residual_candidate — второй детектор.
    Поэтому статистически корректная полная выборка является default.
    """
    eng = get_engine()
    cd = pd.read_sql("SELECT * FROM clean_data", eng, dtype={"ad_id": str})
    ids = set(cd.loc[cd["is_suspicious"] == 1, "ad_id"])
    stratum = {}
    if include_queue and Path(QUEUE_CSV).exists():
        q = pd.read_csv(QUEUE_CSV, dtype={"ad_id": str})
        ids |= set(q["ad_id"])
        stratum = dict(zip(q["ad_id"], q["sampling_stratum"]))
    rows = cd[cd["ad_id"].isin(ids)].copy()
    # Из какого слоя очереди объявление. Без этого разметчик не понимает,
    # ЧТО именно проверяет: у правиловых вопрос «верен ли флаг», а у
    # контрольных — «не пропустили ли мы обман», и это разные задачи.
    default = pd.Series(np.where(rows["is_suspicious"] == 1, "rule_positive", ""),
                        index=rows.index)
    rows["stratum"] = rows["ad_id"].map(stratum).fillna(default)

    # Доп. поля со страницы, которых нет в clean_data.
    enr = pd.read_sql("SELECT ad_id, options_text, page_condition, has_vin, "
                      "fetched_at FROM enriched", eng, dtype={"ad_id": str})
    rows = rows.merge(enr, on="ad_id", how="left")

    photos = pd.read_sql("SELECT ad_id, position, url FROM photos", eng,
                         dtype={"ad_id": str})
    photos = photos[photos["url"].fillna("").str.startswith("http")]
    photos = photos.sort_values(["ad_id", "position"])
    gal = photos.groupby("ad_id")["url"].apply(list)
    pos = photos.groupby("ad_id")["position"].apply(list)
    rows["photos"] = rows["ad_id"].map(gal)
    rows["photos"] = rows["photos"].apply(lambda v: v if isinstance(v, list) else [])
    # Позиции нужны, чтобы найти локально скачанный файл: он назван по
    # ad_id и позиции, а не по URL.
    rows["photo_positions"] = rows["ad_id"].map(pos)
    rows["photo_positions"] = rows["photo_positions"].apply(
        lambda v: v if isinstance(v, list) else [])

    # Уже размеченные помечаем, но НЕ выкидываем: удобно перепроверить.
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        done = lab[lab["verdict"].isin(["fraud", "legit"])]
        rows["existing_verdict"] = rows["ad_id"].map(
            dict(zip(done["ad_id"], done["verdict"])))
    else:
        rows["existing_verdict"] = None
    return rows.sort_values(["existing_verdict", "price_z"], na_position="first")
