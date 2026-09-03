# -*- coding: utf-8 -*-
"""Select the database rows shown in the anomaly-review queue.

The queue contains all evidence required for a verdict without requesting
kolesa.kz. Text, price, badges, and damage terms are local; photos use a
separate CDN. This prevents manual browsing from bypassing the request budget.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kz.core.db import get_engine
from kz.report.label_cards.journal import LABELS_CSV

QUEUE_CSV = "data/eda/labeling_queue.csv"


def load_rows(include_queue: bool = True) -> pd.DataFrame:
    """Return the complete verdict queue; ``False`` is for rule-only audits.

    Rule positives alone cannot estimate missed fraud, and excluding residual
    candidates prevents evaluation of the second detector. The statistically
    complete sample is therefore the default.
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
    # Record the queue stratum because rule positives evaluate flag precision,
    # while random controls evaluate missed cases and recall.
    default = pd.Series(np.where(rows["is_suspicious"] == 1, "rule_positive", ""), index=rows.index)
    rows["stratum"] = rows["ad_id"].map(stratum).fillna(default)

    # Enriched detail-page fields that are absent from clean_data.
    enr = pd.read_sql(
        "SELECT ad_id, options_text, page_condition, has_vin, fetched_at FROM enriched",
        eng,
        dtype={"ad_id": str},
    )
    rows = rows.merge(enr, on="ad_id", how="left")

    photos = pd.read_sql("SELECT ad_id, position, url FROM photos", eng, dtype={"ad_id": str})
    photos = photos[photos["url"].fillna("").str.startswith("http")]
    photos = photos.sort_values(["ad_id", "position"])
    gal = photos.groupby("ad_id")["url"].apply(list)
    pos = photos.groupby("ad_id")["position"].apply(list)
    rows["photos"] = rows["ad_id"].map(gal)
    rows["photos"] = rows["photos"].apply(lambda v: v if isinstance(v, list) else [])
    # Local filenames use ad_id plus gallery position rather than the URL.
    rows["photo_positions"] = rows["ad_id"].map(pos)
    rows["photo_positions"] = rows["photo_positions"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    # Keep completed rows visible so previous decisions can be reviewed.
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        done = lab[lab["verdict"].isin(["fraud", "legit"])]
        rows["existing_verdict"] = rows["ad_id"].map(dict(zip(done["ad_id"], done["verdict"])))
    else:
        rows["existing_verdict"] = None
    return rows.sort_values(["existing_verdict", "price_z"], na_position="first")
