"""Implementation for the `kz.transform.clean` module."""

import pathlib as _p

_expected = "clean.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files may have been mixed up while copying."
    )


import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from kz.core.db import get_engine

LABELS_CSV = "data/manual_labels.csv"
OUT_CSV = "data/clean/clean_data.csv"


# noqa: F401 -- intentional exception


from kz.transform.damage import DAMAGE_PATTERNS, has_damage as _has_damage  # noqa: F401
from kz.transform.price_basis import classify_price_basis

CURRENT_YEAR = date.today().year


MIN_YEAR = 1950
MAX_PRICE = 300_000_000
MIN_PRICE = 200_000
MAX_MILEAGE = 1_000_000
MAX_KM_PER_YEAR = 100_000
ROBUST_Z_THRESHOLD = 3.5
MIN_GROUP_SIZE = 8

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `normalize`."""
    df = df.copy()

    df["price_tenge"] = pd.to_numeric(df["price_tenge"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    df["engine_volume"] = pd.to_numeric(df["engine_volume"], errors="coerce")

    for col in ["brand", "model", "city", "engine_type", "transmission", "body_type", "condition"]:
        df[col] = df[col].astype("string").str.strip()

    is_new_label = df["labels"].fillna("").str.contains("Новая")
    df.loc[df["condition"].isna() & is_new_label, "condition"] = "новый"

    df["age"] = CURRENT_YEAR - df["year"] + 1

    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `add_missing_indicators`."""
    df = df.copy()
    df["is_mileage_missing"] = df["mileage_km"].isna().astype(int)
    df["is_description_missing"] = df["description"].isna().astype(int)
    return df


def apply_hard_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `apply_hard_rules`."""
    df = df.copy()
    reasons = [[] for _ in range(len(df))]

    def flag(mask: pd.Series, reason: str):
        for i in np.where(mask.fillna(False).to_numpy())[0]:
            reasons[i].append(reason)

    flag(df["year"] < MIN_YEAR, "year_too_old")
    flag(df["year"] > CURRENT_YEAR + 1, "year_in_future")
    flag(df["price_tenge"] < MIN_PRICE, "price_too_low")
    flag(df["price_tenge"] > MAX_PRICE, "price_too_high")
    flag(df["mileage_km"] > MAX_MILEAGE, "mileage_extreme")

    flag(
        (df["condition"] == "б/у") & (df["mileage_km"] == 0) & (df["age"] >= 2),
        "used_but_zero_mileage",
    )

    km_per_year = df["mileage_km"] / df["age"]
    flag(km_per_year > MAX_KM_PER_YEAR, "km_per_year_extreme")

    flag(
        (df["age"] <= 4) & (df["price_tenge"] < 4_000_000) & (df["condition"] != "новый"),
        "young_car_cheap",
    )

    df["rule_reasons"] = ["|".join(r) for r in reasons]
    return df


def add_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `add_price_outliers`."""
    df = df.copy()
    df["log_price"] = np.log(df["price_tenge"])

    bins = [0, 3, 7, 12, 20, np.inf]
    labels = ["0-3", "4-7", "8-12", "13-20", "21+"]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels)

    def robust_z(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(0.0, index=s.index)
        return 0.6745 * (s - med) / mad

    df["model_key"] = (df["brand"].fillna("") + " " + df["model"].fillna("")).str.strip()

    levels = [
        ("model", ["model_key", "age_bucket"]),
        ("brand", ["brand", "age_bucket"]),
        ("age", ["age_bucket"]),
    ]
    df["price_z"] = np.nan
    df["z_group_level"] = ""
    for name, keys in levels:
        grp = df.groupby(keys, observed=True)["log_price"]
        z = grp.transform(robust_z)
        size = grp.transform("size")
        take = df["price_z"].isna() & (size >= MIN_GROUP_SIZE)
        df.loc[take, "price_z"] = z[take]
        df.loc[take, "z_group_level"] = name
    df["price_z"] = df["price_z"].fillna(0.0).round(2)

    df["stat_reasons"] = ""
    df.loc[df["price_z"] < -ROBUST_Z_THRESHOLD, "stat_reasons"] = "price_anomaly_low"
    df["info_flags"] = ""
    df.loc[df["price_z"] > ROBUST_Z_THRESHOLD, "info_flags"] = "price_anomaly_high"
    return df


def add_duplicate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `add_duplicate_flags`."""
    df = df.copy()
    dup = df.duplicated(subset=["title", "year", "price_tenge"], keep=False)

    dealer_like = df["condition"].eq("новый").fillna(False) | df["labels"].fillna("").str.contains(
        "От дилера|Новая"
    )
    dealer_like = dealer_like.astype(bool)

    def _base_color(v) -> str:
        """Implement `_base_color`."""
        if pd.isna(v):
            return ""
        return str(v).strip().lower().split(" ")[0]

    def corroborated(g: pd.DataFrame) -> bool:
        if len(g) > 5:
            return False
        has_color = "color" in g.columns
        rows = g.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]

                if has_color:
                    ca, cb = _base_color(a.get("color")), _base_color(b.get("color"))
                    if ca and cb and ca != cb:
                        continue

                ma, mb = a.get("mileage_km"), b.get("mileage_km")
                same_mileage = bool(pd.notna(ma) and ma > 1000 and ma == mb)
                da = "" if pd.isna(a.get("description")) else str(a.get("description"))
                db = "" if pd.isna(b.get("description")) else str(b.get("description"))
                if same_mileage or (da and da == db):
                    return True
        return False

    strong_groups = set()
    for key, g in df[dup].groupby(["title", "year", "price_tenge"]):
        if corroborated(g):
            strong_groups.add(key)
    keys = list(zip(df["title"], df["year"], df["price_tenge"]))
    strong = pd.Series([k in strong_groups for k in keys], index=df.index)

    df["dup_reasons"] = np.where(dup & ~dealer_like & strong, "possible_repost", "")
    weak = dup & ~dealer_like & ~strong
    df.loc[weak, "info_flags"] = (
        df.loc[weak, "info_flags"]
        .fillna("")
        .apply(lambda s: (s + "|" if s else "") + "repost_unconfirmed")
    )
    return df


def add_photo_reuse_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `add_photo_reuse_flags`."""
    df = df.copy()
    df["photo_reasons"] = ""
    engine = get_engine()
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT to_regclass('public.photo_duplicates')")).scalar()
        if not exists:
            return df
        dups = pd.read_sql(
            "SELECT ad_id_a, ad_id_b FROM photo_duplicates",
            conn,
            dtype={"ad_id_a": str, "ad_id_b": str},
        )
    if dups.empty:
        return df
    flagged = set(dups["ad_id_a"]) | set(dups["ad_id_b"])
    df.loc[df["ad_id"].isin(flagged), "photo_reasons"] = "shared_photo_diff_car"
    return df


def exculpate(df: pd.DataFrame) -> pd.DataFrame:
    """Implement `exculpate`."""
    df = df.copy()

    has_damage = _has_damage

    explained = df["text_full"].fillna("").map(has_damage)
    if "damage_keywords" in df.columns:
        explained |= df["damage_keywords"].fillna("").str.len() > 0
    if "customs_cleared" in df.columns:
        explained |= df["customs_cleared"].eq("Нет").fillna(False)
    if "verdict" in df.columns:
        explained |= df["verdict"].eq("legit").fillna(False)

    if "page_status_badge" in df.columns:
        badge = df["page_status_badge"].fillna("").str.lower()
        explained |= badge.str.contains("аварийн|не на ходу|заложен")

    mask = (df["stat_reasons"] == "price_anomaly_low") & explained
    df.loc[mask, "stat_reasons"] = ""
    df.loc[mask, "info_flags"] = (
        df.loc[mask, "info_flags"]
        .fillna("")
        .apply(lambda s: (s + "|" if s else "") + "low_price_explained")
    )

    def _drop(reasons: str, token: str) -> str:
        return "|".join(p for p in str(reasons).split("|") if p and p != token)

    if "rule_reasons" in df.columns:
        ycc = df["rule_reasons"].str.contains("young_car_cheap", na=False) & explained
        df.loc[ycc, "rule_reasons"] = df.loc[ycc, "rule_reasons"].apply(
            lambda s: _drop(s, "young_car_cheap")
        )

        need = ycc & ~df["info_flags"].fillna("").str.contains("low_price_explained")
        df.loc[need, "info_flags"] = (
            df.loc[need, "info_flags"]
            .fillna("")
            .apply(lambda s: (s + "|" if s else "") + "low_price_explained")
        )

    dealer_fin = df["condition"].eq("новый").fillna(False) | df["labels"].fillna("").str.contains(
        "Официальный дилер", case=False
    )
    mask = (df["stat_reasons"] == "price_anomaly_low") & dealer_fin
    df.loc[mask, "stat_reasons"] = ""
    df.loc[mask, "info_flags"] = (
        df.loc[mask, "info_flags"]
        .fillna("")
        .apply(lambda s: (s + "|" if s else "") + "dealer_financing_price")
    )

    if "kolesa_avg_price" in df.columns:
        avg = pd.to_numeric(df["kolesa_avg_price"], errors="coerce")

        near_market = (avg > 0) & (df["price_tenge"] >= 0.80 * avg)
        mask = (df["stat_reasons"] == "price_anomaly_low") & near_market.fillna(False)
        df.loc[mask, "stat_reasons"] = ""
        df.loc[mask, "info_flags"] = (
            df.loc[mask, "info_flags"]
            .fillna("")
            .apply(lambda s: (s + "|" if s else "") + "kolesa_price_ok")
        )

    urgency = pd.Series(False, index=df.index)
    for col in ["description", "seller_comment"]:
        if col in df.columns:
            urgency |= df[col].fillna("").str.lower().str.contains("срочн")
    cheap_urgent = (df["stat_reasons"] == "price_anomaly_low") & urgency
    df.loc[cheap_urgent, "stat_reasons"] = "price_anomaly_low|cheap_and_urgent"

    if "verdict" in df.columns:
        fraud = df["verdict"].eq("fraud").fillna(False)
        df.loc[fraud & (df["stat_reasons"] == ""), "stat_reasons"] = "confirmed_by_review"
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def join_reasons(row):
        parts = [row["rule_reasons"], row["stat_reasons"], row["dup_reasons"], row["photo_reasons"]]
        return "|".join(p for p in parts if p)

    df["suspicion_reasons"] = df.apply(join_reasons, axis=1)
    df["is_suspicious"] = (df["suspicion_reasons"] != "").astype(int)
    return df.drop(columns=["rule_reasons", "stat_reasons", "dup_reasons", "photo_reasons"])


def quality_report(df: pd.DataFrame):
    """Implement `quality_report`."""
    log.info("=" * 60)
    log.info(
        f"Rows: {len(df)}, suspicious: {df['is_suspicious'].sum()} "
        f"({df['is_suspicious'].mean():.1%})"
    )
    log.info("-" * 60)
    all_reasons = (
        df.loc[df["is_suspicious"] == 1, "suspicion_reasons"]
        .str.split("|")
        .explode()
        .value_counts()
    )
    for reason, cnt in all_reasons.items():
        log.info(f"  {reason:<25} {cnt}")
    info = df["info_flags"].replace("", np.nan).dropna().value_counts()
    if len(info):
        log.info("Informational flags (not counted as suspicious):")
        for reason, cnt in info.items():
            log.info(f"  {reason:<25} {cnt}")
    log.info("-" * 60)
    log.info("Most common missing fields:")
    na = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    for col, pct in na[na > 0].head(6).items():
        log.info(f"  {col:<20} {pct}%")
    log.info("=" * 60)


def main():
    engine = get_engine()

    df = pd.read_sql("SELECT * FROM raw_ads", engine, dtype={"ad_id": str})

    enr = pd.read_sql("SELECT * FROM enriched", engine, dtype={"ad_id": str})
    if not enr.empty:
        cols = [
            "ad_id",
            "customs_cleared",
            "drive",
            "steering",
            "color",
            "generation",
            "page_mileage_km",
            "damage_keywords",
            "seller_comment",
            "kolesa_avg_price",
            "page_status_badge",
        ]
        df = df.merge(enr[[c for c in cols if c in enr.columns]], on="ad_id", how="left")
        filled = df["mileage_km"].isna() & df["page_mileage_km"].notna()
        df.loc[filled, "mileage_km"] = df.loc[filled, "page_mileage_km"]
        log.info(f"Enriched rows: {enr.shape[0]}, mileage values backfilled: {filled.sum()}")

    df["text_full"] = df.get("seller_comment", pd.Series(index=df.index, dtype="object")).fillna(
        df["description"]
    )
    customs = df.get("customs_cleared", pd.Series(index=df.index, dtype="object"))
    df["price_basis"] = [
        classify_price_basis(text, clearance, price)
        for text, clearance, price in zip(df["text_full"], customs, df["price_tenge"])
    ]

    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str}).drop_duplicates("ad_id", keep="last")
        df = df.merge(lab[["ad_id", "verdict"]], on="ad_id", how="left")
        log.info(f"Manual verdicts: {lab.shape[0]}")

    df = normalize(df)
    df = add_missing_indicators(df)
    df = apply_hard_rules(df)
    df = add_price_outliers(df)
    df = exculpate(df)
    df = add_duplicate_flags(df)
    df = add_photo_reuse_flags(df)
    df = finalize(df)

    st = pd.read_sql("SELECT ad_id, status FROM ad_status", engine, dtype={"ad_id": str})
    if not st.empty:
        df = df.merge(st, on="ad_id", how="left")
        df["status"] = df["status"].fillna("active")
    else:
        df["status"] = "active"

    df.to_csv(OUT_CSV, index=False)

    with engine.begin() as conn:
        df.to_sql("clean_data", conn, if_exists="replace", index=False)
    quality_report(df)
    log.info(f"Saved → {OUT_CSV} and the clean_data table")


if __name__ == "__main__":
    main()
