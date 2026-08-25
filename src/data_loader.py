"""
Data loading utilities.

Column-name convention matches the Dunnhumby Complete Journey CSVs:
`household_key` is lowercase; every other ID/column is UPPERCASE.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------
def load_transactions() -> pd.DataFrame:
    """~2.6M rows. WEEK_NO already provided; no need to derive from DAY."""
    dtype = {
        "household_key":     "int32",
        "BASKET_ID":         "int64",
        "DAY":               "int16",
        "PRODUCT_ID":        "int32",
        "QUANTITY":          "int32",
        "SALES_VALUE":       "float32",
        "STORE_ID":          "int32",
        "RETAIL_DISC":       "float32",
        "TRANS_TIME":        "int16",
        "WEEK_NO":           "int16",
        "COUPON_DISC":       "float32",
        "COUPON_MATCH_DISC": "float32",
    }
    return pd.read_csv(DATA_DIR / "transaction_data.csv", dtype=dtype)


def load_product() -> pd.DataFrame:
    """~92k SKUs with department/commodity/sub-commodity hierarchy."""
    return pd.read_csv(DATA_DIR / "product.csv")


def load_campaign_desc() -> pd.DataFrame:
    """30 campaigns with DESCRIPTION (Type A/B/C), START_DAY, END_DAY."""
    return pd.read_csv(DATA_DIR / "campaign_desc.csv")


def load_campaign_table() -> pd.DataFrame:
    """Ground truth: which household received which campaign."""
    return pd.read_csv(DATA_DIR / "campaign_table.csv")


def load_coupon() -> pd.DataFrame:
    """COUPON_UPC -> PRODUCT_ID -> CAMPAIGN mapping."""
    return pd.read_csv(DATA_DIR / "coupon.csv")


def load_coupon_redempt() -> pd.DataFrame:
    """Actual redemptions."""
    return pd.read_csv(DATA_DIR / "coupon_redempt.csv")


def load_hh_demographic() -> pd.DataFrame:
    """~800 households only — coverage is NON-RANDOM (see EDA Q6)."""
    return pd.read_csv(DATA_DIR / "hh_demographic.csv")


def load_causal_data() -> pd.DataFrame:
    """~36M rows, 700MB — only load when needed."""
    return pd.read_csv(DATA_DIR / "causal_data.csv")


# ---------------------------------------------------------------------------
# Convenience: everything except causal_data
# ---------------------------------------------------------------------------
def load_core() -> dict[str, pd.DataFrame]:
    """All tables except the huge causal_data. Returns a dict keyed by short name."""
    return {
        "transactions":     load_transactions(),
        "product":          load_product(),
        "campaign_desc":    load_campaign_desc(),
        "campaign_table":   load_campaign_table(),
        "coupon":           load_coupon(),
        "coupon_redempt":   load_coupon_redempt(),
        "hh_demographic":   load_hh_demographic(),
    }


# ---------------------------------------------------------------------------
# Derived helper: DAY range for a set of weeks
# ---------------------------------------------------------------------------
def day_range_for_weeks(transactions: pd.DataFrame, weeks: list[int]) -> tuple[int, int]:
    """Return (min_DAY, max_DAY) observed for the given weeks in transactions."""
    mask = transactions["WEEK_NO"].isin(weeks)
    days = transactions.loc[mask, "DAY"]
    return int(days.min()), int(days.max())
