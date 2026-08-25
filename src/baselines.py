"""
Baselines for the Dunnhumby coupon-campaign recommender.

All baselines are trained on TRAIN_WEEKS only — no test-window leakage.
Each returns a `recs_df` with columns [household_key, rank_1, rank_2, rank_3]
that can be scored by src.eval_harness.evaluate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import K, TRAIN_WEEKS, RFM_QUANTILES


# ===========================================================================
# 0. Random baseline (sanity floor)
# ===========================================================================
def random_baseline(
    households: list[int],
    candidate_campaigns: list[int],
    k: int = K,
    seed: int = 42,
) -> pd.DataFrame:
    """Uniformly random top-K for each household — sanity floor."""
    rng = np.random.default_rng(seed)
    rows = []
    for hh in households:
        picks = list(rng.choice(candidate_campaigns, size=k, replace=False))
        rows.append({"household_key": hh, **{f"rank_{i+1}": picks[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Popularity baseline (same top-K for everyone)
# ===========================================================================
def popularity_baseline(
    campaign_table: pd.DataFrame,
    households: list[int],
    candidate_campaigns: set[int],
    k: int = K,
) -> pd.DataFrame:
    """
    Most-assigned campaigns overall (train-window assignments only).
    Returns the same top-K for every household.
    """
    counts = (
        campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)]
        .groupby("CAMPAIGN")
        .size()
        .sort_values(ascending=False)
    )
    top = counts.head(k).index.tolist()
    while len(top) < k:
        top.append(top[-1] if top else next(iter(candidate_campaigns)))
    rows = [
        {"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}}
        for hh in households
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# 2. RFM segmentation
# ===========================================================================
def compute_rfm(
    transactions: pd.DataFrame,
    train_weeks: list[int] = None,
    quantiles: int = RFM_QUANTILES,
) -> pd.DataFrame:
    """
    Recency / Frequency / Monetary scores per household, computed on TRAIN_WEEKS only.

    Returns DataFrame indexed by household_key with columns:
        recency, frequency, monetary  (raw values, in days / baskets / dollars)
        R, F, M                        (1..quantiles quintile scores)
        segment                        (e.g. "555" for RFM top; string concatenation)
    """
    if train_weeks is None:
        train_weeks = TRAIN_WEEKS

    train = transactions[transactions["WEEK_NO"].isin(train_weeks)]
    ref_day = int(train["DAY"].max())

    per_hh = train.groupby("household_key").agg(
        last_day=("DAY", "max"),
        frequency=("BASKET_ID", "nunique"),
        monetary=("SALES_VALUE", "sum"),
    )
    per_hh["recency"] = ref_day - per_hh["last_day"]
    per_hh = per_hh.drop(columns=["last_day"])

    # Lower recency (more recent) = better R score. Higher F/M = better score.
    per_hh["R"] = pd.qcut(-per_hh["recency"], q=quantiles, labels=False, duplicates="drop") + 1
    per_hh["F"] = pd.qcut(per_hh["frequency"], q=quantiles, labels=False, duplicates="drop") + 1
    per_hh["M"] = pd.qcut(per_hh["monetary"], q=quantiles, labels=False, duplicates="drop") + 1

    per_hh["segment"] = (
        per_hh["R"].astype(int).astype(str)
        + per_hh["F"].astype(int).astype(str)
        + per_hh["M"].astype(int).astype(str)
    )
    return per_hh


# ===========================================================================
# 3. RFM-segment popularity — the primary baseline
# ===========================================================================
def segment_popularity_baseline(
    campaign_table: pd.DataFrame,
    rfm_df: pd.DataFrame,
    households: list[int],
    candidate_campaigns: set[int],
    k: int = K,
) -> pd.DataFrame:
    """
    Per household, recommend the top-K campaigns most assigned to that
    household's RFM segment, based on train-window assignments only.

    Fallback for households/segments with insufficient data:
        - Household not in rfm_df       -> global popularity
        - Segment with < k campaigns    -> pad with global popularity
    """
    # Global popularity fallback ranking
    global_counts = (
        campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)]
        .groupby("CAMPAIGN")
        .size()
        .sort_values(ascending=False)
    )
    global_top = global_counts.index.tolist()

    # Segment -> ranked list of campaigns (candidate-set filtered, popularity sorted)
    hh_seg = rfm_df["segment"].to_dict()
    ct = campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)].copy()
    ct["segment"] = ct["household_key"].map(hh_seg)
    seg_rank = (
        ct.dropna(subset=["segment"])
        .groupby(["segment", "CAMPAIGN"])
        .size()
        .reset_index(name="n")
        .sort_values(["segment", "n"], ascending=[True, False])
    )
    seg_to_ranked = seg_rank.groupby("segment")["CAMPAIGN"].apply(list).to_dict()

    rows = []
    for hh in households:
        seg = hh_seg.get(hh)
        top = list(seg_to_ranked.get(seg, [])) if seg is not None else []
        # Pad with global popularity if segment ranking is short
        for c in global_top:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        top = top[:k]
        while len(top) < k:  # last-resort pad
            top.append(global_top[0])
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 4. Last-purchase-repeat (grocery-specific strong baseline)
# ===========================================================================
def last_category_baseline(
    transactions: pd.DataFrame,
    product: pd.DataFrame,
    coupon: pd.DataFrame,
    households: list[int],
    candidate_campaigns: set[int],
    k: int = K,
    train_weeks: list[int] = None,
) -> pd.DataFrame:
    """
    Rank campaigns by how much the household has recently spent in
    the commodities each campaign's coupons cover. Train-window only.

    Strong grocery baseline — grocery is highly habit-driven, so recent
    spend in a category is a solid signal for coupon interest.
    """
    if train_weeks is None:
        train_weeks = TRAIN_WEEKS

    # Household x commodity spend in train window
    tx = transactions[transactions["WEEK_NO"].isin(train_weeks)]
    tx = tx.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    hh_com_spend = (
        tx.groupby(["household_key", "COMMODITY_DESC"])["SALES_VALUE"]
        .sum()
        .reset_index()
    )

    # Campaign -> set of commodities its coupons cover
    cp = coupon[coupon["CAMPAIGN"].isin(candidate_campaigns)]
    cp = cp.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    campaign_coms = cp.groupby("CAMPAIGN")["COMMODITY_DESC"].apply(set).to_dict()

    # Per household, score each campaign = sum of household spend across covered commodities
    hh_com_spend = hh_com_spend.set_index(["household_key", "COMMODITY_DESC"])["SALES_VALUE"]

    rows = []
    for hh in households:
        scores = {}
        try:
            hh_series = hh_com_spend.loc[hh]  # commodity -> spend
            if isinstance(hh_series, float):  # single-row edge case
                hh_series = pd.Series({hh_com_spend.loc[hh].name: hh_series})
        except KeyError:
            hh_series = pd.Series(dtype=float)
        for camp, coms in campaign_coms.items():
            scores[camp] = float(hh_series.reindex(list(coms)).fillna(0).sum())
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_ids = [t[0] for t in top[:k]]
        while len(top_ids) < k:
            top_ids.append(top_ids[-1] if top_ids else next(iter(candidate_campaigns)))
        rows.append({"household_key": hh, **{f"rank_{i+1}": top_ids[i] for i in range(k)}})
    return pd.DataFrame(rows)
