"""
Study 1 — campaign recommendation models (non-FM).

Every model returns `recs_df` with columns [household_key, rank_1..rank_k]
of CAMPAIGN ids, ready to be scored by `src.eval_harness.evaluate`.
All models are trained on TRAIN_WEEKS only — no test-window leakage.

Contents:
    Baselines
        random_baseline                    — sanity floor
        popularity_baseline                — global campaign popularity
        compute_rfm                        — RFM quintile scores per household
        segment_popularity_baseline        — RFM-segment-conditional popularity
        segment_demographic_baseline       — RFM × demographic cell (winner)
        last_category_baseline             — spend-in-covered-commodities score

    Latent-factor
        build_hh_commodity_matrix
        build_hh_campaign_matrix
        train_als_model
        als_commodity_recommendations      — score campaigns via covered commodities
        als_campaign_recommendations       — score campaigns directly via ALS

External deps: numpy, pandas, scipy, implicit.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from .config import K, TRAIN_WEEKS, RFM_QUANTILES, ALS_DIM, ALS_ALPHA, ALS_REG, SEED


# ===========================================================================
# 0. Random baseline (sanity floor)
# ===========================================================================
def random_baseline(
    households: list[int],
    candidate_campaigns: list[int],
    k: int = K,
    seed: int = SEED,
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
    """Most-assigned campaigns overall — same top-K for every household."""
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
    train_weeks: list[int] | None = None,
    quantiles: int = RFM_QUANTILES,
) -> pd.DataFrame:
    """
    Per household: Recency / Frequency / Monetary computed on TRAIN_WEEKS only.
    Returns DataFrame indexed by household_key with raw values, quintile
    scores (R, F, M), and a 3-digit `segment` string like '555'.
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
# 3. RFM-segment popularity baseline
# ===========================================================================
def segment_popularity_baseline(
    campaign_table: pd.DataFrame,
    rfm_df: pd.DataFrame,
    households: list[int],
    candidate_campaigns: set[int],
    k: int = K,
) -> pd.DataFrame:
    """
    Recommend the top-K campaigns most assigned to each household's RFM segment.
    Households missing from rfm_df fall back to global popularity.
    """
    global_counts = (
        campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)]
        .groupby("CAMPAIGN")
        .size()
        .sort_values(ascending=False)
    )
    global_top = global_counts.index.tolist()

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
        for c in global_top:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        top = top[:k]
        while len(top) < k:
            top.append(global_top[0])
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 4. RFM × demographic baseline (Study 1 winner)
# ===========================================================================
_DEMO_COLS = ["AGE_DESC", "INCOME_DESC", "KID_CATEGORY_DESC"]


def build_demographic_key(hh_demographic: pd.DataFrame) -> pd.Series:
    """household_key -> demographic bucket string (e.g. '45-54|50-74K|None/Unknown')."""
    df = hh_demographic.copy()
    for c in _DEMO_COLS:
        df[c] = df[c].fillna("MISSING").astype(str)
    df["demo_key"] = df[_DEMO_COLS[0]].str.cat(df[_DEMO_COLS[1:]], sep="|")
    return df.set_index("household_key")["demo_key"]


def segment_demographic_baseline(
    campaign_table: pd.DataFrame,
    rfm_df: pd.DataFrame,
    hh_demographic: pd.DataFrame,
    households: list[int],
    candidate_campaigns: set[int],
    k: int = K,
    min_cell_support: int = 5,
) -> pd.DataFrame:
    """
    Three-tier hierarchical fallback:
        1. (RFM × demographic) cell popularity, if cell has >= min_cell_support
        2. RFM-segment popularity
        3. Global popularity
    """
    hh_seg = rfm_df["segment"].to_dict()
    hh_demo = build_demographic_key(hh_demographic).to_dict()

    ct = campaign_table[campaign_table["CAMPAIGN"].isin(candidate_campaigns)].copy()
    ct["segment"] = ct["household_key"].map(hh_seg)
    ct["demo_key"] = ct["household_key"].map(hh_demo)

    global_top = ct.groupby("CAMPAIGN").size().sort_values(ascending=False).index.tolist()

    seg_rank = (
        ct.dropna(subset=["segment"])
        .groupby(["segment", "CAMPAIGN"])
        .size()
        .reset_index(name="n")
        .sort_values(["segment", "n"], ascending=[True, False])
    )
    seg_to_ranked = seg_rank.groupby("segment")["CAMPAIGN"].apply(list).to_dict()

    combo_rank = (
        ct.dropna(subset=["segment", "demo_key"])
        .groupby(["segment", "demo_key", "CAMPAIGN"])
        .size()
        .reset_index(name="n")
    )
    combo_totals = combo_rank.groupby(["segment", "demo_key"])["n"].sum().rename("total").reset_index()
    combo_valid = combo_totals[combo_totals["total"] >= min_cell_support][["segment", "demo_key"]]
    combo_rank = combo_rank.merge(combo_valid, on=["segment", "demo_key"], how="inner")
    combo_rank = combo_rank.sort_values(["segment", "demo_key", "n"], ascending=[True, True, False])
    combo_to_ranked = (
        combo_rank.groupby(["segment", "demo_key"])["CAMPAIGN"].apply(list).to_dict()
    )

    rows = []
    for hh in households:
        seg = hh_seg.get(hh)
        demo = hh_demo.get(hh)
        top: list = []
        if seg is not None and demo is not None:
            top = list(combo_to_ranked.get((seg, demo), []))
        if len(top) < k and seg is not None:
            for c in seg_to_ranked.get(seg, []):
                if len(top) >= k:
                    break
                if c not in top:
                    top.append(c)
        for c in global_top:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        top = top[:k]
        while len(top) < k:
            top.append(top[-1] if top else next(iter(candidate_campaigns)))
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 5. Last-category baseline (grocery habit signal)
# ===========================================================================
def last_category_baseline(
    transactions: pd.DataFrame,
    product: pd.DataFrame,
    coupon: pd.DataFrame,
    households: list[int],
    candidate_campaigns: set[int],
    k: int = K,
    train_weeks: list[int] | None = None,
) -> pd.DataFrame:
    """
    Score each campaign by household spend across the commodities its coupons
    cover (train window only). Strong grocery baseline — recent spend in a
    category is a solid signal for coupon interest.
    """
    if train_weeks is None:
        train_weeks = TRAIN_WEEKS

    tx = transactions[transactions["WEEK_NO"].isin(train_weeks)]
    tx = tx.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    hh_com_spend = (
        tx.groupby(["household_key", "COMMODITY_DESC"])["SALES_VALUE"]
        .sum()
        .reset_index()
    )

    cp = coupon[coupon["CAMPAIGN"].isin(candidate_campaigns)]
    cp = cp.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    campaign_coms = cp.groupby("CAMPAIGN")["COMMODITY_DESC"].apply(set).to_dict()

    hh_com_spend = hh_com_spend.set_index(["household_key", "COMMODITY_DESC"])["SALES_VALUE"]

    rows = []
    for hh in households:
        try:
            hh_series = hh_com_spend.loc[hh]
            if isinstance(hh_series, float):
                hh_series = pd.Series({hh_com_spend.loc[hh].name: hh_series})
        except KeyError:
            hh_series = pd.Series(dtype=float)
        scores = {}
        for camp, coms in campaign_coms.items():
            scores[camp] = float(hh_series.reindex(list(coms)).fillna(0).sum())
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_ids = [t[0] for t in top[:k]]
        while len(top_ids) < k:
            top_ids.append(top_ids[-1] if top_ids else next(iter(candidate_campaigns)))
        rows.append({"household_key": hh, **{f"rank_{i+1}": top_ids[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 6. ALS on household × commodity spend (Study 1 latent-factor)
# ===========================================================================
def build_hh_commodity_matrix(
    transactions: pd.DataFrame,
    product: pd.DataFrame,
    train_weeks: list[int] | None = None,
    log_transform: bool = True,
) -> tuple[csr_matrix, list[int], list[str]]:
    if train_weeks is None:
        train_weeks = TRAIN_WEEKS
    tx = transactions[transactions["WEEK_NO"].isin(train_weeks)]
    tx = tx.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    agg = (
        tx.groupby(["household_key", "COMMODITY_DESC"])["SALES_VALUE"]
        .sum()
        .reset_index()
    )
    agg = agg[agg["SALES_VALUE"] > 0].copy()
    agg["value"] = np.log1p(agg["SALES_VALUE"]) if log_transform else agg["SALES_VALUE"]

    households = sorted(agg["household_key"].unique().tolist())
    commodities = sorted(agg["COMMODITY_DESC"].unique().tolist())
    hh_idx = {h: i for i, h in enumerate(households)}
    com_idx = {c: i for i, c in enumerate(commodities)}

    rows = agg["household_key"].map(hh_idx).values
    cols = agg["COMMODITY_DESC"].map(com_idx).values
    data = agg["value"].values.astype(np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(len(households), len(commodities)))
    return mat, households, commodities


def als_commodity_recommendations(
    model,
    households: list[int],
    commodities: list[str],
    coupon: pd.DataFrame,
    product: pd.DataFrame,
    candidate_campaigns: set[int],
    eval_households: list[int],
    fallback_ranking: list[int],
    k: int = K,
    aggregation: str = "max",  # "max" or "mean"
) -> pd.DataFrame:
    """Rank campaigns by MAX (default) predicted commodity preference."""
    hh_idx = {h: i for i, h in enumerate(households)}
    com_idx = {c: i for i, c in enumerate(commodities)}
    user_factors = model.user_factors
    item_factors = model.item_factors

    cp = coupon[coupon["CAMPAIGN"].isin(candidate_campaigns)]
    cp = cp.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    campaign_com_idx: dict[int, np.ndarray] = {}
    for camp, grp in cp.groupby("CAMPAIGN"):
        idxs = [com_idx[c] for c in grp["COMMODITY_DESC"].unique() if c in com_idx]
        campaign_com_idx[camp] = np.array(idxs, dtype=np.int64)
    for camp in candidate_campaigns:
        campaign_com_idx.setdefault(camp, np.array([], dtype=np.int64))

    rows = []
    for hh in eval_households:
        if hh not in hh_idx:
            top = list(fallback_ranking[:k])
            while len(top) < k:
                top.append(top[-1] if top else next(iter(candidate_campaigns)))
            rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
            continue

        u = user_factors[hh_idx[hh]]
        com_scores = item_factors @ u

        camp_scores: dict[int, float] = {}
        for camp, idxs in campaign_com_idx.items():
            if len(idxs) == 0:
                camp_scores[camp] = -np.inf
            elif aggregation == "max":
                camp_scores[camp] = float(com_scores[idxs].max())
            else:
                camp_scores[camp] = float(com_scores[idxs].mean())

        top_pairs = sorted(camp_scores.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in top_pairs[:k]]
        for c in fallback_ranking:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        while len(top) < k:
            top.append(top[-1] if top else next(iter(candidate_campaigns)))
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})

    return pd.DataFrame(rows)


# ===========================================================================
# 7. ALS on household × campaign assignment (Study 1 latent-factor)
# ===========================================================================
def build_hh_campaign_matrix(
    campaign_table: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    train_day_max: int,
) -> tuple[csr_matrix, list[int], list[int]]:
    """Binary household × campaign matrix from campaigns started within train window."""
    train_camps = set(
        campaign_desc[campaign_desc["START_DAY"] <= train_day_max]["CAMPAIGN"]
    )
    ct = campaign_table[campaign_table["CAMPAIGN"].isin(train_camps)]

    households = sorted(ct["household_key"].unique().tolist())
    campaigns = sorted(ct["CAMPAIGN"].unique().tolist())
    hh_idx = {h: i for i, h in enumerate(households)}
    cp_idx = {c: i for i, c in enumerate(campaigns)}

    rows = ct["household_key"].map(hh_idx).values
    cols = ct["CAMPAIGN"].map(cp_idx).values
    data = np.ones(len(ct), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(len(households), len(campaigns)))
    return mat, households, campaigns


def als_campaign_recommendations(
    model,
    households: list[int],
    campaigns_in_matrix: list[int],
    candidate_campaigns: set[int],
    eval_households: list[int],
    fallback_ranking: list[int],
    k: int = K,
) -> pd.DataFrame:
    """Score each candidate campaign directly from ALS user/item factors."""
    hh_idx = {h: i for i, h in enumerate(households)}
    cp_idx = {c: i for i, c in enumerate(campaigns_in_matrix)}
    user_factors = model.user_factors
    item_factors = model.item_factors

    known_candidates = [c for c in candidate_campaigns if c in cp_idx]
    unknown_candidates = [c for c in candidate_campaigns if c not in cp_idx]
    known_col_ix = np.array([cp_idx[c] for c in known_candidates], dtype=np.int64)

    rows = []
    for hh in eval_households:
        if hh not in hh_idx or len(known_candidates) == 0:
            top = list(fallback_ranking[:k])
            while len(top) < k:
                top.append(top[-1] if top else next(iter(candidate_campaigns)))
            rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
            continue

        u = user_factors[hh_idx[hh]]
        scores = item_factors[known_col_ix] @ u
        order = np.argsort(-scores)
        top = [known_candidates[i] for i in order[:k]]
        for c in unknown_candidates:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        for c in fallback_ranking:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        while len(top) < k:
            top.append(top[-1] if top else next(iter(candidate_campaigns)))
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})

    return pd.DataFrame(rows)


# ===========================================================================
# Shared ALS training (both variants above)
# ===========================================================================
def train_als_model(
    matrix: csr_matrix,
    dim: int = ALS_DIM,
    alpha: float = ALS_ALPHA,
    reg: float = ALS_REG,
    iterations: int = 15,
    seed: int = SEED,
):
    model = AlternatingLeastSquares(
        factors=dim,
        regularization=reg,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
    )
    model.fit(matrix, show_progress=False)
    return model
