"""
Study 2 — personalised COUPON_UPC recommendation from real redemptions.

Study 1 recommends CAMPAIGNS scored against retailer assignments (mostly blanket).
Study 2 recommends individual COUPONs scored against actual redemptions — a
denser candidate set (~350 coupons) with behavioural ground truth, the setting
where latent-factor models are expected to outperform the segmented baseline.

Contents:
    Task builder
        compute_coupon_values           — per-COUPON_UPC expected line-item value
        build_redemption_task           — assemble the RedemptionTask container
    Evaluator (coupon-target)
        evaluate_coupons                — Recall@K, NDCG@K, E[revenue]@K, coverage
        format_coupon_results           — one-line-per-metric text summary
    Baselines
        random_coupon_baseline
        popularity_coupon_baseline
        repeat_buy_baseline
        item_knn_baseline
        last_category_coupon_baseline
    Latent-factor
        train_coupon_als                — implicit-ALS on binary hh × coupon matrix
        als_coupon_recommendations      — rank candidate coupons per household

The evaluator imports `recall_at_k`, `ndcg_at_k`, `bootstrap_mean_ci` from
the frozen src.eval_harness so metric formulas are shared with Study 1.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from implicit.als import AlternatingLeastSquares

from .config import (
    TRAIN_WEEKS, TEST_WEEKS, K, SEED,
    ALS_DIM, ALS_ALPHA, ALS_REG,
)
from .data_loader import day_range_for_weeks
from .eval_harness import (
    active_campaigns_in_test,
    recall_at_k,
    ndcg_at_k,
    bootstrap_mean_ci,
)


# ===========================================================================
# Coupon → expected line-item revenue (train-window average)
# ===========================================================================
def compute_coupon_values(
    coupon:       pd.DataFrame,
    transactions: pd.DataFrame,
    train_weeks:  list[int] | None = None,
) -> tuple[dict[int, float], float]:
    """
    Per COUPON_UPC, expected line-item SALES_VALUE when the coupon is used.

    Estimator: for every product the coupon covers, take the mean SALES_VALUE
    of that product's transaction lines in the train window; the coupon's
    value is the mean across its covered products. Coupons whose covered
    products never appeared in train transactions get the global median as
    a fallback.

    Returns (coupon_value_map, median_value).
    """
    train_weeks = train_weeks or TRAIN_WEEKS
    tx = transactions[transactions["WEEK_NO"].isin(train_weeks)]
    prod_val = (
        tx.groupby("PRODUCT_ID")["SALES_VALUE"]
        .mean()
        .rename("prod_mean_value")
    )
    cp = coupon[["COUPON_UPC", "PRODUCT_ID"]].drop_duplicates()
    cp = cp.merge(prod_val, on="PRODUCT_ID", how="left")
    coupon_value = (
        cp.groupby("COUPON_UPC")["prod_mean_value"]
        .mean()
        .dropna()
    )
    global_median = float(coupon_value.median()) if len(coupon_value) else 0.0
    return {int(k): float(v) for k, v in coupon_value.items()}, global_median


# ===========================================================================
# Task container
# ===========================================================================
@dataclass
class RedemptionTask:
    """Everything a Study 2 model needs to train, predict and be scored."""
    train_day_range:  tuple[int, int]
    test_day_range:   tuple[int, int]
    active_campaigns: set[int]                    # campaigns overlapping test window
    candidate_coupons: list[int]                  # sorted list of COUPON_UPCs
    train_redempt:    pd.DataFrame                # train-window redemptions
    test_redempt:     pd.DataFrame                # test-window redemptions (in candidate set)
    ground_truth:     dict[int, set[int]]         # hh -> set of COUPON_UPCs redeemed in test
    hh_eval:          list[int]                   # households with >=1 test redemption
    hh_train_history: dict[int, list[int]]        # hh -> list of COUPON_UPCs redeemed in train
                                                  #   (order = descending recency)
    coupon_to_campaign: dict[int, int]            # COUPON_UPC -> parent CAMPAIGN
    coupon_values: dict[int, float] = field(default_factory=dict)  # COUPON_UPC -> expected line-item value
    median_coupon_value: float = 0.0              # fallback for coupons with no train signal
    meta: dict = field(default_factory=dict)      # size stats for logging


# ===========================================================================
# Build the task from raw tables
# ===========================================================================
def build_redemption_task(
    coupon_redempt: pd.DataFrame,
    coupon:         pd.DataFrame,
    campaign_desc:  pd.DataFrame,
    transactions:   pd.DataFrame,
    train_weeks:    list[int] | None = None,
    test_weeks:     list[int] | None = None,
) -> RedemptionTask:
    """
    Assemble the Study 2 task.

    Candidate universe: COUPON_UPCs whose parent CAMPAIGN overlaps the test
    window (same "closed set" convention as Study 1, transposed to coupons).
    Ground truth: actual redemptions in the test window that fall in the
    candidate set.
    """
    train_weeks = train_weeks or TRAIN_WEEKS
    test_weeks  = test_weeks  or TEST_WEEKS

    d_train = day_range_for_weeks(transactions, train_weeks)
    d_test  = day_range_for_weeks(transactions, test_weeks)

    # ---- Candidate coupon universe --------------------------------------
    active = active_campaigns_in_test(campaign_desc, d_test[0], d_test[1])
    cp_active = coupon[coupon["CAMPAIGN"].isin(active)]
    candidate_coupons = sorted(cp_active["COUPON_UPC"].unique().tolist())
    coupon_to_campaign = (
        cp_active.drop_duplicates("COUPON_UPC")
        .set_index("COUPON_UPC")["CAMPAIGN"]
        .astype(int)
        .to_dict()
    )

    # ---- Train/test redemption slices -----------------------------------
    train_r = coupon_redempt[
        (coupon_redempt["DAY"] >= d_train[0]) & (coupon_redempt["DAY"] <= d_train[1])
    ].copy()
    test_r = coupon_redempt[
        (coupon_redempt["DAY"] >= d_test[0]) & (coupon_redempt["DAY"] <= d_test[1])
        & (coupon_redempt["COUPON_UPC"].isin(candidate_coupons))
    ].copy()

    # ---- Ground truth (test window) -------------------------------------
    ground_truth: dict[int, set[int]] = (
        test_r.groupby("household_key")["COUPON_UPC"]
        .apply(lambda s: set(int(x) for x in s))
        .to_dict()
    )
    hh_eval = sorted(ground_truth.keys())

    # ---- Train history per household (recency-ordered) ------------------
    hh_train_history: dict[int, list[int]] = {}
    if len(train_r) > 0:
        tr_sorted = train_r.sort_values(["household_key", "DAY"], ascending=[True, False])
        for hh, grp in tr_sorted.groupby("household_key"):
            seen: list[int] = []
            seen_set: set[int] = set()
            for c in grp["COUPON_UPC"].astype(int):
                if c not in seen_set:
                    seen.append(int(c))
                    seen_set.add(int(c))
            hh_train_history[int(hh)] = seen

    # ---- Coupon expected line-item value (train-window average) ---------
    coupon_values, median_val = compute_coupon_values(coupon, transactions, train_weeks)

    meta = {
        "n_active_campaigns":  len(active),
        "n_candidate_coupons": len(candidate_coupons),
        "n_train_redempt":     int(len(train_r)),
        "n_test_redempt":      int(len(test_r)),
        "n_hh_eval":           len(hh_eval),
        "n_hh_train_history":  len(hh_train_history),
        "median_train_history":
            int(np.median([len(v) for v in hh_train_history.values()])) if hh_train_history else 0,
        "median_test_truth":
            int(np.median([len(v) for v in ground_truth.values()])) if ground_truth else 0,
        "n_coupons_with_value": len(coupon_values),
        "median_coupon_value":  round(median_val, 4),
    }

    return RedemptionTask(
        train_day_range=d_train,
        test_day_range=d_test,
        active_campaigns=active,
        candidate_coupons=candidate_coupons,
        train_redempt=train_r,
        test_redempt=test_r,
        ground_truth=ground_truth,
        hh_eval=hh_eval,
        hh_train_history=hh_train_history,
        coupon_to_campaign=coupon_to_campaign,
        coupon_values=coupon_values,
        median_coupon_value=median_val,
        meta=meta,
    )


# ===========================================================================
# Coupon-level evaluate() — mirrors eval_harness.evaluate for a coupon target
# ===========================================================================
def evaluate_coupons(
    recs_df: pd.DataFrame,
    task:    RedemptionTask,
    k:       int = K,
) -> dict:
    """
    Score a Study 2 model against the frozen task.

    Metrics — identical formulas to Study 1's eval_harness, only the ground
    truth is coupon-level rather than campaign-level:
        recall_at_k         : mean per-household Recall@K
        recall_at_k_ci      : 95% bootstrap CI
        ndcg_at_k           : mean per-household NDCG@K
        ndcg_at_k_ci        : 95% bootstrap CI
        coverage            : distinct coupons in top-K / candidate coupons
        campaign_coverage   : distinct parent campaigns in top-K / active campaigns
        cold_start_recall   : recall among hh WITHOUT train history
        warm_recall         : recall among hh WITH train history
    """
    rank_cols = [f"rank_{i}" for i in range(1, k + 1)]
    truth = task.ground_truth
    hh_with_truth = set(truth.keys())
    eval_df = recs_df[recs_df["household_key"].isin(hh_with_truth)].copy()
    dropped = len(recs_df) - len(eval_df)

    def _val(c: int) -> float:
        return task.coupon_values.get(int(c), task.median_coupon_value)

    recalls, ndcgs, revenues, top_flat = [], [], [], []
    warm_r, cold_r = [], []
    for _, row in eval_df.iterrows():
        hh = int(row["household_key"])
        pred = [int(row[c]) for c in rank_cols if pd.notna(row[c])]
        gt = truth[hh]
        r = recall_at_k(pred, gt, k)
        n = ndcg_at_k(pred, gt, k)
        # Expected revenue @ K: sum of coupon-values across hits in top-K
        rev = float(sum(_val(c) for c in pred[:k] if c in gt))
        recalls.append(r)
        ndcgs.append(n)
        revenues.append(rev)
        top_flat.extend(pred)
        if hh in task.hh_train_history and len(task.hh_train_history[hh]) > 0:
            warm_r.append(r)
        else:
            cold_r.append(r)

    r_mean, r_lo, r_hi = bootstrap_mean_ci(recalls)
    n_mean, n_lo, n_hi = bootstrap_mean_ci(ndcgs)
    rev_mean, rev_lo, rev_hi = bootstrap_mean_ci(revenues)
    warm_mean, _, _ = bootstrap_mean_ci(warm_r) if warm_r else (float("nan"), float("nan"), float("nan"))
    cold_mean, _, _ = bootstrap_mean_ci(cold_r) if cold_r else (float("nan"), float("nan"), float("nan"))

    coupons_top = set(top_flat) & set(task.candidate_coupons)
    coverage = len(coupons_top) / max(len(task.candidate_coupons), 1)
    top_campaigns = {task.coupon_to_campaign[c] for c in coupons_top if c in task.coupon_to_campaign}
    camp_cov = len(top_campaigns) / max(len(task.active_campaigns), 1)

    return {
        "recall_at_k":            r_mean,
        "recall_at_k_ci":         (r_lo, r_hi),
        "ndcg_at_k":              n_mean,
        "ndcg_at_k_ci":           (n_lo, n_hi),
        "expected_revenue_at_k":     rev_mean,
        "expected_revenue_at_k_ci":  (rev_lo, rev_hi),
        "coverage":               coverage,
        "campaign_coverage":      camp_cov,
        "warm_recall_at_k":       warm_mean,
        "cold_recall_at_k":       cold_mean,
        "n_warm":                 len(warm_r),
        "n_cold":                 len(cold_r),
        "n_households_evaluated": len(eval_df),
        "n_households_dropped":   dropped,
        "n_candidate_coupons":    len(task.candidate_coupons),
        "k":                      k,
    }


def format_coupon_results(name: str, res: dict) -> str:
    r_lo, r_hi = res["recall_at_k_ci"]
    n_lo, n_hi = res["ndcg_at_k_ci"]
    v_lo, v_hi = res["expected_revenue_at_k_ci"]
    return (
        f"=== {name} ===\n"
        f"Recall@{res['k']}         : {res['recall_at_k']:.3f}  95% CI [{r_lo:.3f}, {r_hi:.3f}]\n"
        f"NDCG@{res['k']}           : {res['ndcg_at_k']:.3f}  95% CI [{n_lo:.3f}, {n_hi:.3f}]\n"
        f"E[revenue]@{res['k']}     : ${res['expected_revenue_at_k']:.3f}  95% CI [${v_lo:.3f}, ${v_hi:.3f}]\n"
        f"Coupon coverage     : {res['coverage']:.3f}  ({res['n_candidate_coupons']} candidates)\n"
        f"Campaign coverage   : {res['campaign_coverage']:.3f}\n"
        f"Warm hh Recall@{res['k']}  : {res['warm_recall_at_k']:.3f}  (n={res['n_warm']})\n"
        f"Cold hh Recall@{res['k']}  : {res['cold_recall_at_k']:.3f}  (n={res['n_cold']})\n"
        f"Households evaluated: {res['n_households_evaluated']}  (dropped {res['n_households_dropped']})"
    )


# ===========================================================================
# 0. Random baseline
# ===========================================================================
def random_coupon_baseline(
    task: RedemptionTask,
    k: int = K,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cands = np.array(task.candidate_coupons, dtype=np.int64)
    rows = []
    for hh in task.hh_eval:
        picks = rng.choice(cands, size=k, replace=False)
        rows.append({"household_key": hh, **{f"rank_{i+1}": int(picks[i]) for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Global popularity of redemptions
# ===========================================================================
def _popularity_ranking(task: RedemptionTask) -> list[int]:
    if len(task.train_redempt) == 0:
        return list(task.candidate_coupons)
    counts = (
        task.train_redempt[task.train_redempt["COUPON_UPC"].isin(task.candidate_coupons)]
        .groupby("COUPON_UPC")
        .size()
        .sort_values(ascending=False)
    )
    top = counts.index.astype(int).tolist()
    # Pad with unseen candidates for full coverage
    seen = set(top)
    for c in task.candidate_coupons:
        if c not in seen:
            top.append(int(c))
    return top


def popularity_coupon_baseline(
    task: RedemptionTask,
    k: int = K,
) -> pd.DataFrame:
    top = _popularity_ranking(task)
    while len(top) < k:
        top.append(top[-1] if top else int(task.candidate_coupons[0]))
    rows = [
        {"household_key": hh, **{f"rank_{i+1}": int(top[i]) for i in range(k)}}
        for hh in task.hh_eval
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# 2. Repeat-buy (personal history first, popularity-padded)
# ===========================================================================
def repeat_buy_baseline(
    task: RedemptionTask,
    k: int = K,
) -> pd.DataFrame:
    """
    Rank a household's OWN past redeemed coupons first (recency-ordered),
    then pad with global-popularity coupons the household has not yet seen.
    """
    pop = _popularity_ranking(task)
    cand_set = set(task.candidate_coupons)
    rows = []
    for hh in task.hh_eval:
        history = task.hh_train_history.get(hh, [])
        top: list[int] = []
        seen: set[int] = set()
        for c in history:
            if c in cand_set and c not in seen:
                top.append(int(c))
                seen.add(int(c))
            if len(top) >= k:
                break
        for c in pop:
            if len(top) >= k:
                break
            if c not in seen:
                top.append(int(c))
                seen.add(int(c))
        while len(top) < k:
            top.append(top[-1] if top else int(task.candidate_coupons[0]))
        rows.append({"household_key": hh, **{f"rank_{i+1}": top[i] for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 3. Item-kNN cosine on binary train redemption matrix
# ===========================================================================
def _build_hh_coupon_binary(task: RedemptionTask) -> tuple[csr_matrix, list[int], list[int]]:
    """Binary hh x coupon redemption matrix from the train window."""
    tr = task.train_redempt[task.train_redempt["COUPON_UPC"].isin(task.candidate_coupons)]
    households = sorted(tr["household_key"].astype(int).unique().tolist())
    coupons = list(task.candidate_coupons)
    hh_idx = {h: i for i, h in enumerate(households)}
    cp_idx = {c: i for i, c in enumerate(coupons)}
    rows = tr["household_key"].astype(int).map(hh_idx).values
    cols = tr["COUPON_UPC"].astype(int).map(cp_idx).values
    data = np.ones(len(tr), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(len(households), len(coupons)))
    # Deduplicate — a household redeeming the same coupon twice should still be 1
    mat.data = np.minimum(mat.data, 1.0)
    return mat, households, coupons


def item_knn_baseline(
    task: RedemptionTask,
    k: int = K,
    top_n_neighbours: int = 50,
) -> pd.DataFrame:
    """
    Item-item cosine kNN.

    Similarity S = normalize(M).T @ normalize(M) where M is hh x coupon binary.
    User score for item j = sum over i in history_u of S[i,j], evaluated via
    (normalize(M_u).T-like row) @ S. We drop the user's own historic items
    from the recommendation set (repeat_buy is a separate baseline).
    """
    mat, households, coupons = _build_hh_coupon_binary(task)
    n_hh, n_cp = mat.shape
    hh_idx = {h: i for i, h in enumerate(households)}
    cp_idx = {c: i for i, c in enumerate(coupons)}

    if n_hh == 0 or n_cp == 0:
        # Nothing to learn from — fall back to popularity
        return popularity_coupon_baseline(task, k=k)

    # Cosine similarity between coupons via L2-normalised item vectors
    item_norm = normalize(mat.T, norm="l2", axis=1)         # (n_cp, n_hh)
    sim = (item_norm @ item_norm.T).toarray()               # (n_cp, n_cp) dense — 354^2 is fine
    np.fill_diagonal(sim, 0.0)

    # Optionally sparsify by keeping only top_n_neighbours per item
    if top_n_neighbours < n_cp:
        keep = np.argpartition(-sim, top_n_neighbours, axis=1)[:, :top_n_neighbours]
        mask = np.zeros_like(sim, dtype=bool)
        rows_ix = np.arange(n_cp)[:, None]
        mask[rows_ix, keep] = True
        sim = np.where(mask, sim, 0.0)

    pop = _popularity_ranking(task)
    cand_set = set(task.candidate_coupons)
    rows = []
    for hh in task.hh_eval:
        history = [c for c in task.hh_train_history.get(hh, []) if c in cand_set]
        if not history or hh not in hh_idx:
            # Cold start -> popularity
            top = pop[:k]
        else:
            hist_ix = np.array([cp_idx[c] for c in history if c in cp_idx], dtype=np.int64)
            # Score every coupon = sum of similarities from historic coupons
            scores = sim[hist_ix].sum(axis=0)
            # Exclude items the household has already redeemed in train
            scores[hist_ix] = -np.inf
            order = np.argsort(-scores)
            top = [coupons[i] for i in order[:k] if scores[i] > -np.inf]
            # Pad with popularity if kNN produced fewer than k items
            seen = set(top)
            for c in pop:
                if len(top) >= k:
                    break
                if c not in seen:
                    top.append(int(c))
                    seen.add(int(c))
        while len(top) < k:
            top.append(top[-1] if top else int(task.candidate_coupons[0]))
        rows.append({"household_key": hh, **{f"rank_{i+1}": int(top[i]) for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# 4. Last-category coupon baseline (Study 1's last_category ported to coupons)
# ===========================================================================
def last_category_coupon_baseline(
    task:         RedemptionTask,
    transactions: pd.DataFrame,
    product:      pd.DataFrame,
    coupon:       pd.DataFrame,
    k:            int = K,
    train_weeks:  list[int] | None = None,
) -> pd.DataFrame:
    """
    Rank each candidate coupon by the household's train-window spend in the
    commodities that coupon covers.

    coupon → product → COMMODITY_DESC gives each coupon a set of commodities;
    the household's train spend by commodity gives each household a spend
    vector; the score is the sum. Fallback for households with no train
    spend / coupons with no commodities → global popularity.
    """
    train_weeks = train_weeks or TRAIN_WEEKS

    # Household × commodity spend in train window
    tx = transactions[transactions["WEEK_NO"].isin(train_weeks)]
    tx = tx.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    hh_com_spend = (
        tx.groupby(["household_key", "COMMODITY_DESC"])["SALES_VALUE"]
        .sum()
    )

    # Coupon → set of commodities it covers (candidate coupons only)
    cand_set = set(task.candidate_coupons)
    cp = coupon[coupon["COUPON_UPC"].isin(cand_set)]
    cp = cp.merge(product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="inner")
    coupon_coms = {
        int(cu): list(grp["COMMODITY_DESC"].unique())
        for cu, grp in cp.groupby("COUPON_UPC")
    }

    pop = _popularity_ranking(task)
    rows = []
    for hh in task.hh_eval:
        try:
            hh_series = hh_com_spend.loc[hh]
        except KeyError:
            hh_series = pd.Series(dtype=float)

        scores: dict[int, float] = {}
        for c in task.candidate_coupons:
            coms = coupon_coms.get(int(c), [])
            if not coms:
                scores[c] = 0.0
            else:
                scores[c] = float(hh_series.reindex(coms).fillna(0).sum())

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, s in ordered if s > 0][:k]
        seen = set(top)
        # Pad with popularity if fewer than k coupons had positive spend
        for c in pop:
            if len(top) >= k:
                break
            if c not in seen:
                top.append(int(c))
                seen.add(int(c))
        while len(top) < k:
            top.append(top[-1] if top else int(task.candidate_coupons[0]))
        rows.append({"household_key": hh, **{f"rank_{i+1}": int(top[i]) for i in range(k)}})
    return pd.DataFrame(rows)


# ===========================================================================
# Latent-factor: implicit ALS on the binary hh × coupon matrix
# ===========================================================================
def train_coupon_als(
    task: RedemptionTask,
    dim: int = ALS_DIM,
    alpha: float = ALS_ALPHA,
    reg: float = ALS_REG,
    iterations: int = 15,
    seed: int = SEED,
):
    """
    Weighted implicit ALS on the binary hh x coupon redemption matrix.
    Returns (model, households, coupons, matrix). The `implicit` library
    applies confidence = 1 + alpha * value internally.
    """
    mat, households, coupons = _build_hh_coupon_binary(task)
    model = AlternatingLeastSquares(
        factors=dim,
        regularization=reg,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
    )
    model.fit(mat, show_progress=False)
    return model, households, coupons, mat


def als_coupon_recommendations(
    model,
    households: list[int],
    coupons: list[int],
    task: RedemptionTask,
    k: int = K,
    exclude_seen: bool = False,
) -> pd.DataFrame:
    """
    Rank candidate coupons for each eval household by user-factor dot item-factor.
    Cold-start households (no train redemptions) fall back to popularity.

    exclude_seen=False by default because a household who redeemed a coupon
    once often redeems it again — matches the repeat-buy pattern in grocery.
    """
    hh_idx = {h: i for i, h in enumerate(households)}
    cp_idx = {c: i for i, c in enumerate(coupons)}
    user_factors = model.user_factors
    item_factors = model.item_factors

    known_cand_cols = np.array(
        [cp_idx[c] for c in task.candidate_coupons if c in cp_idx],
        dtype=np.int64,
    )
    known_cand_ids = [c for c in task.candidate_coupons if c in cp_idx]

    pop = _popularity_ranking(task)
    rows = []
    for hh in task.hh_eval:
        if hh not in hh_idx or len(known_cand_cols) == 0:
            top = pop[:k]
        else:
            u = user_factors[hh_idx[hh]]
            scores = item_factors[known_cand_cols] @ u
            if exclude_seen:
                seen_ix = {
                    cp_idx[c] for c in task.hh_train_history.get(hh, []) if c in cp_idx
                }
                for i, col in enumerate(known_cand_cols):
                    if col in seen_ix:
                        scores[i] = -np.inf
            order = np.argsort(-scores)
            top = [known_cand_ids[i] for i in order[:k]]
            # Pad with popularity if needed
            seen = set(top)
            for c in pop:
                if len(top) >= k:
                    break
                if c not in seen:
                    top.append(int(c))
                    seen.add(int(c))
        while len(top) < k:
            top.append(top[-1] if top else int(task.candidate_coupons[0]))
        rows.append({"household_key": hh, **{f"rank_{i+1}": int(top[i]) for i in range(k)}})
    return pd.DataFrame(rows)
