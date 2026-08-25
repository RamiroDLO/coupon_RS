"""
========================================================================
FROZEN EVALUATION HARNESS  —  Dunnhumby coupon-campaign recommender
========================================================================

Rules (see docs/EVAL_HARNESS_SOP.md for the full policy):
    1. This file is frozen at the start of modelling work and DOES NOT
       CHANGE while models are being compared.
    2. Every model (baseline, ALS, LightFM, hybrid, whatever) is scored
       through `evaluate()` — no local variants.
    3. If a genuine bug is found after freeze, the fix is escalated to the
       team, ALL prior models are re-scored, and the change is logged in
       docs/EVAL_HARNESS_SOP.md under "Change log".

Model contract:
    Every model outputs a `recs_df` with columns:
        household_key : int
        rank_1        : int   (CAMPAIGN id, highest predicted score)
        rank_2        : int   (CAMPAIGN id, next)
        rank_3        : int   (CAMPAIGN id, next)
    Rows for households with no valid recommendation may be omitted.

Owner:      <team member — fill in at freeze>
Frozen on:  <date — fill in at freeze>
========================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import K, TEST_WEEKS, BOOTSTRAP_N, BOOTSTRAP_ALPHA, SEED


# ===========================================================================
# Ground truth
# ===========================================================================
def active_campaigns_in_test(
    campaign_desc: pd.DataFrame,
    day_start: int,
    day_end: int,
) -> set[int]:
    """Campaigns whose [START_DAY, END_DAY] window overlaps the test window."""
    mask = (campaign_desc["END_DAY"] >= day_start) & (campaign_desc["START_DAY"] <= day_end)
    return set(campaign_desc.loc[mask, "CAMPAIGN"])


def build_ground_truth(
    campaign_table: pd.DataFrame,
    active_campaigns: set[int],
) -> dict[int, set[int]]:
    """
    Per household, the set of *active* campaigns assigned by the retailer.
    Households with zero active assignments are excluded from the returned dict.
    """
    filtered = campaign_table[campaign_table["CAMPAIGN"].isin(active_campaigns)]
    grouped = filtered.groupby("household_key")["CAMPAIGN"].apply(set)
    return grouped.to_dict()


# ===========================================================================
# Per-household ranking metrics
# ===========================================================================
def recall_at_k(pred: list[int], truth: set[int], k: int = K) -> float | None:
    """Fraction of truth items appearing in the top-K. None if truth is empty."""
    if not truth:
        return None
    top = pred[:k]
    return len(set(top) & truth) / len(truth)


def ndcg_at_k(pred: list[int], truth: set[int], k: int = K) -> float | None:
    """Normalised DCG at K with binary relevance. None if truth is empty."""
    if not truth:
        return None
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(pred[:k]) if item in truth)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ===========================================================================
# Redemption-value uplift (the "money" metric)
# ===========================================================================
def redemption_uplift(
    recs_df: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    coupon_redempt: pd.DataFrame,
    day_start: int,
    day_end: int,
    k: int = K,
) -> dict[str, float]:
    """
    Compare redeemed coupon value between:
        MATCHED    - households whose top-K contains >=1 assigned campaign
        UNMATCHED  - households whose top-K contains  0 assigned campaigns
    Returns per-household mean redeemed value in the test window for each group,
    plus the ratio (matched / unmatched). See docs/EVAL_HARNESS_SOP.md for the
    proxy interpretation — this is NOT a causal uplift, it's a correlational
    signal that motivates a future A/B test.
    """
    rank_cols = [f"rank_{i}" for i in range(1, k + 1)]

    # Restrict to test-window redemptions
    r = coupon_redempt[(coupon_redempt["DAY"] >= day_start) & (coupon_redempt["DAY"] <= day_end)]
    if "AMOUNT" in r.columns:
        redeemed_value_per_hh = (
            r.groupby("household_key")["AMOUNT"].sum().rename("redeemed_value").reset_index()
        )
    else:
        redeemed_value_per_hh = (
            r.groupby("household_key")["COUPON_UPC"].count().rename("redeemed_value").reset_index()
        )

    matched_hh, unmatched_hh = [], []
    for _, row in recs_df.iterrows():
        hh = int(row["household_key"])
        if hh not in ground_truth:
            continue
        top = [int(row[c]) for c in rank_cols if pd.notna(row[c])]
        if set(top) & ground_truth[hh]:
            matched_hh.append(hh)
        else:
            unmatched_hh.append(hh)

    v = redeemed_value_per_hh.set_index("household_key")["redeemed_value"]
    matched_mean = float(v.reindex(matched_hh).fillna(0).mean()) if matched_hh else 0.0
    unmatched_mean = float(v.reindex(unmatched_hh).fillna(0).mean()) if unmatched_hh else 0.0
    ratio = matched_mean / unmatched_mean if unmatched_mean > 0 else float("nan")

    return {
        "matched_mean_value":   matched_mean,
        "unmatched_mean_value": unmatched_mean,
        "uplift_ratio":         ratio,
        "n_matched":            len(matched_hh),
        "n_unmatched":          len(unmatched_hh),
    }


# ===========================================================================
# Bootstrap confidence intervals
# ===========================================================================
def bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_N,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI for the mean of `values`. Returns (mean, lo, hi)."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    resamples = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    means = resamples.mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi))


# ===========================================================================
# The one function every model routes through
# ===========================================================================
def evaluate(
    recs_df: pd.DataFrame,
    campaign_table: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
    transactions: pd.DataFrame,
    k: int = K,
) -> dict:
    """
    Score a recommender's output against the frozen protocol.

    Returns dict with:
        recall_at_k              : mean per-household Recall@K
        recall_at_k_ci           : 95% bootstrap CI (lo, hi)
        ndcg_at_k                : mean per-household NDCG@K
        ndcg_at_k_ci             : 95% bootstrap CI
        coverage                 : distinct campaigns in top-K / active campaigns
        n_households_evaluated   : after filtering
        n_households_dropped     : filtered out (no test-window signal)
        n_active_campaigns       : candidate-set size in test window
        k                        : the K used
        redemption_uplift        : dict from redemption_uplift(...)
    """
    from .data_loader import day_range_for_weeks

    day_start, day_end = day_range_for_weeks(transactions, TEST_WEEKS)
    active = active_campaigns_in_test(campaign_desc, day_start, day_end)
    truth = build_ground_truth(campaign_table, active)

    hh_with_truth = set(truth.keys())
    eval_df = recs_df[recs_df["household_key"].isin(hh_with_truth)].copy()
    dropped = len(recs_df) - len(eval_df)

    rank_cols = [f"rank_{i}" for i in range(1, k + 1)]

    recalls, ndcgs, top_k_flat = [], [], []
    for _, row in eval_df.iterrows():
        hh = int(row["household_key"])
        pred = [int(row[c]) for c in rank_cols if pd.notna(row[c])]
        gt = truth[hh]
        recalls.append(recall_at_k(pred, gt, k))
        ndcgs.append(ndcg_at_k(pred, gt, k))
        top_k_flat.extend(pred)

    r_mean, r_lo, r_hi = bootstrap_mean_ci(recalls)
    n_mean, n_lo, n_hi = bootstrap_mean_ci(ndcgs)

    coverage = len(set(top_k_flat) & active) / max(len(active), 1)

    uplift = redemption_uplift(eval_df, truth, coupon_redempt, day_start, day_end, k=k)

    return {
        "recall_at_k":            r_mean,
        "recall_at_k_ci":         (r_lo, r_hi),
        "ndcg_at_k":              n_mean,
        "ndcg_at_k_ci":           (n_lo, n_hi),
        "coverage":               coverage,
        "n_households_evaluated": len(eval_df),
        "n_households_dropped":   dropped,
        "n_active_campaigns":     len(active),
        "k":                      k,
        "redemption_uplift":      uplift,
        "day_range":              (day_start, day_end),
    }


# ===========================================================================
# Pretty printer
# ===========================================================================
def format_results(name: str, res: dict) -> str:
    """One-line-per-metric text summary."""
    r_lo, r_hi = res["recall_at_k_ci"]
    n_lo, n_hi = res["ndcg_at_k_ci"]
    u = res["redemption_uplift"]
    return (
        f"=== {name} ===\n"
        f"Recall@{res['k']}          : {res['recall_at_k']:.3f}  95% CI [{r_lo:.3f}, {r_hi:.3f}]\n"
        f"NDCG@{res['k']}            : {res['ndcg_at_k']:.3f}  95% CI [{n_lo:.3f}, {n_hi:.3f}]\n"
        f"Coverage             : {res['coverage']:.3f}  ({res['n_active_campaigns']} active campaigns)\n"
        f"Households evaluated : {res['n_households_evaluated']}  (dropped {res['n_households_dropped']})\n"
        f"Redemption uplift    : matched {u['matched_mean_value']:.3f} vs unmatched {u['unmatched_mean_value']:.3f}"
        f"  ratio {u['uplift_ratio']:.2f}  (n_matched={u['n_matched']}, n_unmatched={u['n_unmatched']})"
    )
