"""
Product-level coupon recommendation  —  products x purchases framing.

Task
    For each household, rank the Top-K *coupon-eligible products* it is most
    likely to buy in the test window.

Ground truth
    The coupon-eligible products the household actually purchased in the test
    weeks (`src.config.TEST_WEEKS`).

Candidate set
    Coupon-eligible products (appear in `coupon.csv`) that were purchased by at
    least one household during the training weeks.

Metric formulas (`recall_at_k`, `ndcg_at_k`, `bootstrap_mean_ci`) are imported
from the frozen `src.eval_harness` so they cannot diverge from the rest of the
project. Every model — baseline or ALS — is scored through `evaluate()` here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TRAIN_WEEKS, TEST_WEEKS, K, SEED
from .data_loader import load_purchases, load_product, load_coupon
from .eval_harness import recall_at_k, ndcg_at_k, bootstrap_mean_ci


# ===========================================================================
# Task container
# ===========================================================================
@dataclass
class PurchaseTask:
    candidate_products: list[int]                # eligible + >=1 train purchase (sorted)
    ground_truth: dict[int, set[int]]            # hh -> candidate products bought in test
    hh_eval: list[int]                           # households with >=1 ground-truth product
    hh_train_products: dict[int, list[int]]      # hh -> candidate products bought in train, recent first
    hh_train_freq: dict[int, list[int]]          # hh -> same products, most-often-bought first
    hh_tier: dict[int, str]                      # hh -> "light" | "mid" | "heavy" (trip-count terciles; for stratified reporting)
    hh_rfm: dict[int, str]                       # hh -> RFM value tier "v1".."v5" (v5 = best customers)
    popularity: list[int]                        # candidate products, most-purchased in train first
    trending: list[int]                          # popularity with exponential recency decay (~30-day half-life)
    wilson: list[int]                            # candidates ranked by Wilson lower bound of adoption rate
    rfm_popularity: dict[str, list[int]]         # RFM tier -> candidate products ranked within that tier
    product_commodity: dict[int, str]            # candidate product -> COMMODITY_DESC
    hh_commodity_spend: pd.Series                # (hh, commodity) -> train SALES_VALUE
    meta: dict = field(default_factory=dict)


def build_purchase_task(
    purchases: pd.DataFrame | None = None,
    product:   pd.DataFrame | None = None,
    coupon:    pd.DataFrame | None = None,
    train_weeks: list[int] | None = None,
    test_weeks:  list[int] | None = None,
) -> PurchaseTask:
    purchases   = load_purchases() if purchases is None else purchases
    product     = load_product()   if product   is None else product
    coupon      = load_coupon()    if coupon    is None else coupon
    train_weeks = train_weeks or TRAIN_WEEKS
    test_weeks  = test_weeks  or TEST_WEEKS

    eligible = set(coupon["PRODUCT_ID"].unique())

    train   = purchases[purchases["WEEK_NO"].isin(train_weeks)]
    test    = purchases[purchases["WEEK_NO"].isin(test_weeks)]
    train_e = train[train["PRODUCT_ID"].isin(eligible)]
    test_e  = test[test["PRODUCT_ID"].isin(eligible)]

    # --- candidate set: eligible products with >=1 training purchase ---
    candidate_products = sorted(int(p) for p in train_e["PRODUCT_ID"].unique())
    cand_set = set(candidate_products)
    test_e = test_e[test_e["PRODUCT_ID"].isin(cand_set)]

    # --- ground truth: candidate products each household bought in the test weeks ---
    ground_truth = (
        test_e.groupby("household_key")["PRODUCT_ID"]
        .apply(lambda s: {int(x) for x in s})
        .to_dict()
    )
    hh_eval = sorted(int(h) for h in ground_truth)

    # --- per-household training history ---
    #   hh_train_products : most-recent purchase first
    #   hh_train_freq     : most-often bought first (the repeat-purchase signal)
    hh_train_products: dict[int, list[int]] = {}
    hh_train_freq: dict[int, list[int]] = {}
    tr = train_e.sort_values(["household_key", "DAY"], ascending=[True, False])
    for hh, grp in tr.groupby("household_key"):
        seen, seen_set = [], set()
        for p in grp["PRODUCT_ID"].astype(int):
            if p not in seen_set:
                seen.append(p)
                seen_set.add(p)
        hh_train_products[int(hh)] = seen
        freq = grp["PRODUCT_ID"].astype(int).value_counts()          # count, ties broken by recency
        hh_train_freq[int(hh)] = [int(p) for p in freq.index]

    # --- activity tier: trip-count terciles (used only for stratified reporting) ---
    trips = purchases.groupby("household_key")["BASKET_ID"].nunique()
    hh_tier = {int(h): str(t) for h, t in pd.qcut(trips, 3, labels=["light", "mid", "heavy"]).items()}

    # --- RFM household segments (training window) ---
    #   Recency  = days since the household's last trip (smaller is better -> quintile 5)
    #   Frequency = number of distinct trips
    #   Monetary = total spend
    #   value tier = quintiles of (R + F + M), v5 = best customers.
    end_day = int(train["DAY"].max())
    rfm = train.groupby("household_key").agg(
        recency=("DAY", lambda d: end_day - d.max()),
        frequency=("BASKET_ID", "nunique"),
        monetary=("SALES_VALUE", "sum"),
    )
    r_q = pd.qcut(rfm["recency"].rank(method="first"),   5, labels=[5, 4, 3, 2, 1]).astype(int)
    f_q = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m_q = pd.qcut(rfm["monetary"].rank(method="first"),  5, labels=[1, 2, 3, 4, 5]).astype(int)
    value_tier = pd.qcut((r_q + f_q + m_q).rank(method="first"), 5,
                         labels=["v1", "v2", "v3", "v4", "v5"])
    hh_rfm = {int(h): str(t) for h, t in value_tier.items()}

    # --- rung 2: global popularity of candidate products (train purchase-line count) ---
    counts = (
        train_e.groupby("PRODUCT_ID").size()
        .reindex(candidate_products, fill_value=0)
        .sort_values(ascending=False)
    )
    popularity = [int(p) for p in counts.index]

    # --- rung 4: trending = recency-weighted popularity (exp. decay, ~30-day half-life) ---
    decay = np.exp(-np.log(2) * (end_day - train_e["DAY"].to_numpy()) / 30.0)
    trend = (
        pd.Series(decay, index=train_e["PRODUCT_ID"].to_numpy())
        .groupby(level=0).sum()
        .reindex(candidate_products, fill_value=0.0)
        .sort_values(ascending=False)
    )
    trending = [int(p) for p in trend.index]

    # --- rung 5: uncertainty-aware = Wilson 95% lower bound on each product's
    #     adoption rate (distinct buyers / all households). Down-weights products
    #     bought heavily by only a handful of households. ---
    n_hh = int(purchases["household_key"].nunique())
    buyers = (
        train_e.groupby("PRODUCT_ID")["household_key"].nunique()
        .reindex(candidate_products, fill_value=0)
        .to_numpy(dtype=float)
    )
    p_hat = buyers / n_hh
    z = 1.96
    denom = 1.0 + z ** 2 / n_hh
    centre = p_hat + z ** 2 / (2 * n_hh)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_hh + z ** 2 / (4 * n_hh ** 2))
    wilson_lb = (centre - margin) / denom
    wilson = [int(candidate_products[i]) for i in np.argsort(-wilson_lb)]

    # --- popularity within each RFM value tier ---
    train_e_rfm = train_e.assign(vtier=train_e["household_key"].map(hh_rfm))
    rfm_popularity: dict[str, list[int]] = {}
    for tier in ("v1", "v2", "v3", "v4", "v5"):
        c = (
            train_e_rfm.loc[train_e_rfm["vtier"] == tier]
            .groupby("PRODUCT_ID").size()
            .reindex(candidate_products, fill_value=0)
            .sort_values(ascending=False)
        )
        rfm_popularity[tier] = [int(p) for p in c.index]

    # --- product -> commodity, and household -> commodity training spend ---
    prod_com = product.drop_duplicates("PRODUCT_ID").set_index("PRODUCT_ID")["COMMODITY_DESC"]
    product_commodity = {p: str(prod_com.get(p, "UNKNOWN")) for p in candidate_products}
    tr_com = train_e.merge(
        product[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="left"
    )
    hh_commodity_spend = tr_com.groupby(["household_key", "COMMODITY_DESC"])["SALES_VALUE"].sum()

    gt_sizes   = [len(v) for v in ground_truth.values()]
    hist_sizes = [len(v) for v in hh_train_products.values()]
    meta = {
        "n_candidate_products":    len(candidate_products),
        "n_eval_households":       len(hh_eval),
        "n_hh_with_train_history": len(hh_train_products),
        "median_train_history":    int(np.median(hist_sizes)) if hist_sizes else 0,
        "median_ground_truth":     int(np.median(gt_sizes)) if gt_sizes else 0,
        "eval_tier_counts": {
            t: sum(1 for h in hh_eval if hh_tier.get(h) == t) for t in ("light", "mid", "heavy")
        },
    }

    return PurchaseTask(
        candidate_products=candidate_products,
        ground_truth=ground_truth,
        hh_eval=hh_eval,
        hh_train_products=hh_train_products,
        hh_train_freq=hh_train_freq,
        hh_tier=hh_tier,
        hh_rfm=hh_rfm,
        popularity=popularity,
        trending=trending,
        wilson=wilson,
        rfm_popularity=rfm_popularity,
        product_commodity=product_commodity,
        hh_commodity_spend=hh_commodity_spend,
        meta=meta,
    )


# ===========================================================================
# Evaluator  —  every model routes through this
# ===========================================================================
def evaluate(recs_df: pd.DataFrame, task: PurchaseTask, k: int = K) -> dict:
    rank_cols = [f"rank_{i}" for i in range(1, k + 1)]
    truth = task.ground_truth
    eval_df = recs_df[recs_df["household_key"].isin(truth)].copy()
    dropped = len(recs_df) - len(eval_df)

    recalls, ndcgs, hits, top_flat = [], [], [], []
    tier_recalls: dict[str, list[float]] = {"light": [], "mid": [], "heavy": []}
    warm_r, cold_r = [], []

    for _, row in eval_df.iterrows():
        hh = int(row["household_key"])
        pred = [int(row[c]) for c in rank_cols if pd.notna(row[c])]
        gt = truth[hh]
        r = recall_at_k(pred, gt, k)
        recalls.append(r)
        ndcgs.append(ndcg_at_k(pred, gt, k))
        hits.append(1.0 if set(pred[:k]) & gt else 0.0)
        top_flat.extend(pred[:k])
        tier = task.hh_tier.get(hh)
        if tier in tier_recalls:
            tier_recalls[tier].append(r)
        (warm_r if task.hh_train_products.get(hh) else cold_r).append(r)

    r_mean, r_lo, r_hi = bootstrap_mean_ci(recalls)
    n_mean, n_lo, n_hi = bootstrap_mean_ci(ndcgs)
    coverage = len(set(top_flat) & set(task.candidate_products)) / max(len(task.candidate_products), 1)

    out = {
        "recall_at_k": r_mean, "recall_at_k_ci": (r_lo, r_hi),
        "ndcg_at_k": n_mean, "ndcg_at_k_ci": (n_lo, n_hi),
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
        "coverage": coverage,
        "warm_recall": bootstrap_mean_ci(warm_r)[0] if warm_r else float("nan"),
        "cold_recall": bootstrap_mean_ci(cold_r)[0] if cold_r else float("nan"),
        "n_warm": len(warm_r), "n_cold": len(cold_r),
        "n_households_evaluated": len(eval_df),
        "n_households_dropped": dropped,
        "n_candidate_products": len(task.candidate_products),
        "k": k,
    }
    for t in ("light", "mid", "heavy"):
        out[f"recall_{t}"] = bootstrap_mean_ci(tier_recalls[t])[0] if tier_recalls[t] else float("nan")
        out[f"n_{t}"] = len(tier_recalls[t])
    return out


# ===========================================================================
# Baselines  —  each returns recs_df: household_key, rank_1..rank_k
# ===========================================================================
def _take_k(ordered: list[int], k: int, exclude: list[int], fallback: list[int]) -> list[int]:
    """First k items of `ordered` not in `exclude`, padded from `fallback`."""
    out, seen = [], set(exclude)
    for p in ordered:
        if len(out) >= k:
            break
        if p not in seen:
            out.append(int(p))
            seen.add(p)
    for p in fallback:
        if len(out) >= k:
            break
        if p not in seen:
            out.append(int(p))
            seen.add(p)
    while len(out) < k:                       # last resort (tiny candidate sets only)
        out.append(int(fallback[0] if fallback else ordered[0]))
    return out


def _to_frame(task: PurchaseTask, top_by_hh: dict[int, list[int]], k: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"household_key": hh, **{f"rank_{i + 1}": top_by_hh[hh][i] for i in range(k)}}
        for hh in task.hh_eval
    )


def random_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False,
                    seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cand = np.array(task.candidate_products, dtype=np.int64)
    top_by_hh = {}
    for hh in task.hh_eval:
        pool = cand
        if exclude_seen:
            excl = np.array(task.hh_train_products.get(hh, []), dtype=np.int64)
            pool = cand[~np.isin(cand, excl)] if len(excl) else cand
        pick = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
        while len(pick) < k:
            pick.append(int(cand[0]))
        top_by_hh[hh] = [int(x) for x in pick]
    return _to_frame(task, top_by_hh, k)


def popularity_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False) -> pd.DataFrame:
    top_by_hh = {}
    for hh in task.hh_eval:
        excl = task.hh_train_products.get(hh, []) if exclude_seen else []
        top_by_hh[hh] = _take_k(task.popularity, k, excl, task.popularity)
    return _to_frame(task, top_by_hh, k)


def rfm_popularity_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False) -> pd.DataFrame:
    """Most-purchased products within the household's RFM value tier."""
    top_by_hh = {}
    for hh in task.hh_eval:
        ordered = task.rfm_popularity.get(task.hh_rfm.get(hh, "v3"), task.popularity)
        excl = task.hh_train_products.get(hh, []) if exclude_seen else []
        top_by_hh[hh] = _take_k(ordered, k, excl, task.popularity)
    return _to_frame(task, top_by_hh, k)


def trending_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False) -> pd.DataFrame:
    """Recency-weighted popularity — same list for every household."""
    top_by_hh = {}
    for hh in task.hh_eval:
        excl = task.hh_train_products.get(hh, []) if exclude_seen else []
        top_by_hh[hh] = _take_k(task.trending, k, excl, task.popularity)
    return _to_frame(task, top_by_hh, k)


def wilson_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False) -> pd.DataFrame:
    """Products ranked by the Wilson lower bound of their adoption rate."""
    top_by_hh = {}
    for hh in task.hh_eval:
        excl = task.hh_train_products.get(hh, []) if exclude_seen else []
        top_by_hh[hh] = _take_k(task.wilson, k, excl, task.popularity)
    return _to_frame(task, top_by_hh, k)


def repeat_buy_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False) -> pd.DataFrame:
    """A household's own past purchases, most-often-bought first, padded with
    global popularity. With exclude_seen=True the method has nothing of its own
    to recommend and degrades to popularity — reported for completeness only."""
    top_by_hh = {}
    for hh in task.hh_eval:
        hist = task.hh_train_freq.get(hh, [])
        if exclude_seen:
            top_by_hh[hh] = _take_k(task.popularity, k, hist, task.popularity)
        else:
            top_by_hh[hh] = _take_k(hist, k, [], task.popularity)
    return _to_frame(task, top_by_hh, k)


def last_category_baseline(task: PurchaseTask, k: int = K, exclude_seen: bool = False) -> pd.DataFrame:
    """Rank candidate products by the household's training spend in that product's
    commodity (highest-spend commodity first; global popularity as the in-commodity
    tie-break)."""
    pop_rank = {p: i for i, p in enumerate(task.popularity)}
    by_com: dict[str, list[int]] = defaultdict(list)
    for p in task.candidate_products:
        by_com[task.product_commodity.get(p, "UNKNOWN")].append(p)
    for lst in by_com.values():
        lst.sort(key=lambda p: pop_rank.get(p, 1 << 30))

    top_by_hh = {}
    for hh in task.hh_eval:
        try:
            spend = task.hh_commodity_spend.loc[hh].sort_values(ascending=False)
        except KeyError:
            spend = pd.Series(dtype=float)
        ordered: list[int] = []
        for com, s in spend.items():
            if s > 0:
                ordered.extend(by_com.get(com, []))
        excl = task.hh_train_products.get(hh, []) if exclude_seen else []
        top_by_hh[hh] = _take_k(ordered, k, excl, task.popularity)
    return _to_frame(task, top_by_hh, k)


BASELINES = {
    "random":         random_baseline,          # rung 1  sanity floor
    "popularity":     popularity_baseline,      # rung 2  global demand
    "rfm_popularity": rfm_popularity_baseline,  # rung 3  segment / category popular
    "trending":       trending_baseline,        # rung 4  time sensitivity
    "wilson":         wilson_baseline,          # rung 5  uncertainty-aware
    "repeat_buy":     repeat_buy_baseline,      # +       personal purchase history (grocery standard)
    "last_category":  last_category_baseline,   # +       category-content signal
}


# ===========================================================================
# Pretty printer
# ===========================================================================
def format_result(name: str, exclude_seen: bool, res: dict) -> str:
    tag = "exclude-seen" if exclude_seen else "include-seen"
    rlo, rhi = res["recall_at_k_ci"]
    nlo, nhi = res["ndcg_at_k_ci"]
    return (
        f"=== {name}  [{tag}] ===\n"
        f"  Recall@{res['k']}   {res['recall_at_k']:.4f}  CI [{rlo:.4f}, {rhi:.4f}]\n"
        f"  NDCG@{res['k']}     {res['ndcg_at_k']:.4f}  CI [{nlo:.4f}, {nhi:.4f}]\n"
        f"  HitRate@{res['k']}  {res['hit_rate_at_k']:.4f}\n"
        f"  Coverage      {res['coverage']:.4f}  ({res['n_candidate_products']} candidates)\n"
        f"  Recall/tier   light {res['recall_light']:.4f}  mid {res['recall_mid']:.4f}  heavy {res['recall_heavy']:.4f}\n"
        f"  Warm / cold   {res['warm_recall']:.4f} (n={res['n_warm']})  /  {res['cold_recall']:.4f} (n={res['n_cold']})"
    )
